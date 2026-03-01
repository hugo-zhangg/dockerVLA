# ==============================================================================
# [LEARNING LEVEL]: DEEP DIVE (核心学习)
# [ROLE]: 数据管道 (Data Pipeline)。
#         负责从 RLDS (TFRecord) 读取原始数据集（如图片、机械臂状态、文本指令），
#         并通过 Qwen Processor 将其预处理为 PyTorch 训练所需格式，实现流式读取。
# ==============================================================================
# [TOC]:
# - [SKIM]      RLDSDataset Class (Lines 55+): 继承 IterableDataset。
# - [DEEP DIVE] __iter__ (Lines 76-304): 数据打乱与交错采样机制。
# - [DEEP DIVE] process_sample (Lines 305-362): 核心转换 (TF -> PyTorch) 及 Qwen Text Template 包装。
# - [SKIM]      collate_fn (Lines 364+): Qwen2-VL 特定要求的数据打包 (Flatten & Stack)。
# ==============================================================================

import torch
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import random
import json
import os
import time
from torch.utils.data import IterableDataset, get_worker_info
from PIL import Image

# Prevent TF from using GPU (conflict with PyTorch)
tf.config.set_visible_devices([], 'GPU')

_DEBUG_LOG_PATHS = [
    "/home/jin/projects/dockerVLA/.cursor/debug.log",
    "/workspace/.cursor/debug.log",
    ".cursor/debug.log",
]


def _agent_log(run_id, hypothesis_id, location, message, data):
    payload = {
        "id": f"log_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    for log_path in _DEBUG_LOG_PATHS:
        try:
            dir_name = os.path.dirname(log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            continue

class RLDSDataset(IterableDataset):
    def __init__(self, 
                 dataset_path, 
                 processor, 
                 pred_horizon=16, 
                 action_dim=7, 
                 max_episodes=None):
        """
        [SECTION]: Dataset Initialization
        [LEVEL]: SKIM
        [LOGIC]: 
          - dataset_path: RLDS 数据集路径
          - processor: Qwen2-VL 的 Processor (用于图像/文本预处理)
          - pred_horizon: 预测未来多少步动作
        """
        self.dataset_path = dataset_path
        self.processor = processor
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.max_episodes = max_episodes
        
    def __iter__(self):
        """
        [SECTION]: Data Streaming Loop
        [LEVEL]: DEEP DIVE
        [LOGIC]: 
          为了克服在云端训练单体极大数据集（如 100GB）无法完全载入内存的瓶颈，
          使用 TensorFlow Datasets (TFDS) 读取流式文件。
        [MATH]:
          1. Worker Sharding: 防止多卡/多线程重复读取相同数据。
          2. Interleaved Sampling: 同时挂起多个视频流（如 buffer_size=16），
             通过 `random.randrange` 交替提取帧，从而破坏帧与帧之间强烈的时序相关性，提升泛化能力。
        """
        # [CRITICAL]: Ensure each worker has a different random seed!
        # Otherwise, all workers might sample in the same pattern if fork is used.
        worker_info = get_worker_info()
        if worker_info is not None:
             # Seed = Base Seed + Worker ID
             random.seed(worker_info.seed % (2**32))
             np.random.seed(worker_info.seed % (2**32))
        # region agent log
        _agent_log(
            run_id="pre-fix",
            hypothesis_id="H-seed-worker",
            location="src/dataset.py:__iter__:seed",
            message="Worker info and dataset path at iterator entry",
            data={
                "worker_id": worker_info.id if worker_info else None,
                "num_workers": worker_info.num_workers if worker_info else None,
                "dataset_path": self.dataset_path,
                "max_episodes": self.max_episodes,
            },
        )
        # endregion

        # Load dataset
        builder = tfds.builder_from_directory(self.dataset_path)
        ds = builder.as_dataset(split='train')
        # region agent log
        _agent_log(
            run_id="pre-fix",
            hypothesis_id="H-ds-structure",
            location="src/dataset.py:__iter__:after_as_dataset",
            message="Top-level dataset element spec",
            data={"element_spec": str(ds.element_spec)},
        )
        # endregion
        
        # 1. Worker Sharding
        # [LOGIC]: 如果有多个 Worker，每个 Worker 只读 1/N 的数据。
        worker_info = get_worker_info()
        if worker_info is not None:
            # Note: tfds.shard 需要 num_shards 和 index
            # 如果数据集不支持 shard (比如文件数少于 worker 数)，可能需要 fallback
            try:
                ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)
                print(f"[Worker {worker_info.id}] Sharding applied.")
            except Exception as e:
                print(f"[Worker {worker_info.id}] Sharding failed (maybe dataset too small): {e}")
                # region agent log
                _agent_log(
                    run_id="pre-fix",
                    hypothesis_id="H-shard-failure",
                    location="src/dataset.py:__iter__:shard_exception",
                    message="Sharding failed",
                    data={"error": str(e), "worker_id": worker_info.id},
                )
                # endregion
        
        # Shuffle episodes globally
        ds = ds.shuffle(100)
        
        if self.max_episodes:
            if worker_info is not None:
                # Distribute max_episodes among workers
                local_max = max(1, self.max_episodes // worker_info.num_workers)
                ds = ds.take(local_max)
            else:
                ds = ds.take(self.max_episodes)
            
        # 2. Interleaved Sampling Logic
        # [MATH]: 同时维护 buffer_size 个 episode 的迭代器。
        # 每次随机选一个 iterator 取数据，直到该 episode 耗尽，再补一个新的。
        
        buffer_size = 16 # Number of concurrent episodes to sample from
        episode_iterators = []
        # region agent log
        _agent_log(
            run_id="pre-fix",
            hypothesis_id="H-as-numpy-top-level",
            location="src/dataset.py:__iter__:before_top_level_iter",
            message="Using python iterator for top-level dataset to preserve nested steps dataset",
            data={"element_spec": str(ds.element_spec)},
        )
        # endregion
        dataset_iterator = iter(ds)
        
        # Helper to load a new episode and return its iterator
        def load_next_episode_iterator():
            try:
                episode = next(dataset_iterator)
                # region agent log
                _agent_log(
                    run_id="pre-fix",
                    hypothesis_id="H-episode-structure",
                    location="src/dataset.py:load_next_episode_iterator:after_next_episode",
                    message="Episode object structure sampled",
                    data={
                        "episode_type": str(type(episode)),
                        "episode_keys": list(episode.keys()) if isinstance(episode, dict) else None,
                    },
                )
                # endregion
                
                # Pre-load all steps for this episode to make random access faster or just sequential
                # Since we want to interleave frames, we can't just yield one by one from stream if we want random access *within* episode.
                # BUT, efficient way is: Keep N episodes open, and yield NEXT frame from a RANDOM open episode.
                # This mixes steps from different episodes.
                
                steps_dataset = episode['steps']
                if hasattr(steps_dataset, "as_numpy_iterator"):
                    steps = list(steps_dataset.as_numpy_iterator())
                else:
                    # Some TFDS layouts may already return numpy-like step entries.
                    steps = list(steps_dataset)
                # region agent log
                _agent_log(
                    run_id="pre-fix",
                    hypothesis_id="H-steps-structure",
                    location="src/dataset.py:load_next_episode_iterator:steps_loaded",
                    message="Inner steps structure",
                    data={
                        "steps_dataset_type": str(type(steps_dataset)),
                        "steps_len": len(steps),
                    },
                )
                # endregion
                
                # Check length
                if len(steps) < self.pred_horizon:
                    return None
                
                # Pre-process raw data to avoid repeated parsing
                # (Optimization: Do this lazily if memory is tight, but here we do eagerly for simplicity)
                
                # Find image key
                first_obs = steps[0]['observation']
                image_key = next((k for k in first_obs.keys() if 'image' in k or 'rgb' in k), None)
                
                if image_key is None:
                    return None
                    
                images = []
                actions = []
                
                # Handle instruction
                raw_instr = steps[0]['language_instruction']
                if isinstance(raw_instr, bytes):
                    instruction = raw_instr.decode('utf-8')
                else:
                    instruction = str(raw_instr)
                
                for step in steps:
                    images.append(step['observation'][image_key])
                    actions.append(step['action'])
                    
                actions = np.array(actions, dtype=np.float32)
                
                # Create a generator for this episode that yields processed samples
                def episode_generator():
                    episode_len = len(actions)
                    for i in range(episode_len - self.pred_horizon + 1):
                         current_image = images[i]
                         action_chunk = actions[i : i + self.pred_horizon]
                         if action_chunk.shape[0] == self.pred_horizon:
                             yield current_image, instruction, action_chunk
                
                return episode_generator()
                
            except StopIteration:
                return None
            except Exception as e:
                print(f"Error loading episode: {e}")
                # region agent log
                _agent_log(
                    run_id="pre-fix",
                    hypothesis_id="H-episode-load-exception",
                    location="src/dataset.py:load_next_episode_iterator:exception",
                    message="Exception while loading episode iterator",
                    data={"error": str(e)},
                )
                # endregion
                return None

        # Fill buffer initially
        while len(episode_iterators) < buffer_size:
            ep_iter = load_next_episode_iterator()
            if ep_iter is None:
                # Dataset might be exhausted or empty
                if len(episode_iterators) == 0 and ep_iter is None:
                     # Stop if we can't load even one episode and buffer is empty
                     return 
                break # Stop filling if dataset exhausted
            episode_iterators.append(ep_iter)
            
        # Main Interleaved Loop
        while len(episode_iterators) > 0:
            # Pick a random episode index
            idx = random.randrange(len(episode_iterators))
            
            try:
                # Get next sample from this episode
                data = next(episode_iterators[idx])
                current_image, instruction, action_chunk = data
                
                # Process and Yield
                try:
                    yield self.process_sample(current_image, instruction, action_chunk)
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    # If processing fails, we just skip this frame but keep iterator alive
                    continue
                    
            except StopIteration:
                # This episode is done. Replace it with a new one.
                del episode_iterators[idx]
                new_ep_iter = load_next_episode_iterator()
                if new_ep_iter is not None:
                    episode_iterators.append(new_ep_iter)
        
    def process_sample(self, image, instruction, action_chunk):
        """
        [SECTION]: Qwen Processor
        [LEVEL]: DEEP DIVE
        [LOGIC]: 使用 Qwen 的 Processor 将原始图像和文本转换为模型输入 Tensor。
        """
        # Convert numpy image to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Prepare Qwen inputs
        # Construct prompt with chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        
        # Format text prompt
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding="max_length",
            max_length=512, # Qwen can be long, but limit for efficiency
            return_tensors="pt"
        )
        
        # Extract tensors (remove batch dim 0 since we process single sample)
        # pixel_values: (num_patches, hidden_dim) - flattened
        pixel_values = inputs['pixel_values'] 
        
        # image_grid_thw: (1, 3) -> (3,)
        if 'image_grid_thw' in inputs:
            image_grid_thw = inputs['image_grid_thw'].squeeze(0)
        else:
            # Fallback for older versions or if not present
            image_grid_thw = torch.tensor([]) 
            
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        # Action to Tensor
        action_tensor = torch.from_numpy(action_chunk).float()
        
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "actions": action_tensor
        }

def collate_fn(batch):
    """
    [SECTION]: Custom Collate Function
    [LEVEL]: SKIM
    [LOGIC]: 
      Handle Qwen2-VL's specific batching requirements:
      1. pixel_values are concatenated (flattened).
      2. image_grid_thw are stacked.
      3. input_ids/attention_mask are stacked (padded).
    """
    # 1. Flatten pixel_values from all samples into one long tensor
    pixel_values = torch.cat([x['pixel_values'] for x in batch], dim=0)
    
    # 2. Stack image_grid_thw
    if batch[0]['image_grid_thw'].numel() > 0:
        image_grid_thw = torch.stack([x['image_grid_thw'] for x in batch])
    else:
        image_grid_thw = None
        
    # 3. Stack text inputs (assuming they are padded to max_length in process_sample)
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    
    # 4. Stack actions
    actions = torch.stack([x['actions'] for x in batch])
    
    return {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "actions": actions
    }
