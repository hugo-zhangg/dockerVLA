# ==============================================================================
# [LEARNING LEVEL]: DEEP DIVE (重点实验)
# [ROLE]: 单条或多条序列过拟合脚本 (Overfitting & Inference Verification Script)
# [LOGIC]: 
#   为了验证模型在小规模数据集下能否彻底记住并复现轨迹（作为架构正确的测试），
#   我们强行提取 N 个完整视频（如 2 episode），并将其循环训练上万次。
#   如果在测试阶段输出的 MSE 小于 0.05，则说明整个多模态到扩散的闭环没有任何 Bug。
#
# 【与 train.py 的差异与必要性解释】：
# 1. 并没有使用 `src.dataset.RLDSDataset`，而是自定义了 `get_episodes_full`：
#    - 必要性：RLDSDataset 默认包含跨 episode 打乱（shuffle）和流式读取逻辑。
#    - 为了在最后能够**按时间顺序绘制真实轨迹与预测轨迹的对比图**，我们必须完整且顺序地加载整个 episode 到内存。
# 2. 数据被强行全部放置在了 GPU 内存中 (`gpu_samples`)：
#    - 必要性：过拟合测试需要对这几百帧数据进行数万次迭代，如果每次都从 DataLoader 经过 CPU 处理并转移到 GPU，I/O 开销极大。缓存在 GPU 可以加速百倍。
# 3. 自定义的梯度累加与随机 Batch 采样循环：
#    - 必要性：由于数据都在 GPU 显存列表里，直接手动 `torch.cat` 拼接 batch 是最快的，省去了 DataLoader 的复杂机制。
# 4. 保留了推理验证与画图代码：
#    - 必要性：直观验证动作空间（Action Dim）的预测曲线是否与真实曲线重合，这是验证架构有效性的核心环节。
# ==============================================================================
# [TOC]:
# - [SKIM]      OverfitConfig (Lines 42-64): 配置超小 Batch/Accumulation 参数以防显存泄漏。
# - [DEEP DIVE] get_episodes_full (Lines 66-135): 强行加载并拆解指定数量的 episode 为多帧独立样本。
# - [DEEP DIVE] main - Train (Lines 137-251): 使用梯度累加机制强行让 BatchSize 膨胀，确保降噪方向稳定。
# - [DEEP DIVE] main - Evaluate (Lines 253-316): 无标签去噪生成全序列并对齐误差，分 Episode 画图。
# ==============================================================================

import os
import sys

# Fix ModuleNotFoundError for 'src' (与 train.py 保持一致)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from PIL import Image
from accelerate import Accelerator
from src.vla_model import VLA_Model
from torch.optim import AdamW
from diffusers.optimization import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt

# 禁用 TF 占用 GPU
tf.config.set_visible_devices([], 'GPU')

# Config (与 train.py 保持一致)
class OverfitConfig:
    def __init__(self, args):
        local_model_path = "models/Qwen2-VL-2B-Instruct"
        if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
             self.vision_backbone = local_model_path
        else:
             self.vision_backbone = "Qwen/Qwen2-VL-2B-Instruct" 
        
        self.use_lora = True
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        
        self.action_dim = 7
        self.pred_horizon = 16
        self.obs_horizon = 1
        
        self.diffusion_steps = 100
        self.use_flow_matching = True
        self.dit_hidden_dim = 512
        self.dit_num_heads = 8
        self.dit_num_layers = 6
        self.dit_dropout = 0.1
        
        self.device = None 

def get_episodes_full(dataset_path, processor, pred_horizon=16, num_episodes=2):
    """
    【与 train.py 不同点】：顺序读取指定数量的 episode。
    必要性：便于后续在不打乱的顺序下绘制整个 episode 的真实与预测轨迹图。
    """
    print(f"Loading {num_episodes} episode(s) from {dataset_path}...")
    builder = tfds.builder_from_directory(dataset_path)
    ds = builder.as_dataset(split='train')
    
    episode_iter = iter(ds)
    
    all_processed_samples = []
    
    for ep_idx in range(num_episodes):
        try:
            episode = next(episode_iter)
        except StopIteration:
            print("Dataset exhausted before reaching requested num_episodes.")
            break
            
        steps = list(episode['steps'])
        print(f"Episode {ep_idx + 1}/{num_episodes} length: {len(steps)}")
        
        if len(steps) < pred_horizon:
            continue
            
        first_obs = steps[0]['observation']
        image_key = next((k for k in first_obs.keys() if 'image' in k or 'rgb' in k), None)
                
        if image_key is None:
            raise ValueError(f"Could not find image key in {first_obs.keys()}")
            
        images = []
        actions = []
        
        raw_instr = steps[0]['language_instruction']
        instruction = raw_instr.numpy().decode('utf-8') if hasattr(raw_instr, 'numpy') else str(raw_instr)
        
        for step in steps:
            img = step['observation'][image_key].numpy() if hasattr(step['observation'][image_key], 'numpy') else step['observation'][image_key]
            act = step['action'].numpy() if hasattr(step['action'], 'numpy') else step['action']
            images.append(img)
            actions.append(act)
            
        actions = np.array(actions, dtype=np.float32)
        valid_indices = range(0, len(actions) - pred_horizon + 1, 1)
        
        for i in valid_indices:
            current_image = Image.fromarray(images[i])
            action_chunk = actions[i : i + pred_horizon]
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt],
                images=[current_image],
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            sample = {
                "pixel_values": inputs['pixel_values'].squeeze(0),
                "image_grid_thw": inputs.get('image_grid_thw', torch.tensor([])).squeeze(0),
                "input_ids": inputs['input_ids'].squeeze(0),
                "attention_mask": inputs['attention_mask'].squeeze(0),
                "actions": torch.from_numpy(action_chunk).float(),
                "episode_idx": ep_idx  # 记录属于哪个 episode 便于画图
            }
            all_processed_samples.append(sample)
            
    return all_processed_samples

def run_inference_and_plot(model, gpu_samples, loss_history, args, step):
    """
    抽取为独立函数：在训练期间每 N 步进行一次验证推理，并绘制当前 loss 和预测对比图。
    这样不用等 10w 步跑完就能实时观测效果。
    """
    print(f"\n--- 2. Inference Phase (Verification at Step {step}) ---")
    model.eval()
    
    os.makedirs("results", exist_ok=True)
    
    # Plot Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Overfitting Training Loss (Step {step})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("results", f"overfit_loss_curve_step_{step}.png"))
    # 同时覆盖一个通用名称方便快速预览
    plt.savefig(os.path.join("results", "overfit_loss_curve_latest.png"))
    plt.close()
    
    # 分 Episode 记录推理结果以绘制连续曲线
    for ep_idx in range(args.episodes):
        ep_samples = [s for s in gpu_samples if s.get('episode_idx') == ep_idx]
        if not ep_samples:
            continue
            
        print(f"Running inference on Episode {ep_idx} ({len(ep_samples)} frames)...")
        plot_gt_actions = []
        plot_pred_actions = []
        mse_list = []
        
        for i, batch in enumerate(ep_samples):
            gt_actions = batch['actions']
            with torch.no_grad():
                output = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    image_grid_thw=batch['image_grid_thw'],
                    gt_actions=None
                )
                pred_actions = output['actions']
                
            gt = gt_actions.cpu().numpy()[0]
            pred = pred_actions.cpu().numpy()[0]
            
            mse = np.mean((gt - pred) ** 2)
            mse_list.append(mse)
            
            plot_gt_actions.append(gt[0]) # 取当前帧的首个动作
            plot_pred_actions.append(pred[0])
            
        avg_mse = np.mean(mse_list)
        print(f"📉 Average MSE for Episode {ep_idx} at Step {step}: {avg_mse:.6f}")
        
        plot_gt_actions = np.array(plot_gt_actions)
        plot_pred_actions = np.array(plot_pred_actions)
        
        plt.figure(figsize=(12, 6))
        plt.plot(plot_gt_actions[:, 0], label=f"Ground Truth (Dim 0) - Ep {ep_idx}", linestyle='dashed', color='blue')
        plt.plot(plot_pred_actions[:, 0], label=f"Predicted (Dim 0) - Ep {ep_idx}", color='red', alpha=0.7)
        plt.xlabel("Frame Index")
        plt.ylabel("Action Value (Dim 0)")
        plt.title(f"Action Trajectory Comparison - Episode {ep_idx} (Step {step})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join("results", f"overfit_action_ep{ep_idx}_step_{step}.png"))
        plt.savefig(os.path.join("results", f"overfit_action_ep{ep_idx}_latest.png"))
        plt.close()
    
    # 恢复训练模式
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000, help="Training steps (过拟合通常需要较大 step)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=2, help="Number of episodes to overfit")
    parser.add_argument("--dataset_path", type=str, default="data/libero_rlds/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/libero_plus_rlds00/libero_plus_mixdata/libero_mix/1.0.0")
    # 【与 train.py 一致的断点保存参数】
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_overfit")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save checkpoint every N steps")
    parser.add_argument("--log_steps", type=int, default=100)
    args = parser.parse_args()

    # 【与 train.py 一致的 ClearML 监控初始化】
    try:
        from clearml import Task
        task = Task.init(project_name='DockerVLA', task_name='Overfit_Validation')
        task.connect(vars(args))
    except ImportError:
        print("ClearML not installed or not configured, skipping ClearML initialization.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        log_with="tensorboard",
        project_dir="runs_overfit"
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers("vla_overfit_experiment")
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    config = OverfitConfig(args)
    config.device = accelerator.device
    
    print(f"🚀 Starting Overfit Test ({args.episodes} Episodes) | Device: {config.device}")

    # 1. Model
    model = VLA_Model(config)
    model.to(config.device)
    
    # 2. Data
    processor = model.vision_encoder.processor
    samples = get_episodes_full(args.dataset_path, processor, config.pred_horizon, args.episodes)
    
    # 【与 train.py 不同点】：显存 Cache
    # 提前移入 GPU，避免在长达 10w step 的过拟合中反复产生 CPU-GPU 拷贝。
    gpu_samples = []
    for s in samples:
        gpu_s = {}
        for k, v in s.items():
            if isinstance(v, torch.Tensor):
                gpu_s[k] = v.to(config.device).unsqueeze(0) # (1, ...)
            else:
                gpu_s[k] = v # e.g. episode_idx
        gpu_samples.append(gpu_s)
        
    print(f"Loaded {len(gpu_samples)} frames to GPU.")

    # 3. Optimizer & Scheduler (与 train.py 保持一致，加入 LR Scheduler)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=args.steps
    )
    
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    
    # 4. Training Loop
    print("\n--- 1. Training Phase (Overfitting) ---")
    model.train()
    
    num_samples = len(gpu_samples)
    loss_history = []
    
    for step in range(1, args.steps + 1):
        model.train()
        accum_loss = 0.0
        
        # 【与 train.py 不同点】：手动采样与拼接
        # 因为数据都在列表中，随机选 index 即可。不用 dataloader.
        for accum_step in range(args.grad_accum):
            with accelerator.accumulate(model):
                batch_indices = np.random.choice(num_samples, size=args.batch_size, replace=False if num_samples >= args.batch_size else True)
                
                pixel_values = torch.cat([gpu_samples[i]['pixel_values'] for i in batch_indices], dim=0)
                input_ids = torch.cat([gpu_samples[i]['input_ids'] for i in batch_indices], dim=0)
                attention_mask = torch.cat([gpu_samples[i]['attention_mask'] for i in batch_indices], dim=0)
                gt_actions = torch.cat([gpu_samples[i]['actions'] for i in batch_indices], dim=0)
                
                grids = [gpu_samples[i]['image_grid_thw'] for i in batch_indices]
                if grids[0].numel() > 0:
                    image_grid_thw = torch.cat(grids, dim=0)
                else:
                    image_grid_thw = torch.tensor([]).to(config.device)
                    
                loss_dict = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_grid_thw=image_grid_thw,
                    gt_actions=gt_actions
                )
                
                loss = loss_dict['loss']
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                accum_loss += loss.item()
                
        avg_loss = accum_loss / args.grad_accum
        loss_history.append(avg_loss)
        
        if step % args.log_steps == 0:
            if accelerator.is_main_process:
                current_lr = lr_scheduler.get_last_lr()[0]
                print(f"[Step {step}/{args.steps}] Loss: {avg_loss:.6f} | LR: {current_lr:.6f}")
                accelerator.log({"train_loss": avg_loss, "lr": current_lr}, step=step)
                
        # 【与 train.py 一致】：保存模型
        if step % args.save_steps == 0:
            if accelerator.is_main_process:
                # 1. 保存模型
                save_path = os.path.join(args.checkpoint_dir, f"vla_overfit_step_{step}.pt")
                print(f"Saving checkpoint to {save_path}...")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), save_path)
                print("Saved.")
                
                # 2. 生成验证图表
                run_inference_and_plot(unwrapped_model, gpu_samples, loss_history, args, step)

    print(f"✅ Final Loss: {loss_history[-1]:.6f}")
    accelerator.end_training()

    # 5. Inference Verification (最后再跑一次确保收尾)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        run_inference_and_plot(unwrapped_model, gpu_samples, loss_history, args, "final")

if __name__ == "__main__":
    main()