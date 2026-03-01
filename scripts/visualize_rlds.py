# ==============================================================================
# [LEARNING LEVEL]: SKIM (快速浏览)
# [ROLE]: RLDS 数据可视化工具。
#         [LOGIC]: 读取 TFRecord 格式数据 -> 解析 Example -> 提取图像和动作 -> 保存为 sample_rlds.jpg。
#         这是适配 RLDS (Reinforcement Learning Datasets) 格式的新版可视化脚本。
# ==============================================================================
# [TOC]:
# - [SKIM]      Visualization Main Loop (Lines 24-121): Iterate episodes.
#   - [SKIM]      TFDS Loading (Lines 33-55): Load dataset from disk.
#   - [DEEP DIVE] Data Inspection (Lines 70-109): Check Image & Action format (Critical for debugging).
# ==============================================================================

import tensorflow_datasets as tfds
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt

# 禁用 GPU 以避免 TF 占用显存干扰 PyTorch
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

def visualize_rlds(dataset_path, num_episodes=1):
    """
    [SECTION]: Visualization Main Loop
    [LEVEL]: SKIM
    [LOGIC]: 
      1. 使用 tfds.builder_from_directory 加载本地数据。
      2. 遍历 episode，再遍历 step。
      3. 提取 observation 中的图像和 action 并打印。
    """
    print(f"Reading RLDS dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Path not found: {dataset_path}")
        return

    try:
        # Load dataset using TFDS builder from directory
        # [SECTION]: TFDS Loading
        # [LEVEL]: SKIM
        # [LOGIC]: RLDS 数据集通常是标准的 TFDS 数据集。我们需要指向包含 dataset_info.json 的目录。
        
        # Construct the builder
        builder = tfds.builder_from_directory(dataset_path)
        
        print(f"Dataset Info:")
        print(f" - Name: {builder.name}")
        print(f" - Version: {builder.version}")
        print(f" - Description: {builder.info.description}")
        print(f" - Features: {builder.info.features}")
        
        # Load the dataset (train split)
        ds = builder.as_dataset(split='train')
        
        # Iterate through episodes
        for i, episode in enumerate(ds.take(num_episodes)):
            print(f"\n--- Episode {i+1} ---")
            
            # RLDS episodes are datasets themselves (steps)
            steps = list(episode['steps'].as_numpy_iterator())
            print(f" - Length: {len(steps)} steps")
            
            # Visualize a random step
            if len(steps) > 0:
                step_idx = np.random.randint(0, len(steps))
                step = steps[step_idx]
                
                print(f" - Inspecting Step {step_idx}:")
                
                # Check for images in observation
                # [SECTION]: Data Inspection
                # [LEVEL]: DEEP DIVE (数据格式)
                # [PHYSICS]: 检查 observation 中是否有 RGB 图像，以及 action 的维度。
                #            这是确认数据是否符合模型输入的关键步骤。
                obs = step.get('observation', {})
                print(f"   - Observation keys: {list(obs.keys())}")
                
                # Try to find an image
                image_keys = [k for k in obs.keys() if 'image' in k or 'rgb' in k]
                if image_keys:
                    img_key = image_keys[0]
                    image = obs[img_key]
                    print(f"   - Found image: {img_key}, Shape: {image.shape}, Type: {image.dtype}")
                    
                    # Convert to BGR for OpenCV
                    if image.shape[-1] == 3:
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite("sample_rlds.jpg", image_bgr)
                        print(f"   - Saved sample frame to sample_rlds.jpg")
                    else:
                        print(f"   - Image shape {image.shape} not standard RGB.")
                else:
                    print("   - No image found in observation.")
                
                # Check Action
                action = step.get('action', None)
                if action is not None:
                    print(f"   - Action: {action}")
                    print(f"   - Action Shape: {action.shape}")
                else:
                    print("   - No action found in step.")
                    
                # Check Instruction/Language
                lang = step.get('language_instruction', None)
                if lang is not None:
                    print(f"   - Instruction: {lang}")
            
    except Exception as e:
        print(f"Error reading RLDS dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RLDS Dataset")
    parser.add_argument("--path", type=str, required=True, help="Path to the RLDS dataset directory (containing dataset_info.json)")
    args = parser.parse_args()
    
    visualize_rlds(args.path)
