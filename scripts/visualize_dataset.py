# ==============================================================================
# [LEARNING LEVEL]: SKIM (快速浏览)
# [ROLE]: 数据可视化工具。用于检查 HDF5 数据集文件的结构和内容。
#         [LOGIC]: 读取 .hdf5 -> 打印 Key 结构 -> 提取一张图片和动作 -> 保存为 sample_image.jpg。
#         研究者应运行它来确认数据是否正确下载，不需要深究 HDF5 读取细节。
# ==============================================================================
# [TOC]:
# - [SKIM]      HDF5 Loading (Lines 21-41): Open file and inspect structure.
# - [SKIM]      Image Visualization (Lines 53-91): Extract RGB image and save as JPG.
# - [DEEP DIVE] Action Inspection (Lines 97-104): Verify action dimensions (Crucial for model config).
# ==============================================================================

import h5py
import numpy as np
import cv2
import random
import os
import argparse
import json

def visualize(file_path):
    print(f"Reading dataset: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with h5py.File(file_path, 'r') as f:
            # Print top-level structure
            print(f"HDF5 File Structure:")
            print(f" - Keys: {list(f.keys())}")
            print(f" - Attributes: {list(f.attrs.keys())}")

            # Usually datasets are stored under 'data' group for robosuite/libero
            if 'data' not in f:
                print("Error: 'data' group not found in HDF5 file.")
                return

            demos = list(f['data'].keys())
            print(f"Number of demonstrations: {len(demos)}")
            
            if not demos:
                print("No demonstrations found.")
                return

            # Select a random demo to inspect
            demo_key = random.choice(demos)
            demo_group = f['data'][demo_key]
            print(f"\nSelected Random Demo: {demo_key}")
            print(f" - Keys in demo group: {list(demo_group.keys())}")

            # Get observations
            if 'obs' in demo_group:
                obs_group = demo_group['obs']
                print(f" - Observation keys: {list(obs_group.keys())}")
                
                # Find an image key (agentview_rgb is common)
                image_keys = [k for k in obs_group.keys() if 'rgb' in k]
                # If no explicit rgb key, look for generic 'image' keys (excluding depth)
                if not image_keys:
                     image_keys = [k for k in obs_group.keys() if 'image' in k and 'depth' not in k]

                if image_keys:
                    img_key = image_keys[0] # Take first available image
                    print(f" - Visualizing image key: {img_key}")
                    images = obs_group[img_key][()] # Load image data
                    
                    # Select a random frame
                    frame_idx = random.randint(0, len(images) - 1)
                    image = images[frame_idx]
                    
                    # [SECTION]: Image Preprocessing
                    # [LEVEL]: SKIM
                    # [LOGIC]: 处理通道顺序 (H, W, C) 和颜色格式 (RGB -> BGR for OpenCV)
                    if image.shape[0] == 3 and image.shape[2] != 3: 
                        image = np.transpose(image, (1, 2, 0))
                    
                    # Convert RGB to BGR for OpenCV
                    # Robosuite/Libero usually stores images as RGB uint8
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Save image
                    output_path = "sample_image.jpg"
                    cv2.imwrite(output_path, image_bgr)
                    print(f"\nSaved sample frame {frame_idx} to {output_path}")
                else:
                    print(" - No RGB images found to visualize.")
            else:
                print(" - 'obs' group not found.")

            # Get Action
            # [SECTION]: Action Inspection
            # [LEVEL]: DEEP DIVE (数据格式)
            # [PHYSICS]: 这里的 Action 是关节速度或位置。通常是 7维 (x,y,z,r,p,y,gripper)。
            #            确认 Shape 是否为 (Time, 7) 很重要。
            if 'actions' in demo_group:
                actions = demo_group['actions'][()]
                print(f"\nAction Info:")
                print(f" - Shape: {actions.shape}")
                if 'frame_idx' in locals():
                     print(f" - Action at frame {frame_idx}: {actions[frame_idx]}")
            else:
                print(" - Actions not found in demo group.")

            # Get Instruction / Problem Info
            print(f"\nInstruction Info:")
            # Instructions are often stored in attributes of the file or the data group
            instruction_found = False
            
            # Check file attributes
            if 'problem_info' in f.attrs:
                info = f.attrs['problem_info']
                print(f" - Problem Info (Attribute): {info}")
                instruction_found = True
            
            if 'env_args' in f.attrs:
                env_args = f.attrs['env_args']
                print(f" - Env Args (Attribute): {env_args}")
                instruction_found = True

            # Sometimes stored as dataset
            if 'problem_info' in f:
                print(f" - Problem Info (Dataset): {f['problem_info'][()]}")
                instruction_found = True

            if not instruction_found:
                print(" - No specific instruction/problem info found in standard locations.")

    except Exception as e:
        print(f"Error visualizing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LIBERO Dataset")
    parser.add_argument("--file", type=str, required=True, help="Path to the hdf5 dataset file")
    args = parser.parse_args()
    
    visualize(args.file)
