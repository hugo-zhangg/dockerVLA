# ==============================================================================
# [LEARNING LEVEL]: SKIM
# [ROLE]: Evaluation and Visualization Script. Loads a checkpoint, predicts actions, and plots them against ground truth.
# ==============================================================================
# [TOC]:
# - [SKIM] plot_curves (Lines 23-86): Configure model, load data, run inference, and plot action curves.
# ==============================================================================

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

# Fix ModuleNotFoundError for 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.vla_model import VLA_Model
from scripts.train import TrainConfig

def plot_curves():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained .pt checkpoint")
    args = parser.parse_args()

    # 1. 临时配置环境与模型
    # 我们构造一个假的 args 喂给 TrainConfig
    class DummyArgs: pass
    dummy = DummyArgs()
    dummy.checkpoint_dir = ""
    config = TrainConfig(dummy)
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VLA_Model(config).to(config.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=config.device))
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint}")

    # 2. 构造一个假数据或从您提取的单个 Episode 中取出数据
    # 这里需要替换为您真实 dataset 中取出的一个 batch 的一条数据
    # 例如：
    # dataset = RLDSDataset(...)
    # dataloader = DataLoader(dataset, batch_size=1, ...)
    # batch = next(iter(dataloader))
    
    # 【占位】假设 batch 是拿到的真实数据
    # bs = 1
    # gt_actions = batch['actions'][0].cpu().numpy()  # 形状: (16, 7)
    # predicted_actions = model.inference(
    #     pixel_values=batch['pixel_values'],
    #     input_ids=batch['input_ids'],
    #     attention_mask=batch['attention_mask'],
    #     image_grid_thw=batch['image_grid_thw']
    # )[0].cpu().numpy() # 形状: (16, 7)
    
    # 模拟数据为了演示绘图（请将下面换成实际网络输出的 numpy 数组）
    horizon = config.pred_horizon
    gt_actions = np.sin(np.linspace(0, 3, horizon))[:, None] * np.ones((horizon, 7))
    predicted_actions = gt_actions + np.random.normal(0, 0.05, (horizon, 7))

    # 3. 绘制 7 个动作维度（6 自由度 + 1 夹爪）的对比曲线
    fig, axs = plt.subplots(7, 1, figsize=(10, 15), sharex=True)
    action_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper']

    for i in range(7):
        axs[i].plot(gt_actions[:, i], label='Ground Truth', color='blue', linestyle='--')
        axs[i].plot(predicted_actions[:, i], label='Predicted', color='red', alpha=0.8)
        axs[i].set_ylabel(action_names[i])
        axs[i].legend(loc="upper right")
        axs[i].grid(True)

    axs[-1].set_xlabel('Time Step (within horizon)')
    plt.suptitle("Action Trajectory: Prediction vs Ground Truth", fontsize=16)
    plt.tight_layout()
    
    # 保存图片
    save_path = "action_curves.png"
    plt.savefig(save_path)
    print(f"Curves saved to {save_path}")

if __name__ == "__main__":
    with torch.no_grad():
        plot_curves()