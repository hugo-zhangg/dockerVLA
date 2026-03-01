# ==============================================================================
# [LEARNING LEVEL]: SKIM (快速浏览)
# [ROLE]: 项目配置文件。定义了模型参数、训练参数和数据路径。
#         [LOGIC]: 使用 Python dataclass 管理配置，方便在代码中以 cfg.xxx 访问。
# ==============================================================================
# [TOC]:
# - [SKIM]      Model Architecture (Lines 19-25): VLM backbone settings.
# - [DEEP DIVE] Diffusion Head Parameters (Lines 27-36): Action dim, Horizon steps (Physics related).
# - [DEEP DIVE] DiT Architecture (Lines 38-43): Transformer-based Action Expert Config.
# - [SKIM]      Training & Data (Lines 45-57): Batch size, LR, paths.
# ==============================================================================

import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class VLAConfig:
    # [SECTION]: Model Architecture
    # [LEVEL]: SKIM
    vision_backbone: str = "Qwen/Qwen2-VL-2B-Instruct"  # Pretrained model path (Use 2B for faster debug/train)
    use_lora: bool = True  # Whether to use LoRA fine-tuning
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # [SECTION]: Diffusion Head Parameters
    # [LEVEL]: DEEP DIVE (关键参数)
    # [PHYSICS]: 
    #   action_dim=7: 对应机械臂的 7 个自由度 (DoF) - 通常是 (x, y, z, roll, pitch, yaw) + 夹爪开合。
    #   pred_horizon=16: 模型一次性预测未来 16 步的动作。
    #   obs_horizon=1: 模型只看当前这一帧图像 (没有使用历史帧作为输入)。
    action_dim: int = 7  # (x, y, z, roll, pitch, yaw, gripper)
    diffusion_steps: int = 100 # [MATH]: 扩散过程的时间步数 T
    obs_horizon: int = 1  # Number of past frames to consider
    pred_horizon: int = 16 # Number of future actions to predict
    
    # [SECTION]: DiT Architecture
    # [LEVEL]: DEEP DIVE (DiT 关键参数)
    dit_hidden_dim: int = 512
    dit_num_layers: int = 6
    dit_num_heads: int = 8
    dit_dropout: float = 0.1
    
    # [SECTION]: Training Hyperparameters
    # [LEVEL]: SKIM
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 10
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # [SECTION]: Data Configuration
    # [LEVEL]: SKIM
    dataset_name: str = "libero_spatial"
    max_seq_len: int = 2048 # Max context length for LLM
