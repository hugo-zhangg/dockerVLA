# ==============================================================================
# [LEARNING LEVEL]: DEEP DIVE (核心学习)
# [ROLE]: 扩散策略头（Diffusion Policy Head）。
#         它是机器人的“小脑”，负责把来自视觉语言大模型（大脑）的抽象意图转化为具体的手部动作。
#         本项目中，它基于 DiT (Diffusion Transformer) 而非传统的 UNet。
# ==============================================================================
# [TOC]:
# - [SKIM]      SinusoidalPosEmb (Lines 20-39): 正弦位置编码（时间步嵌入）。
# - [DEEP DIVE] DiT1D Architecture (Lines 41-119): 基于 Transformer 的扩散骨干网络。
# - [SKIM]      DiffusionPolicy Init (Lines 121-162): 整合调度器 (Scheduler) 和网络。
# - [DEEP DIVE] DiffusionPolicy predict_action (Lines 164-192): 预测去噪方向。
# ==============================================================================

import torch
import torch.nn as nn
import math
from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler
from typing import Dict, Any

class SinusoidalPosEmb(nn.Module):
    """
    [SECTION]: Time Step Embedding
    [LEVEL]: SKIM
    [MATH]: Transformer 原理中的位置编码公式: PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    [LOGIC]: 在扩散模型中，不仅序列有空间位置，扩散步数 `t` 也需要嵌入到模型中。
             此类将标量 `t` (例如步数 50) 映射为高维向量，让模型感知当前在去噪的哪个阶段。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiT1D(nn.Module):
    """
    [SECTION]: Diffusion Transformer (1D)
    [LEVEL]: DEEP DIVE
    [LOGIC]: 
      这是一个基于 Transformer Decoder 的扩散模型骨干网络 (DiT, Scalable Diffusion Models with Transformers)。
      传统 Diffusion Policy 使用 CNN-UNet，而我们使用 1D Transformer 处理时序动作。
    [PHYSICS]:
      输入：
        1. x: (B, T, D) 带有噪声的动作序列，例如形状 (B, 16, 7)。代表未来 16 步，每步 7 个关节速度的无序组合。
        2. t: (B,) 扩散时间步，告诉模型当前噪声有多大。
        3. context: (B, L, H) 视觉语言大模型 (Qwen) 提取出的环境特征和指令，作为 Cross-Attention 的 Memory。
      输出：
        预测出的噪声 或 速度向量 (B, T, D)。
    """
    def __init__(self, action_dim, hidden_dim, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input projection: (B, T, action_dim) -> (B, T, hidden_dim)
        self.input_proj = nn.Linear(action_dim, hidden_dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Positional embedding for sequence length
        self.pos_emb = nn.Embedding(512, hidden_dim) # Max seq len 512 is enough for action horizon

        # Transformer Decoder Layers
        # Note: In PyTorch's TransformerDecoderLayer, `memory` is the context (encoder output).
        # We use Pre-Norm (norm_first=True) for better stability.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim*4, 
            dropout=dropout, 
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Final projection
        self.final_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, t, context):
        """
        x: (B, T, action_dim) - Noisy action sequence
        t: (B,) - Timestep
        context: (B, L, hidden_dim) - Visual/Text features from Qwen (projected)
        """
        B, T, _ = x.shape
        
        # 1. Input Projection
        x = self.input_proj(x) # (B, T, H)
        
        # 2. Add Positional Embedding (Sequence Position)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)
        
        # 3. Add Time Embedding (Diffusion Step)
        # t_emb: (B, H) -> (B, 1, H)
        t_emb = self.time_mlp(t).unsqueeze(1)
        x = x + t_emb
        
        # 4. Transformer Decoder
        # tgt: (B, T, H), memory: (B, L, H)
        # No causal mask needed for diffusion because we predict whole sequence at once.
        # We attend to 'context' (VLM features) via Cross-Attention.
        out = self.transformer(tgt=x, memory=context)
        
        # 5. Output Projection
        return self.final_proj(out)

class DiffusionPolicy(nn.Module):
    def __init__(self, config):
        """
        [SECTION]: Diffusion Policy Initialization
        [LEVEL]: SKIM (快速浏览)
        [LOGIC]: 配置噪声调度器 (Scheduler) 和 DiT 网络。
        """
        super().__init__()
        self.config = config
        
        # Action dimensions
        self.action_dim = config.action_dim
        self.obs_horizon = config.obs_horizon
        self.pred_horizon = config.pred_horizon
        
        # [SECTION]: Noise Scheduler
        # [LEVEL]: DEEP DIVE
        # [LOGIC]: Choose between DDPM or Flow Matching (ODE)
        if getattr(config, 'use_flow_matching', False):
            print("🌊 Using FlowMatchEulerDiscreteScheduler (Flow Matching/Rectified Flow)")
            self.scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=config.diffusion_steps,
            )
        else:
            # [MATH]: DDPM (Denoising Diffusion Probabilistic Models)
            self.scheduler = DDPMScheduler(
                num_train_timesteps=config.diffusion_steps,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                prediction_type="epsilon" # Predict noise
            )
        
        # [SECTION]: DiT Backbone
        # [LEVEL]: DEEP DIVE
        # [LOGIC]: 使用 Transformer 替代 UNet
        self.dit = DiT1D(
            action_dim=config.action_dim,
            hidden_dim=getattr(config, 'dit_hidden_dim', 512),
            n_heads=getattr(config, 'dit_num_heads', 8),
            n_layers=getattr(config, 'dit_num_layers', 6),
            dropout=getattr(config, 'dit_dropout', 0.1)
        )

    def predict_action(self, noisy_action, timestep, condition_embedding):
        """
        [SECTION]: Denoising Step (Prediction)
        [LEVEL]: DEEP DIVE
        [LOGIC]: 给定带有噪声的动作序列，预测其中的噪声是什么。
        [ARGS]:
            noisy_action: (B, pred_horizon, action_dim) - 此时刻混乱的动作
            timestep: (B,) - 当前处于扩散过程的第几步 (t)
            condition_embedding: (B, L, hidden_dim) - 视觉语言的提示 (projected)
        [RETURNS]:
            noise_pred: 模型认为“这里面包含的噪声”。
        """
        # Ensure correct shapes
        # noisy_action should be (B, T, D)
        # condition_embedding should be (B, L, H)
        
        # If noisy_action came from UNet legacy code, it might be (B, D, 1, T) or (B, D, T)
        # We need (B, T, D) for Transformer
        if len(noisy_action.shape) == 4: # (B, D, 1, T)
             noisy_action = noisy_action.squeeze(2).transpose(1, 2) # (B, T, D)
        elif len(noisy_action.shape) == 3 and noisy_action.shape[1] == self.action_dim:
             # Heuristic check: if dim 1 is action_dim (e.g. 7), it's likely (B, D, T)
             # But if T=7, this is ambiguous. Assuming standard layout (B, D, T) -> (B, T, D)
             if noisy_action.shape[2] == self.pred_horizon:
                 noisy_action = noisy_action.transpose(1, 2)

        noise_pred = self.dit(noisy_action, timestep, condition_embedding)
        
        return noise_pred
