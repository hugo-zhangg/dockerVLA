# ==============================================================================
# [LEARNING LEVEL]: DEEP DIVE (核心学习)
# [ROLE]: VLA 模型的主体架构。将视觉感知（Vision Encoder）与动作生成（Diffusion Head）结合。
#         这是整个项目的“大脑”和“中枢”。
# ==============================================================================
# [TOC]:
# - [SKIM]      Model Initialization (Lines 18-47): Load sub-modules (Vision, Projector, DiT).
# - [DEEP DIVE] Forward Pass (Lines 49+): Main logic.
#   - [DEEP DIVE] Training Loop (Lines 74-124): Noise prediction & Loss.
#   - [DEEP DIVE] Inference Loop (Lines 128-158): Denoising loop (Action generation).
# ==============================================================================

import torch
import torch.nn as nn
from src.vision_encoder import VisionEncoder
from src.diffusion_policy import DiffusionPolicy

class VLA_Model(nn.Module):
    def __init__(self, config):
        """
        [SECTION]: Model Initialization
        [LEVEL]: SKIM (快速浏览)
        [LOGIC]: 初始化三个核心模块：
          1. VisionEncoder (Qwen2-VL): 提取视觉语言特征 (B, L, H_qwen)
          2. Projector (MLP): 维度对齐 H_qwen -> H_dit
          3. DiffusionPolicy (DiT): 生成动作序列
        """
        super().__init__()
        self.config = config
        self.device = config.device
        
        # 1. Initialize Vision Backbone
        self.vision_encoder = VisionEncoder(config)
        
        # 2. Initialize Projector (Bridge)
        # [LOGIC]: Qwen 的 Hidden Size 很大 (e.g. 3584)，直接喂给 DiT 会太重。
        #          我们用一个 Linear 层把它降维到 DiT 的 Hidden Size (e.g. 512)。
        self.projector = nn.Linear(
            self.vision_encoder.hidden_size, 
            getattr(config, 'dit_hidden_dim', 512)
        ).to(self.device)
        
        # 3. Initialize Diffusion Head (Action Expert)
        self.diffusion_head = DiffusionPolicy(config).to(self.device)
        
        # [MATH]: Mean Squared Error (MSE)
        self.loss_fn = nn.MSELoss().to(self.device)

    def forward(self, pixel_values, input_ids, attention_mask, image_grid_thw=None, gt_actions=None):
        """
        [SECTION]: Forward Pass (Training & Inference)
        [LEVEL]: DEEP DIVE (核心学习)
        [LOGIC]: 
          1. Qwen 提取全序列特征 -> Projector 降维 -> Condition
          2. 如果有真实动作 -> 训练 DiT 预测噪声 -> 计算 Loss
          3. 如果没有真实动作 -> DiT 逐步去噪 -> 生成动作
        """
        batch_size = input_ids.shape[0]
        
        # 1. Vision-Language Encoding
        # Output: (B, Seq_Len, H_qwen)
        vision_outputs = self.vision_encoder(pixel_values, input_ids, attention_mask, image_grid_thw)
        
        # Cast to float32 (or whatever dtype projector is) because Qwen is FP16/BF16 but DiT is usually FP32
        # Also ensure it is on the correct device for the projector
        vision_outputs = vision_outputs.to(dtype=self.projector.weight.dtype, device=self.projector.weight.device)
        
        # 2. Projection
        # Output: (B, Seq_Len, H_dit)
        # [PHYSICS]: 这就是 DiT 的 "Context"，包含了图片中物体的空间位置信息和文本指令的语义信息。
        condition_embedding = self.projector(vision_outputs)
        
        # 3. Diffusion Process (Training)
        if gt_actions is not None:
            # [SECTION]: Training Loop - Noise Prediction
            
            # gt_actions shape: (B, Pred_Horizon, Action_Dim) -> e.g., (B, 16, 7)
            # DiT expects (B, T, D), so we keep it as is.
            
            # [MATH]: Gaussian Noise ε ~ N(0, I)
            noise = torch.randn_like(gt_actions).to(self.device)
            
            # Sample random timesteps t ~ Uniform(0, T_max)
            timesteps = torch.randint(
                0, self.diffusion_head.scheduler.config.num_train_timesteps, 
                (batch_size,), device=self.device
            ).long()
            
            # Add noise to GT actions (Forward Process)
            if getattr(self.config, 'use_flow_matching', False):
                # Diffusers FlowMatch uses add_noise differently sometimes,
                # but basically x_t = (1-t) * data + t * noise
                # Let's use the scheduler's scale
                bsz = gt_actions.shape[0]
                
                # Flow matching usually works with continuous timesteps [0, 1] or [0, 1000]
                # Diffusers scheduler config:
                sigmas = self.diffusion_head.scheduler.sigmas.to(self.device)
                
                # Get sigma for each timestep
                sigma = sigmas[timesteps].flatten()
                while len(sigma.shape) < len(gt_actions.shape):
                    sigma = sigma.unsqueeze(-1)
                    
                # Flow matching interpolation: x_t = (1 - sigma) * x_0 + sigma * x_1 (noise)
                # target is vector field v_t = x_1 - x_0
                noisy_actions = (1.0 - sigma) * gt_actions + sigma * noise
                target = noise - gt_actions
                
                noise_pred = self.diffusion_head.predict_action(
                    noisy_actions, timesteps, condition_embedding
                )
                
                loss = self.loss_fn(noise_pred, target)
            else:
                # Standard DDPM
                noisy_actions = self.diffusion_head.scheduler.add_noise(
                    gt_actions, noise, timesteps
                )
                noise_pred = self.diffusion_head.predict_action(
                    noisy_actions, timesteps, condition_embedding
                )
                loss = self.loss_fn(noise_pred, noise)
                
            return {"loss": loss}
            
        # 4. Inference (Generating Actions)
        else:
            # [SECTION]: Inference Loop - Denoising
            
            # Initialize random noise (B, T, D)
            action_shape = (batch_size, self.config.pred_horizon, self.config.action_dim)
            latents = torch.randn(action_shape, device=self.device)
            
            # Set scheduler timesteps
            self.diffusion_head.scheduler.set_timesteps(self.config.diffusion_steps)
            
            # Denoising loop
            for t in self.diffusion_head.scheduler.timesteps:
                # Predict noise or velocity
                t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                
                noise_pred = self.diffusion_head.predict_action(
                    latents, t_batch, condition_embedding
                )

                # Compute previous sample x_t -> x_t-1
                if getattr(self.config, 'use_flow_matching', False):
                    latents = self.diffusion_head.scheduler.step(
                        noise_pred, t, latents
                    ).prev_sample
                else:
                    latents = self.diffusion_head.scheduler.step(
                        noise_pred, t, latents
                    ).prev_sample
            
            # Final actions: (B, T, D)
            return {"actions": latents}
