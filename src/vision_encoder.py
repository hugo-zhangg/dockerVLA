# ==============================================================================
# [LEARNING LEVEL]: SKIM
# [ROLE]: Vision Encoder wrapper for Qwen2-VL. Extracts features from images and texts.
# ==============================================================================
# [TOC]:
# - [SKIM]      VisionEncoder.__init__ (Lines 17-61): Load model and apply LoRA.
# - [SKIM]      VisionEncoder.forward (Lines 63-95): Extract features.
# ==============================================================================

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
from typing import Optional, List, Dict, Any

class VisionEncoder(nn.Module):
    def __init__(self, config):
        """
        [SECTION]: Qwen2-VL Wrapper Initialization
        [LEVEL]: SKIM (快速浏览)
        [LOGIC]: 加载 HuggingFace 模型，并配置 LoRA (Low-Rank Adaptation) 以减少显存占用。
        """
        super().__init__()
        self.config = config
        
        # Load Pretrained Qwen2-VL Model
        print(f"Loading Qwen2-VL from {config.vision_backbone}...")
        
        # When using accelerate with multiple GPUs, we MUST NOT use device_map="auto"
        # as it conflicts with DDP. We should let accelerate handle device placement.
        # But for FP16 loading, we can keep torch_dtype.
        import os
        is_multi_gpu = int(os.environ.get("WORLD_SIZE", 1)) > 1
        device_map = None if is_multi_gpu else "auto"
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.vision_backbone,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(config.vision_backbone, trust_remote_code=True)
        
        # Apply LoRA if specified
        # [MATH]: LoRA 原理: W' = W + BA，其中 B, A 是低秩矩阵。
        #         这允许我们在不更新巨大 W 的情况下微调模型。
        if config.use_lora:
            print("Applying LoRA Configuration...")
            peft_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Target Qwen Attention layers
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            
        # We might need to access hidden states later for the diffusion head
        # Usually, VLA uses the last hidden state of the LLM as conditioning for diffusion
        # Qwen2VLConfig stores hidden_size in text_config or vision_config
        if hasattr(self.model.config, "hidden_size"):
            self.hidden_size = self.model.config.hidden_size
        elif hasattr(self.model.config, "text_config"):
            self.hidden_size = self.model.config.text_config.hidden_size
        else:
            # Fallback or raise error
            raise ValueError("Could not find hidden_size in model config")

    def forward(self, pixel_values, input_ids, attention_mask, image_grid_thw=None):
        """
        [SECTION]: Feature Extraction
        [LEVEL]: SKIM (快速浏览) -> 稍微关注一下输出是什么
        [SYNTAX]: forward 接收图像和文本输入。
        [PHYSICS]: 将外部世界的像素信号和人类指令，转化为计算机内部的抽象向量表示。
        """
        # Output from QwenCollateFn via processor is already standard tensor dict.
        # No need to handle lists here anymore.
        # Prepare inputs dict
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "output_hidden_states": True,
            "return_dict": True
        }
        
        # Add image_grid_thw if provided (required for Qwen2-VL dynamic resolution)
        if image_grid_thw is not None and image_grid_thw.numel() > 0:
            inputs["image_grid_thw"] = image_grid_thw
            
        outputs = self.model(**inputs)
        
        # Last hidden state: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.hidden_states[-1] 
        
        # [SECTION]: Global Pooling Removed (Returning Full Sequence)
        # [LEVEL]: DEEP DIVE (细节)
        # [LOGIC]: 
        #   Previously we used global average pooling.
        #   Now we return the full sequence (B, Seq_Len, Hidden) to allow DiT
        #   to attend to specific visual/text tokens via Cross-Attention.
        
        return last_hidden_state

    def get_processor(self):
        return self.processor
