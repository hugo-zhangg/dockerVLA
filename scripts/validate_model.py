# ==============================================================================
# [LEARNING LEVEL]: SKIM
# [ROLE]: Validation script for VLA_Model Forward Pass.
# ==============================================================================
# [TOC]:
# - [SKIM]      Validation Main (Lines 15+): Run forward pass with random tensors.
# ==============================================================================

import sys
import os
import torch
import torch.nn as nn
from transformers import AutoProcessor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vla_model import VLA_Model
from src.config import VLAConfig

def validate_model():
    print(">>> Starting Model Validation...")
    
    # 1. Config
    config = VLAConfig()
    config.batch_size = 2
    config.pred_horizon = 16
    config.action_dim = 7
    config.dit_hidden_dim = 512
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {config.device}")
    
    # 2. Initialize Model
    print("Initializing VLA_Model (Qwen2-VL + DiT)...")
    try:
        model = VLA_Model(config).to(config.device)
        print("✅ Model Initialized Successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return

    # 3. Create Dummy Inputs
    print("Creating dummy inputs...")
    B = 2
    T = 16
    D = 7
    Seq_Len = 128
    
    # Qwen2-VL Input Shapes
    # Pixel Values: Typically flattened patches.
    # Let's assume standard image size (224x224) -> patches
    # If using dynamic resolution, shape varies.
    # We simulate (Total_Patches, Hidden_Dim) -> e.g., (B * N_patches, 1280)
    # Qwen2-VL hidden size is usually 1280 or 3584 depending on size. Let's use 1280 for 2B, 3584 for 7B.
    # But wait, VisionEncoder expects raw pixel values or processed features?
    # It expects what processor outputs: 'pixel_values'
    
    # Wait, for Qwen2-VL, pixel_values input to model forward is actually raw pixel values?
    # No, it's processed tensor.
    # Let's check VisionEncoder.forward signature: (pixel_values, input_ids, attention_mask, image_grid_thw)
    
    # Simulate: 2 images, 256x256
    # Patch size 14x14 -> (18x18) patches -> 324 patches per image
    # Total patches = 648
    # pixel_values shape for Qwen2-VL is (N_patches, Patch_Dim) -> (648, 1176) (3*14*14*2) ? 
    # Qwen2-VL uses 3 channels * 14 * 14 patches. = 588. But it varies.
    # Let's try to run processor on random image to get real shape.
    
    from PIL import Image
    import numpy as np
    
    processor = AutoProcessor.from_pretrained(config.vision_backbone, trust_remote_code=True)
    
    dummy_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "Move the red block."},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(
        text=[text_prompt] * B,
        images=[dummy_image] * B,
        padding=True,
        return_tensors="pt"
    )
    
    pixel_values = inputs['pixel_values'].to(config.device)
    input_ids = inputs['input_ids'].to(config.device)
    attention_mask = inputs['attention_mask'].to(config.device)
    image_grid_thw = inputs['image_grid_thw'].to(config.device)
    
    print(f"Dummy Pixel Values: {pixel_values.shape}")
    print(f"Dummy Input IDs: {input_ids.shape}")
    
    # Dummy GT Actions
    gt_actions = torch.randn(B, T, D).to(config.device)
    
    # 4. Forward Pass (Training Mode)
    print("\nRunning Forward Pass (Training Mode)...")
    try:
        output = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            gt_actions=gt_actions
        )
        
        loss = output['loss']
        print(f"✅ Training Step Success! Loss: {loss.item()}")
        
    except Exception as e:
        print(f"❌ Training Step Failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. Forward Pass (Inference Mode)
    print("\nRunning Forward Pass (Inference Mode)...")
    try:
        output = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            gt_actions=None
        )
        
        pred_actions = output['actions']
        print(f"✅ Inference Step Success! Actions Shape: {pred_actions.shape}")
        
        expected_shape = (B, T, D)
        if pred_actions.shape == expected_shape:
            print("✅ Shape matches expected.")
        else:
            print(f"❌ Shape mismatch! Expected {expected_shape}")
            
    except Exception as e:
        print(f"❌ Inference Step Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_model()
