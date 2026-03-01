# ==============================================================================
# [LEARNING LEVEL]: SKIM
# [ROLE]: Validation script for RLDSDataset and Data Pipeline.
# ==============================================================================
# [TOC]:
# - [SKIM]      Validation Main (Lines 15+): Run validation loop.
# ==============================================================================

import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import RLDSDataset, collate_fn
from src.config import VLAConfig

def validate_dataset():
    print(">>> Starting Dataset Validation...")
    
    # 1. Config
    # Use a dummy path or real path depending on environment
    # Update this path to your actual RLDS location
    dataset_path = "data/libero_rlds/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/libero_plus_rlds00/libero_plus_mixdata/libero_mix/1.0.0"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path not found: {dataset_path}")
        return

    # 2. Initialize Processor (Qwen2-VL)
    # Load from config or hardcoded
    config = VLAConfig()
    model_id = config.vision_backbone # Use config value (2B or 7B)
    print(f"Loading processor from {model_id}...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load processor: {e}")
        return

    # 3. Initialize Dataset
    print("Initializing RLDSDataset...")
    dataset = RLDSDataset(
        dataset_path=dataset_path,
        processor=processor,
        pred_horizon=16,
        action_dim=7,
        max_episodes=5 # Check small number of episodes
    )
    
    # 4. Initialize DataLoader
    loader = DataLoader(
        dataset, 
        batch_size=2, # Check batching
        collate_fn=collate_fn
    )
    
    # 5. Fetch one batch
    print("Fetching one batch...")
    try:
        batch = next(iter(loader))
        
        # 6. Verify Shapes
        print("\n=== Validation Results ===")
        print(f"Pixel Values Shape: {batch['pixel_values'].shape}")
        # Expected: (Total_Patches, Hidden_Dim) e.g., (N, 1280) or similar
        
        if batch['image_grid_thw'] is not None:
            print(f"Image Grid THW: {batch['image_grid_thw'].shape}")
            # Expected: (Batch_Size, 3)
            print(f"Image Grid Sample: {batch['image_grid_thw']}")
        else:
            print("Image Grid THW is None (Might be issue for Qwen2-VL)")
            
        print(f"Input IDs Shape: {batch['input_ids'].shape}")
        # Expected: (Batch_Size, Seq_Len)
        
        print(f"Attention Mask Shape: {batch['attention_mask'].shape}")
        # Expected: (Batch_Size, Seq_Len)
        
        # [VERIFICATION]: Decode Input IDs to check instruction
        print("\n=== Instruction Verification ===")
        decoded_text = processor.decode(batch['input_ids'][0], skip_special_tokens=False)
        print(f"Decoded Input[0]:\n{decoded_text}")
        # Should see something like: <|im_start|>user\n<|vision_start|>...<|vision_end|>Instruction...
        
        # [VERIFICATION]: Check Pixel Values
        print("\n=== Image Verification ===")
        pixel_val = batch['pixel_values']
        print(f"Pixel Values Range: [{pixel_val.min():.2f}, {pixel_val.max():.2f}]")
        print(f"Pixel Values Mean: {pixel_val.mean():.2f}")
        if pixel_val.abs().sum() > 0:
             print("✅ Image data contains information (not all zeros).")
        else:
             print(f"❌ Image data is empty!")

        print(f"Actions Shape: {batch['actions'].shape}")
        # Expected: (Batch_Size, Pred_Horizon, Action_Dim) -> (2, 16, 7)
        
        expected_action_shape = (2, 16, 7)
        if batch['actions'].shape == expected_action_shape:
             print("✅ Action Shape matches expected.")
        else:
             print(f"❌ Action Shape mismatch! Expected {expected_action_shape}")

        print(">>> Dataset Validation Completed!")
        
    except Exception as e:
        print(f"❌ Error during dataloading: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_dataset()
