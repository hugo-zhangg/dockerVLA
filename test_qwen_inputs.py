# ==============================================================================
# [LEARNING LEVEL]: IGNORE
# [ROLE]: Temporary debugging script for collate_fn shapes.
# ==============================================================================

import torch
import json
import time

log_path = "/root/.cursor/debug.log"

try:
    from src.dataset import collate_fn
    
    # Simulate a dummy batch where image_grid_thw becomes a 0-d tensor
    dummy_batch = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.randn(10, 1176),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "actions": torch.randn(16, 7)
        },
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "pixel_values": torch.randn(15, 1176),
            "image_grid_thw": torch.tensor([[1, 3, 5]]),
            "actions": torch.randn(16, 7)
        }
    ]
    
    batch = collate_fn(dummy_batch)
    
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(time.time()*1000)}_H11",
            "timestamp": int(time.time()*1000),
            "location": "test_qwen_inputs.py",
            "message": "Testing shape of image_grid_thw in collate_fn",
            "data": {"shape": list(batch['image_grid_thw'].shape)},
            "runId": "run9",
            "hypothesisId": "H11"
        }) + "\n")
        
except Exception as e:
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(time.time()*1000)}_H11_error",
            "timestamp": int(time.time()*1000),
            "location": "test_qwen_inputs.py",
            "message": "Error running collate_fn test",
            "data": {"error": str(e)},
            "runId": "run9",
            "hypothesisId": "H11"
        }) + "\n")
