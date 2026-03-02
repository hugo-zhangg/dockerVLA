# ==============================================================================
# [LEARNING LEVEL]: IGNORE
# [ROLE]: Temporary debugging script for Qwen DDP shape issues.
# ==============================================================================

import torch
import json
import time
from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

log_path = "/root/.cursor/debug.log"

try:
    # Let's inspect the shapes that trigger "TypeError: iteration over a 0-d tensor"
    # The traceback showed: "for t, h, w in grid_thw:"
    # This means grid_thw must be a 2D tensor (N, 3), but it became a 0D scalar or 1D somehow.
    
    # Simulate forward pass shapes to see if accelerate prepare changes it
    grid_thw = torch.cat([torch.tensor([[1, 2, 2]]), torch.tensor([[1, 3, 5]])], dim=0) # [2, 3]
    
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "id": f"log_{int(time.time()*1000)}_H12",
            "timestamp": int(time.time()*1000),
            "location": "test_qwen_ddp_shape.py",
            "message": "Analyzing grid_thw shape",
            "data": {"shape": list(grid_thw.shape)},
            "runId": "run10",
            "hypothesisId": "H12"
        }) + "\n")
        
except Exception as e:
    pass
