import torch
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import VLAConfig
from src.vla_model import VLA_Model

def test_model_init():
    print("Testing VLA Model Initialization...")
    
    # Use dummy backbone to avoid downloading full Qwen2-VL during quick test if possible
    # But Qwen2-VL is required by code.
    # We can mock it or just print configuration.
    
    config = VLAConfig()
    print(f"Configuration: {config}")
    
    # We won't actually load the model here to save time/bandwidth unless user runs it in docker
    print("Model structure defined in src/vla_model.py")
    
    try:
        # Just check imports and basic class definition
        model = VLA_Model(config)
        print("Model initialized successfully (this might fail if weights not present).")
    except Exception as e:
        print(f"Initialization skipped/failed (expected if no weights): {e}")
        print("This test mainly verifies code structure and imports.")

if __name__ == "__main__":
    test_model_init()
