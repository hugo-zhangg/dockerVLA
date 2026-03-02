# ==============================================================================
# [LEARNING LEVEL]: IGNORE
# [ROLE]: Temporary debugging script to check Hugging Face model config fields.
# ==============================================================================

from transformers import AutoConfig
try:
    config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
    print(dir(config))
    if hasattr(config, 'hidden_size'):
        print(f"hidden_size: {config.hidden_size}")
    else:
        print("No hidden_size")
        
    if hasattr(config, 'text_config'):
        print(f"text_config: {config.text_config}")
except Exception as e:
    print(e)
