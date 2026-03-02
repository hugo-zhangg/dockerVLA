# ==============================================================================
# [LEARNING LEVEL]: IGNORE
# [ROLE]: Utility script to download Qwen2-VL model from Hugging Face Hub to local directory.
# ==============================================================================

import os
from huggingface_hub import snapshot_download

# 配置
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LOCAL_DIR = "models/Qwen2-VL-2B-Instruct"

print(f"Downloading {MODEL_ID} to {LOCAL_DIR}...")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,  # 确保下载真实文件，方便拷贝
    resume_download=True
)

print("Download complete.")
