# ==============================================================================
# [LEARNING LEVEL]: IGNORE (无需关注)
# [ROLE]: 数据下载脚本 (Hugging Face)。
#         [LOGIC]: 配置国内镜像 -> 使用 huggingface_hub 下载 Libero 数据集 -> 存入 data 目录。
#         这是替代 download_libero.py 的方案，解决了国内无法连接 Box 的问题。
# ==============================================================================
# [TOC]:
# - [SKIM]      Environment Configuration (Lines 18-24): Set HF mirror.
# - [SKIM]      Download Execution (Lines 32-45): Call snapshot_download.
# ==============================================================================

import os
from huggingface_hub import snapshot_download

# [SECTION]: Environment Configuration
# [LEVEL]: SKIM
# [LOGIC]: 设置国内镜像源 (hf-mirror.com) 以加速下载。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 HF Transfer (加速有时候不稳定，先求稳)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

repo_id = "sii-research/libero_plus_rlds"
# 下载到 data 目录下的子文件夹
local_dir = "./data/libero_rlds"

print(f"Starting download from {repo_id} to {local_dir}...")

# [SECTION]: Download Execution
# [LEVEL]: SKIM
# [LOGIC]: 调用 huggingface SDK 的 snapshot_download 函数。
#          resume_download=True 确保断点续传。
try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        resume_download=True,
        max_workers=8,
        # 只下载部分文件以快速验证连接 (例如只下载 json 或 txt，或者不设置 allow_patterns 先下载全部)
        # allow_patterns=["*.json", "*.yaml", "*.md"] 
    )
    print("Download completed successfully!")
except Exception as e:
    print(f"Download failed: {e}")
