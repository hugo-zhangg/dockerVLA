#!/bin/bash
# ==============================================================================
# [LEARNING LEVEL]: SKIM
# [ROLE]: 运行时依赖安装脚本 (Bypass Docker Build Network Issues)
# [LOGIC]: 在容器启动时强制重装 PyTorch 和其他关键依赖，确保版本正确且下载成功。
# ==============================================================================

set -e  # Exit on error

echo "========================================"
echo "    Installing Runtime Dependencies     "
echo "========================================"

# 1. 强制重装 PyTorch (关键修复)
echo "[1] Reinstalling PyTorch 2.4+ (CUDA 12.1)..."
pip uninstall -y torch torchvision torchaudio || true
pip install --no-cache-dir \
    "torch>=2.4.0" \
    "torchvision>=0.19.0" \
    "torchaudio>=2.4.0" \
    --index-url https://download.pytorch.org/whl/cu121

# 2. 安装其他缺失依赖 (这些在 Dockerfile 中可能因网络失败未装上)
echo "[2] Installing Project Dependencies..."
# Added tensorboard
pip install --no-cache-dir \
    tensorflow-cpu \
    tensorflow_datasets \
    transformers \
    qwen_vl_utils \
    diffusers \
    accelerate \
    peft \
    robosuite \
    libero \
    h5py \
    wandb \
    clearml \
    tensorboard \
    einops \
    sentencepiece \
    timm \
    protobuf \
    opencv-python

echo "========================================"
echo "    Dependencies Installed Successfully "
echo "========================================"

# 3. 启动训练脚本
chmod +x run_cloud.sh
./run_cloud.sh
