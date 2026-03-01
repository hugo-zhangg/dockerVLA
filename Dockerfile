# ==============================================================================
# [LEARNING LEVEL]: SKIM (快速浏览)
# [ROLE]: Docker 镜像定义文件。
#         [LOGIC]: 基于 PyTorch 基础镜像 -> 安装系统库 (libgl, ffmpeg) -> 安装 Python 库 (diffusers, robosuite)。
#         只需要知道它配置了 VLA 运行所需的所有环境即可。
# ==============================================================================
# [TOC]:
# - [SKIM] Base Image & System Deps (Lines 12-25): PyTorch + OpenCV/FFmpeg libs.
# - [SKIM] Python Dependencies (Lines 28-45): Transformers, Diffusers, Robosuite etc.
# ==============================================================================

# Based on official PyTorch image (Trying to use newer version for Qwen2-VL)
# 2.4.0 might not be available as tag directly, let's use 2.2.0 or upgrade manually
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Configure APT to use Aliyun mirror (crucial for cloud environments in China)
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# Install system dependencies
# libgl1-mesa-glx and libglib2.0-0 are crucial for OpenCV and headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configure PIP to use Tsinghua mirror globally
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install Python dependencies
# Includes packages for VLA: transformers, diffusers, robosuite, libero, etc.
# Note: Manually upgrade PyTorch if base image is too old
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir \
    "torch>=2.4.0" "torchvision>=0.19.0" "torchaudio>=2.4.0" --index-url https://download.pytorch.org/whl/cu121 && \
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
    einops \
    sentencepiece \
    timm \
    protobuf \
    opencv-python \
    tensorboard

# Set the working directory inside the container
WORKDIR /workspace

# Default command when container starts
CMD ["/bin/bash"]
