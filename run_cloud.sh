#!/bin/bash
# ==============================================================================
# [LEARNING LEVEL]: SKIM
# [ROLE]: Cloud launch script for multi-GPU or single-GPU training.
#         Handles network mirrors and dynamic GPU utilization.
# ==============================================================================

echo "========================================"
echo "    Starting DockerVLA Cloud Training   "
echo "========================================"

# 1. Network Configuration (Crucial for China Cloud Environments)
echo "[1] Configuring Network Mirrors..."

# Disable system proxies that might interfere with pip or HF downloads
export http_proxy=""
export https_proxy=""
export HTTP_PROXY=""
export HTTPS_PROXY=""
export all_proxy=""
export ALL_PROXY=""

# Set HuggingFace mirror
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER="0"

# 2. Check and Download Data
echo "[2] Checking Dataset..."
# We expect the dataset at data/libero_rlds/inspire/...
# Let's just check if the data directory has the expected folder structure
if [ ! -d "data/libero_rlds" ] || [ -z "$(ls -A data/libero_rlds 2>/dev/null)" ]; then
    echo "Dataset not found. Starting automatic download..."
    python download_hf.py
    if [ $? -ne 0 ]; then
        echo "❌ Error downloading dataset. Please check network or download manually."
        exit 1
    fi
else
    echo "Dataset found. Skipping download."
fi

# 3. Check GPUs
echo "[3] Detecting GPUs..."
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi -L | grep "GPU" | wc -l)
        echo "CUDA_VISIBLE_DEVICES is not set. Found $NUM_GPUS physical GPU(s)."
        nvidia-smi -L
    else
        echo "Error: nvidia-smi not found. Ensure NVIDIA drivers and container toolkit are installed."
        NUM_GPUS=0
    fi
else
    # Calculate number of GPUs specified by counting commas + 1
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
    echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"
    echo "Using $NUM_GPUS targeted GPU(s)."
fi

# 4. Launch Training
echo "[4] Launching Training..."

# Add current directory to PYTHONPATH so python can find 'src'
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Accelerate requires a config file or default arguments.
# We will use accelerate launch if NUM_GPUS > 1, else standard python.

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU detected. Launching with Accelerate..."
    # Launch with all available GPUs. 
    # "$@" passes any extra arguments to the script (e.g. --resume_from_checkpoint, --save_steps)
    accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        scripts/train.py "$@"
else
    echo "Single-GPU or No-GPU detected. Launching with standard python..."
    python scripts/train.py "$@"
fi

echo "========================================"
echo "          Training Finished             "
echo "========================================"
