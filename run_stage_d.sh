#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# 取消可能影响 ClearML 的代理
unset http_proxy https_proxy all_proxy

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vla

echo "================================================="
echo "   Stage D: Multi-GPU Smoke Test                 "
echo "================================================="

# 默认使用 4 张卡进行冒烟测试，如果你有更多或更少卡，可以修改 CUDA_VISIBLE_DEVICES 和 num_processes
# 这里指定使用 0,1,2,3 号卡
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

echo "Using GPUs: $CUDA_VISIBLE_DEVICES (Total: $NUM_GPUS)"

# 跑 4 个 episode，最多跑 200 步，主要验证多卡通信和前向/反向过程是否卡死
accelerate launch --multi_gpu --num_processes $NUM_GPUS scripts/train.py \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --max_episodes 4 \
  --max_steps 200 \
  --log_steps 10 \
  --save_steps 200 \
  --checkpoint_dir checkpoints/test_stage_d

echo "================================================="
echo "        Stage D Test Completed!                  "
echo "================================================="
