#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vla

# We use NCCL_DEBUG and TORCH_DISTRIBUTED_DEBUG to log any potential distributed training issues.
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL \
accelerate launch --multi_gpu --num_processes 2 scripts/train.py \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --max_episodes 4 \
  --max_steps 300 \
  --log_steps 10 \
  --save_steps 1000000
