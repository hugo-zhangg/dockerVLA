#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vla

echo "================================================="
echo "   Long Training: 2 GPUs, 10 Episodes, 200k Steps"
echo "================================================="

# 我们设置一个足够大的 epochs，以确保模型能完整跑完 200,000 步
# max_episodes=10 表示只加载 10 个轨迹数据反复进行训练
# save_steps=20000 表示每隔 2万步 保存一次 checkpoint

accelerate launch --multi_gpu --num_processes 2 scripts/train.py \
  --epochs 5000 \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --max_episodes 10 \
  --max_steps 200000 \
  --log_steps 100 \
  --save_steps 20000 \
  --checkpoint_dir checkpoints/train_200k

echo "================================================="
echo "        200k Steps Training Completed!           "
echo "================================================="
