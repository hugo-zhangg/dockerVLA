#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vla

echo "================================================="
echo "   Stage C - Phase 1: Train and Save Checkpoint  "
echo "================================================="
# Run for 50 steps, saving checkpoint at step 50
accelerate launch --multi_gpu --num_processes 2 scripts/train.py \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --max_episodes 4 \
  --max_steps 50 \
  --log_steps 10 \
  --save_steps 50 \
  --checkpoint_dir checkpoints/test_stage_c

echo "================================================="
echo "   Stage C - Phase 2: Resume from Checkpoint     "
echo "================================================="
# Find the exact checkpoint file name (e.g. vla_model_step_50.pt)
LATEST_CKPT=$(ls -t checkpoints/test_stage_c/vla_model_step_*.pt | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found! Phase 1 failed to save."
    exit 1
fi

echo "Found checkpoint: $LATEST_CKPT"
echo "Resuming training for another 50 steps..."

# Resume training with max_steps 100
accelerate launch --multi_gpu --num_processes 2 scripts/train.py \
  --epochs 1 \
  --batch_size 2 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --max_episodes 4 \
  --max_steps 100 \
  --log_steps 10 \
  --save_steps 200 \
  --checkpoint_dir checkpoints/test_stage_c \
  --resume_from_checkpoint "$LATEST_CKPT"

echo "================================================="
echo "        Stage C Test Completed Successfully!     "
echo "================================================="
