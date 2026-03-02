# ==============================================================================
# [LEARNING LEVEL]: DEEP DIVE (核心学习)
# [ROLE]: 主训练脚本 (Main Training Loop)。
#         用于真实云端大规模训练，支持多 GPU 分布式训练 (`accelerate`)、梯度累加等优化。
# ==============================================================================
# [TOC]:
# - [SKIM]      TrainConfig (Lines 32-62): 定义训练核心超参数 (如 LoRA rank, DiT 层数)。
# - [DEEP DIVE] main - Setup (Lines 63-138): 初始化分布式环境、参数解析、载入数据迭代器和模型。
# - [DEEP DIVE] main - Resume Logic (Lines 139-158): 从 Checkpoint 断点续训逻辑。
# - [DEEP DIVE] main - Training Loop (Lines 159-218): 前向计算、梯度回传、日志与 max_steps 提前停止。
# ==============================================================================

import os
import sys

# Fix ModuleNotFoundError for 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import time
from torch.utils.data import DataLoader
from accelerate import Accelerator
from src.vla_model import VLA_Model
from src.dataset import RLDSDataset, QwenCollateFn
from torch.optim import AdamW
from diffusers.optimization import get_cosine_schedule_with_warmup

# Config class to match VLA_Model expectations
class TrainConfig:
    def __init__(self, args):
        # 优先使用本地下载好的模型
        local_model_path = "models/Qwen2-VL-2B-Instruct"
        if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
             self.vision_backbone = local_model_path
             print(f"Using local model from: {self.vision_backbone}")
        else:
             self.vision_backbone = "Qwen/Qwen2-VL-2B-Instruct" 
             print(f"Local model not found at {local_model_path}, using HF Hub: {self.vision_backbone}")

        # [LOGIC]: 使用 LoRA 训练 Qwen 可以把显存占用从 >20GB 降到 16GB 左右。
        self.use_lora = True
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        
        # 机器人动作维度，7 = 6个空间自由度 + 1个夹爪开合度
        self.action_dim = 7
        # [PHYSICS]: 预测视野。给定当前图，模型要预测未来 16 步该怎么走，有助于让动作平滑。
        self.pred_horizon = 16
        self.obs_horizon = 1
        
        self.diffusion_steps = 100
        self.dit_hidden_dim = 512
        self.dit_num_heads = 8
        self.dit_num_layers = 6
        self.dit_dropout = 0.1
        
        self.device = None # Will be set by Accelerator

def main():
    parser = argparse.ArgumentParser(description="Train VLA Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dataset_path", type=str, default="data/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/libero_plus_rlds00/libero_plus_mixdata/libero_mix/1.0.0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_steps", type=int, default=100000, help="Save checkpoint every N steps")
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Init ClearML
    try:
        from clearml import Task
        # Remove offline mode to allow cloud synchronization
        task = Task.init(project_name='DockerVLA', task_name='VLA_Training')
        # We can also connect argparse
        task.connect(vars(args))
    except Exception as e:
        print(f"ClearML init failed or not configured, skipping. Error: {e}")

    from accelerate.utils import DataLoaderConfiguration
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with="tensorboard",
        project_dir="runs",
        dataloader_config=DataLoaderConfiguration(
            split_batches=True,  # Default, but let's make it explicit
            dispatch_batches=False # Important! Do not let accelerate cut our batch. We'll do it or handle it.
        )
    )
    
    if accelerator.is_main_process:
        print(f"Training on {accelerator.num_processes} GPUs/CPUs")
        accelerator.init_trackers("vla_experiment")
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 2. Config & Model
    config = TrainConfig(args)
    config.device = accelerator.device
    
    # Initialize Model
    # Note: VisionEncoder inside VLA_Model handles Qwen loading
    model = VLA_Model(config)
    
    # 3. Dataset & Dataloader
    # Reuse processor from model to avoid double loading
    processor = model.vision_encoder.processor
    
    ds = RLDSDataset(
        dataset_path=args.dataset_path,
        processor=processor,
        pred_horizon=config.pred_horizon,
        action_dim=config.action_dim,
        max_episodes=args.max_episodes
    )
    
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        collate_fn=QwenCollateFn(processor),
        pin_memory=True
    )
    
    # 4. Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Scheduler
    # Estimate total steps or use a large number for infinite streaming
    steps_per_epoch = 10000 # Rough estimate since we don't know dataset length easily
    max_train_steps = args.epochs * steps_per_epoch
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=max_train_steps
    )
    
    # Note: Accelerator `prepare` explicitly scatters data inside DataLoaders.
    # Because Qwen's `pixel_values` (N, 1176) and `image_grid_thw` (Batch, 3) do NOT share the same
    # first dimension size, DDP scatter will corrupt them.
    # SOLUTION: Use `accelerator.prepare(model, optimizer, lr_scheduler)` but NOT dataloader!
    # Let the accelerate library wrap the dataloader normally, BUT we must tell it not to split
    # the dict keys it doesn't understand properly, OR we manually slice the batch ourselves.
    
    # Actually, accelerate's default dispatch splits all dict values at dim=0.
    # To fix Qwen DDP batch splitting, the standard Hugging Face solution is to pass
    # specific kwargs telling accelerate NOT to split pixel_values.
    
    # We will prepare everything together, but we need to intercept the batch splitting in training loop.
    model, optimizer, dl, lr_scheduler = accelerator.prepare(
        model, optimizer, dl, lr_scheduler
    )
    
    # 6. Resume from checkpoint (optional)
    global_step = 0
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

        checkpoint = torch.load(args.resume_from_checkpoint, map_location=accelerator.device)
        accelerator.unwrap_model(model).load_state_dict(checkpoint)

        # Recover step from filename pattern: vla_model_step_500000.pt
        try:
            if "step_" in args.resume_from_checkpoint:
                step_str = args.resume_from_checkpoint.split("step_")[-1].split(".")[0]
                global_step = int(step_str)
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Warning: failed to parse global_step from checkpoint filename: {e}")

        if accelerator.is_main_process:
            print(f"Checkpoint loaded. Continue from global_step={global_step}")

    # 7. Training Loop
    start_time = time.time()
    
    if accelerator.is_main_process:
        print("Starting training loop...")
    
    stop_training = False
    for epoch in range(args.epochs):
        model.train()
        for batch in dl:
            with accelerator.accumulate(model):
                # Forward
                loss_dict = model(
                    pixel_values=batch['pixel_values'],
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    image_grid_thw=batch['image_grid_thw'],
                    gt_actions=batch['actions']
                )
                
                loss = loss_dict['loss']
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                global_step += 1
                
                # Log
                if global_step % args.log_steps == 0:
                    if accelerator.is_main_process:
                        current_lr = lr_scheduler.get_last_lr()[0]
                        print(f"[Step {global_step}] Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
                        accelerator.log({
                            "train_loss": loss.item(),
                            "lr": current_lr
                        }, step=global_step)
                        
                # Save Checkpoint
                # [CRITICAL]: Modified to save every 100,000 steps per user request
                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.checkpoint_dir, f"vla_model_step_{global_step}.pt")
                        print(f"Saving checkpoint to {save_path}...")
                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), save_path)
                        print("Saved.")

                # Early stop for smoke tests (single-card and multi-card)
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_training = True
                    if accelerator.is_main_process:
                        elapsed = time.time() - start_time
                        print(f"Reached max_steps={args.max_steps}. Stopping training early. Elapsed={elapsed:.1f}s")
                    break
        if stop_training:
            break
            
    accelerator.end_training()

if __name__ == "__main__":
    main()
