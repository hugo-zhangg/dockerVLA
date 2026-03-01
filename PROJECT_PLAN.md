# VLA (Vision-Language-Action) Model Training Project Plan

This is a 5-day project plan for training a VLA model.

## Day 1: Docker Environment Setup & Data Preparation (Completed ✅)
- **Objective**: Set up a robust development environment with Docker and prepare the LIBERO dataset.
- **Status**:
  - [x] Docker environment running with GPU support (RTX 4090).
  - [x] Model architecture implemented (`src/vla_model.py`).
  - [x] Data downloader rewritten to use Hugging Face Mirror (`download_hf.py`).
  - [x] **RLDS Data** (`libero_mix`) downloaded and verified (~75GB).
  - [x] Visualization scripts ready for both HDF5 and RLDS formats.

## Day 2: Data Pipeline & Architecture Upgrade (Completed ✅)
- **Objective**: Create efficient data loading, upgrade architecture to Qwen2-VL + DiT, and verify loop.
- **Status**:
  - [x] Architecture upgraded to Vision-Language Model + Diffusion Transformer (`src/vla_model.py`, `src/diffusion_policy.py`).
  - [x] **RLDS Dataset Class** implemented (`src/dataset.py`) using Qwen Processor.
  - [x] Training loop implemented and verified with dry-runs (`scripts/train.py`).
  - [x] Network and Multi-GPU infrastructure prepared for Cloud (`run_cloud.sh`).

## Day 3: Architecture Verification (Overfitting Test) (Completed ✅)
- **Objective**: Verify that the Qwen2-VL + DiT architecture (with Flow Matching) correctly learns and reproduces robotic actions before launching massive cloud training.
- **Tasks**:
  - [x] Create `train_overfit.py` to extract a single complete episode (e.g., 277 steps).
  - [x] Implement gradient accumulation to bypass CUDA OOM errors on large context lengths.
  - [x] Switch DDPMScheduler to `FlowMatchEulerDiscreteScheduler` for smoother convergence.
  - [x] Train on the single episode for 5000-10000 steps until MSE drops near 0.
  - [x] Run inference verification to ensure Predicted Actions match Ground Truth (Diff < 0.05).
  - [x] If successful, proceed to Day 4.

## Day 4: Cloud Migration & Multi-GPU Pre-training (Next Step 🚀)
- **Objective**: Migrate verified codebase to cloud servers (e.g., 8x H200/H20/4090) and launch full training.
- **Tasks**:
  - [x] Local single-card smoke test with `accelerate` single process before cloud deployment.
    - Purpose: Ensure we do not only validate the `python train.py` code path, and catch `accelerate` integration issues early.
  - [ ] Update README with Git + Docker Image Incremental Update strategy.
  - [ ] Integrate ClearML for advanced experiment tracking.
  - [ ] Refactor `train_overfit.py` for 2-episode consistency check and checkpoint saving.
  - Load Docker image on the cloud node.
  - Run `run_cloud.sh` to auto-detect GPUs and configure `accelerate`.
  - Run staged cloud smoke tests before long runs:
    - **Stage A**: 2 GPUs + 1 episode + 50-100 steps.
      - Goal: Validate DDP/NCCL initialization, process startup, gradient synchronization, and logging stability.
    - **Stage B**: 2 GPUs + 2-4 episodes + 100-300 steps.
      - Goal: Validate TFDS data iteration and multi-worker stability under distributed execution.
    - **Stage C**: 2 GPUs + 10+ episodes + 300-500 steps.
      - Goal: Validate checkpoint saving, resume correctness, and loss curve continuity.
    - **Stage D**: Target GPU count (4/8) + multi-episode smoke test.
      - Goal: Final distributed smoke test before launching long full-dataset training.
  - Train on the full `libero_mix` dataset (approx 75GB).
  - Monitor training metrics via WandB / Tensorboard.
  - Save checkpoints dynamically based on `global_step`.

## Day 5: Simulation Evaluation Setup
- **Objective**: Set up the evaluation pipeline in `robosuite`.
- **Tasks**:
  - Install and configure `robosuite` and `libero` simulation environments.
  - Write `scripts/eval.py` to run the trained model in the simulator.
  - Define success metrics and measure task completion rate.

---
---

# VLA（视觉-语言-动作）模型训练项目计划

这是一个为期 5 天的 VLA 模型训练计划。

## Day 1: Docker 环境搭建与数据准备（已完成 ✅）
- **目标**：使用 Docker 建立稳健的开发环境，并准备 LIBERO 数据集。
- **状态**：
  - [x] 成功运行支持 GPU (RTX 4090) 的 Docker 环境。
  - [x] 实现模型基础架构 (`src/vla_model.py`)。
  - [x] 重写数据下载器以使用 Hugging Face 镜像 (`download_hf.py`)。
  - [x] **RLDS 数据** (`libero_mix`) 下载并验证完毕 (~75GB)。
  - [x] 准备好 HDF5 和 RLDS 格式的可视化脚本。

## Day 2: 数据管道与架构升级（已完成 ✅）
- **目标**：创建高效的数据加载器，将架构升级为 Qwen2-VL + DiT，并验证训练循环。
- **状态**：
  - [x] 架构升级为视觉语言模型 + 扩散 Transformer (`src/vla_model.py`, `src/diffusion_policy.py`)。
  - [x] 使用 Qwen Processor 实现 **RLDS 数据集类** (`src/dataset.py`)。
  - [x] 实现训练循环并通过空跑验证 (`scripts/train.py`)。
  - [x] 为云端准备网络和多 GPU 基础设施 (`run_cloud.sh`)。

## Day 3: 架构验证（过拟合测试）（已完成 ✅）
- **目标**：在大规模云端训练前，验证 Qwen2-VL + DiT 架构（采用 Flow Matching）能否正确学习并复现机器人动作。
- **任务**：
  - [x] 创建 `train_overfit.py` 提取单个完整的回合（如 277 步）。
  - [x] 实现梯度累加，绕过长上下文带来的 CUDA 内存溢出（OOM）错误。
  - [x] 将 DDPMScheduler 切换为 `FlowMatchEulerDiscreteScheduler` 以获得更平滑的收敛。
  - [x] 在单回合上训练 5000-10000 步，直到 MSE 降至接近 0。
  - [x] 运行推理验证，确保预测动作与真实动作匹配（差异 < 0.05）。
  - [x] 如果成功，进入 Day 4。

## Day 4: 云端迁移与多 GPU 预训练（下一步 🚀）
- **目标**：将已验证的代码库迁移到云端服务器（如 8x H200/H20/4090）并启动全面训练。
- **任务**：
  - [x] 在部署云端前，先进行本地单卡 `accelerate` 单进程冒烟测试。
    - 目的：确保不仅仅是验证 `python train.py` 的代码路径，尽早发现 `accelerate` 的集成问题。
  - [ ] 在 README 中补充 Git + Docker Image 增量更新同步策略。
  - [ ] 接入 ClearML 以实现更高级的实验与指标监控。
  - [ ] 重构 `train_overfit.py` 以支持 2-episode 的一致性测试与断点保存。
  - 在云端节点加载 Docker 镜像。
  - 运行 `run_cloud.sh` 自动检测 GPU 并配置 `accelerate`。
  - 在长跑前运行分阶段云端冒烟测试：
    - **阶段 A**：2 卡 + 1 个 episode + 50-100 steps。
      - 目标：验证 DDP/NCCL 初始化、进程启动、梯度同步和日志稳定性。
    - **阶段 B**：2 卡 + 2-4 个 episodes + 100-300 steps。
      - 目标：验证 TFDS 数据迭代以及多 worker 在分布式执行下的稳定性。
    - **阶段 C**：2 卡 + 10+ 个 episodes + 300-500 steps。
      - 目标：验证 checkpoint 保存、断点恢复正确性以及 loss 曲线的连续性。
    - **阶段 D**：目标显卡数 (4/8 卡) + 多 episode 冒烟测试。
      - 目标：正式全量数据集长跑前的最后一次分布式冒烟测试。
  - 在完整 `libero_mix` 数据集上进行训练 (约 75GB)。
  - 通过 WandB / Tensorboard 监控训练指标。
  - 根据 `global_step` 动态保存 checkpoints。

## Day 5: 仿真评估设置
- **目标**：在 `robosuite` 中设置评估管道。
- **任务**：
  - 安装并配置 `robosuite` 和 `libero` 仿真环境。
  - 编写 `scripts/eval.py` 在模拟器中运行已训练的模型。
  - 定义成功指标并测量任务完成率。
