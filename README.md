# DockerVLA: Cloud-Ready Vision-Language-Action Model Training

An end-to-end framework for training VLA models (Qwen2-VL + DiT) on cloud or local servers using Docker.

## 🚀 Project Overview
- **Architecture**: Qwen2-VL (Vision-Language Backbone) + DiT (Diffusion Transformer Action Head).
- **Data**: Libero RLDS dataset.
- **Infrastructure**: Dockerized environment with PyTorch 2.5+, CUDA 12.1+, and Accelerate for multi-GPU training.
- **Monitoring**: TensorBoard logging.

---

## 🛠️ Execution & Testing Guide

**CRITICAL RULE**: ALL training and testing operations MUST be executed INSIDE the Docker container. Do not run training scripts directly on your host machine (e.g., local conda environment).

### Step 1: Start and Enter the Docker Container
Run this on your host machine to start the container and open an interactive bash shell inside it:

```bash
docker run -it --gpus all --ipc=host --name vla_interactive --rm -v "$(pwd)":/workspace vla_offline_ready /bin/bash
```

Once inside the container, you will be at `/workspace`. Run all following commands from here.

### Step 2: Local Single-GPU Smoke Test (Before Cloud Deployment)
Run this command inside the container to verify `accelerate` and your codebase work properly on a single GPU.

```bash
accelerate launch --num_processes 1 scripts/train.py \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --num_workers 2 \
  --max_episodes 1 \
  --max_steps 50 \
  --log_steps 1 \
  --save_steps 1000000
```

### Step 3: Cloud Multi-GPU Staged Smoke Tests
When migrating to the cloud, do not start a massive full-dataset training immediately. Follow these staged validation steps inside the cloud Docker container.

#### Stage A: 2 GPUs + 1 Episode + 100 Steps
**Goal**: Validate DDP/NCCL initialization, process startup, gradient synchronization, and logging.
```bash
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL \
accelerate launch --multi_gpu --num_processes 2 scripts/train.py \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --num_workers 2 \
  --max_episodes 1 \
  --max_steps 100 \
  --log_steps 10 \
  --save_steps 1000000
```

#### Stage B: 2 GPUs + 4 Episodes + 300 Steps
**Goal**: Validate TFDS data iteration and multi-worker stability under distributed execution.
```bash
accelerate launch --multi_gpu --num_processes 2 scripts/train.py \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --max_episodes 4 \
  --max_steps 300 \
  --log_steps 10 \
  --save_steps 1000000
```

#### Stage C: 2 GPUs + 10 Episodes + 200,000 Steps
**Goal**: Observe convergence characteristics and loss downward trend over a longer training period (200k steps). Use checkpoints to run inference and output ground truth vs. predicted action curves to visually evaluate fitting quality.
```bash
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
```

#### Stage D: Target GPUs (e.g., 8 GPUs) + Full Training
Once Stage C is successful, run your full pre-training script.
```bash
./run_cloud.sh
```

---

## 📦 Migration Guide (Offline Transfer)
To migrate this project to a new server (e.g., a secure cloud node with no internet access), follow these steps:

### Step 1: Export from Current Machine
1.  **Commit your environment**:
    ```bash
    docker commit vla_cloud_final vla_offline_ready
    ```
2.  **Save Docker Image to File** (Warning: Large file ~15GB+):
    ```bash
    docker save -o vla_offline_ready.tar vla_offline_ready
    ```
3.  **Pack Project Source Code and Models**:
    ```bash
    ./pack_for_cloud.sh
    ```
4.  **Copy Files**: Transfer `vla_offline_ready.tar`, `dockerVLA_cloud_ready.tar.gz`, and the `data/` folder to your new server.

### Step 2: Import on New Server
1.  **Load Docker Image**:
    ```bash
    docker load -i /path/to/vla_offline_ready.tar
    ```
2.  **Extract Project**:
    ```bash
    tar -xzvf dockerVLA_cloud_ready.tar.gz
    cd dockerVLA
    ```
3.  **Restore Dataset**: Move the `data/` folder into the `dockerVLA/` directory.

### Step 3: Incremental Update (Git + Docker)
For iterative updates after the initial migration, avoid transferring large `.tar` files. Use Git and incremental Docker updates:
1.  **Code Updates (via Git)**:
    - On your local machine: Commit and push changes (`git commit -am "update" && git push`).
    - On the cloud server: Pull the changes (`git pull`).
    - *Note*: If you only changed code (e.g., `.py` scripts), you do not need to rebuild the Docker image. Just restart the container using the command in Step 1 of the Execution & Testing Guide, and the updated code will be mounted automatically.
2.  **Environment Updates (via Docker)**:
    - If you installed new `pip` packages, install them interactively inside the existing cloud container, then commit the running container to update your image (`docker commit <container_id> vla_offline_ready`).
    - Alternatively, build a new image locally, save only the delta, or simply update `install_runtime.sh` and push via Git so the cloud container installs it upon startup.

---

## 📊 Monitoring
Logs are saved locally to `runs/vla_experiment`.

### ClearML Integration (Cloud Tracking)
For advanced experiment tracking, we use ClearML. To set it up on a new server:
1. **Register**: Go to [app.clear.ml](https://app.clear.ml) and create a free account.
2. **Generate Credentials**: Go to `Settings` -> `Workspace` -> `Create new credentials`.
3. **Configure Server**: Run `clearml-init` in your container/terminal and paste the configuration block.
Alternatively, manually create the `~/clearml.conf` file:
```bash
cat << 'EOF' > ~/clearml.conf
api {
    web_server: https://app.clear.ml
    api_server: https://api.clear.ml
    files_server: https://files.clear.ml
    credentials {
        "access_key" = "YOUR_ACCESS_KEY"
        "secret_key" = "YOUR_SECRET_KEY"
    }
}
EOF
```

### Run TensorBoard Inside Docker (Local Alternative)
```bash
docker run -it --rm -p 6006:6006 -v $(pwd)/runs:/workspace/runs vla_offline_ready tensorboard --logdir /workspace/runs --host 0.0.0.0
```
Then open `http://localhost:6006`.

---
---

# DockerVLA: 支持云端部署的视觉语言动作模型训练框架

这是一个基于 Docker 的端到端训练框架，用于在云端或本地服务器上训练 VLA 模型（Qwen2-VL + DiT）。

## 🚀 项目概览
- **核心架构**：Qwen2-VL（视觉语言主干网络）+ DiT（扩散 Transformer 动作头）。
- **数据来源**：Libero RLDS 数据集。
- **基础设施**：Docker 容器化环境，集成 PyTorch 2.5+、CUDA 12.1+ 和 Accelerate（支持多卡训练）。
- **训练监控**：使用 TensorBoard 记录日志。

---

## 🛠️ 执行与测试指南

**核心原则**：所有的训练和测试操作**必须在 Docker 容器内部执行**。不要在你的宿主机（例如本地 Conda 环境）上直接运行训练脚本。

### 第 1 步：启动并进入 Docker 容器
在你的宿主机上运行此命令，以启动容器并进入交互式 bash 终端：

```bash
docker run -it --gpus all --ipc=host --name vla_interactive --rm -v "$(pwd)":/workspace vla_offline_ready /bin/bash
```

进入容器后，你将处于 `/workspace` 目录。接下来的所有命令均在此处执行。

### 第 2 步：本地单卡冒烟测试（上云部署前）
在容器内运行此命令，以验证 `accelerate` 和代码在单张显卡上是否正常工作。

```bash
accelerate launch --num_processes 1 scripts/train.py \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --num_workers 2 \
  --max_episodes 1 \
  --max_steps 50 \
  --log_steps 1 \
  --save_steps 100000
```

### 第 3 步：云端多卡分阶段冒烟测试
迁移到云端后，不要立即启动全量数据集训练。请在云端的 Docker 容器内按照以下阶段进行验证。

#### 阶段 A：2 卡 + 1 个 Episode + 100 Steps
**目标**：验证 DDP/NCCL 初始化、进程启动、梯度同步以及日志写入。
```bash
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL \
accelerate launch --multi_gpu --num_processes 2 scripts/train.py \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --num_workers 2 \
  --max_episodes 1 \
  --max_steps 100 \
  --log_steps 10 \
  --save_steps 1000000
```

#### 阶段 B：2 卡 + 4 个 Episodes + 300 Steps
**目标**：验证 TFDS 数据迭代以及多 worker 在分布式执行下的稳定性。
```bash
accelerate launch --multi_gpu --num_processes 2 scripts/train.py \
  --epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 1 \
  --num_workers 4 \
  --max_episodes 4 \
  --max_steps 300 \
  --log_steps 10 \
  --save_steps 1000000
```

#### 阶段 C：2 卡 + 10 个 Episodes + 200,000 Steps
**目标**：观察在更长训练周期（20万步）下的收敛特性与 Loss 下降趋势。并利用断点进行推理，输出预测动作与 Ground Truth 的对比曲线，以直观判断拟合质量。
```bash
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
```

#### 阶段 D：目标显卡数（如 8 卡）+ 全量训练
当阶段 C 验证成功后，即可运行全量预训练脚本。
```bash
./run_cloud.sh
```

---

## 📦 迁移指南（离线转移）
如果需要将项目迁移到新的服务器（例如无法连接外网的保密云节点），请按照以下步骤操作：

### 第一步：从当前机器导出
1.  **保存 Docker 运行环境 (Commit)**：
    ```bash
    docker commit vla_cloud_final vla_offline_ready
    ```
2.  **将 Docker 镜像保存为文件**（注意：文件很大，约 15GB+）：
    ```bash
    docker save -o vla_offline_ready.tar vla_offline_ready
    ```
3.  **打包项目代码与模型**：
    ```bash
    ./pack_for_cloud.sh
    ```
4.  **拷贝文件**：将 `vla_offline_ready.tar`、`dockerVLA_cloud_ready.tar.gz` 和 `data/` 目录传输到新服务器。

### 第二步：在新服务器导入
1.  **加载镜像**：
    ```bash
    docker load -i /path/to/vla_offline_ready.tar
    ```
2.  **解压项目代码**：
    ```bash
    tar -xzvf dockerVLA_cloud_ready.tar.gz
    cd dockerVLA
    ```
3.  **恢复数据集**：将之前拷贝的 `data/` 文件夹放置到解压后的 `dockerVLA` 目录中。

### 第三步：增量更新（Git + Docker）
在完成初次迁移后，进行后续迭代时，请避免频繁传输庞大的 `.tar` 镜像文件。推荐使用 Git 和 Docker 增量更新的组合策略：
1.  **代码同步（通过 Git）**：
    - 在本地机器：提交并推送代码更改（`git commit -am "update" && git push`）。（*注：如果你需要代码托管平台的账号配置，请提前配置 SSH 密钥或凭证*）。
    - 在云端服务器：拉取最新代码（`git pull`）。
    - *优势*：如果只修改了 `.py` 等代码文件，**无需**更新 Docker 镜像。直接使用“执行与测试指南”中第一步的命令启动容器，更新后的代码会通过挂载目录直接生效。
2.  **环境同步（通过 Docker）**：
    - 如果你新增了 `pip` 包依赖，可以直接进入云端已有的容器内运行 `pip install`，然后通过 `docker commit <容器ID> vla_offline_ready` 将其固化为新镜像。
    - 也可以将新的依赖写入 `install_runtime.sh` 并通过 Git 同步，下次容器启动时会自动安装。

---

## 📊 监控指南
日志文件保存在本地的 `runs/vla_experiment` 目录下。

### ClearML 实验追踪 (云端监控)
为了实现更高级的实验指标监控，项目接入了 ClearML。要在新服务器上进行配置：
1. **注册账号**：前往 [app.clear.ml](https://app.clear.ml) 免费注册。
2. **生成凭证**：登录后，点击右上角头像进入 `Settings` -> `Workspace` -> `Create new credentials`。
3. **配置服务器**：在容器或宿主机终端内运行 `clearml-init` 并粘贴生成的配置代码。
或者直接创建 `~/clearml.conf` 配置文件：
```bash
cat << 'EOF' > ~/clearml.conf
api {
    web_server: https://app.clear.ml
    api_server: https://api.clear.ml
    files_server: https://files.clear.ml
    credentials {
        "access_key" = "你的_ACCESS_KEY"
        "secret_key" = "你的_SECRET_KEY"
    }
}
EOF
```

### 在 Docker 内启动 TensorBoard (本地备选)
```bash
docker run -it --rm -p 6006:6006 -v $(pwd)/runs:/workspace/runs vla_offline_ready tensorboard --logdir /workspace/runs --host 0.0.0.0
```
然后打开浏览器访问: `http://localhost:6006`。