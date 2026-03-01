# 🧠 Memory Bank & Handover Context

**Last Updated**: Day 2 Cloud Migration Preparedness
**Target Audience**: Next Agent Session

## 1. Project Status Summary
We have upgraded the VLA architecture to a **Unified Qwen2-VL + DiT** design and successfully verified the data pipeline and training loop locally. The project is now **Ready for Cloud Migration**.

- **Current Goal**: Deploy to a multi-GPU cloud server (e.g., 8x H200/H20/4090) and begin full-scale training.
- **Active Codebase**: Configured for domestic networks (Tuna/Aliyun/HF-Mirror) and dynamic multi-GPU scaling via `accelerate`.
- **Cloud Launcher**: `run_cloud.sh` is created for one-click deployment.

## 2. Environment Context (Crucial!)
*   **Pip Source**: Use `https://pypi.tuna.tsinghua.edu.cn/simple` (set globally in Dockerfile).
*   **Apt Source**: Aliyun mirrors configured in Dockerfile.
*   **Proxy**: `run_cloud.sh` clears proxy variables to prevent connection issues.
*   **Key Packages**: `transformers`, `diffusers`, `tensorflow-cpu`, `tensorflow_datasets`, `accelerate`.

## 3. Architecture Upgrade (The "Qwen-DiT" Design)
We moved from a traditional "CNN/ViT + UNet" to a **"VLM + Diffusion Transformer"** architecture:

*   **Vision Encoder**: **Qwen2-VL** (Frozen/LoRA).
    *   **Change**: Now returns **Full Sequence** `(B, L, H)` instead of pooled vector.
    *   **Why**: To preserve spatial info (2D-RoPE) and fine-grained details.
    *   **Input**: Dynamic resolution images + Instructions.
*   **Bridge**: **MLP Projector**.
    *   **Role**: Maps Qwen Hidden Size (3584) -> DiT Hidden Size (512).
*   **Action Head**: **DiT (Diffusion Transformer)**.
    *   **Architecture**: 1D Transformer Decoder (6 layers, 8 heads).
    *   **Input**: Noisy Action Sequence `(B, T, D)`.
    *   **Conditioning**: Cross-Attention to Qwen's visual/text tokens.
    *   **Why**: Better sequence modeling than UNet; allows direct attention to specific image patches.

## 4. Codebase Navigation (Rules)
*   **`src/vla_model.py`**: **[DEEP DIVE]** The main `VLA_Model` class assembling VisionEncoder, Projector, and DiT.
*   **`src/diffusion_policy.py`**: **[DEEP DIVE]** Contains `DiT1D` implementation.
*   **`src/vision_encoder.py`**: **[SKIM]** Qwen2-VL wrapper returning `last_hidden_state`.
*   **`src/dataset.py`**: **[DEEP DIVE]** RLDSDataset using Qwen Processor and Action Chunking.

## 5. Next Steps (Cloud Phase)
1.  **Transfer Code & Data**: Push Docker image (or Dockerfile) and dataset to the cloud instance.
2.  **Launch `run_cloud.sh`**: Auto-detects GPUs (e.g., 8-card setup) and runs multi-GPU training.
3.  **Monitor Loss**: Use WandB to ensure convergence during the pre-training phase on the `libero_mix` dataset.
4.  **Hardware Tuning**: Check if `get_optimal_batch_size()` in `scripts/train.py` needs adjustment based on actual cloud GPU (H20 vs 4090).
