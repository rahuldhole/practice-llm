# Blueprint: Continual Pre-training (CPT)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Standard autoregressive Cross-Entropy Loss over the newly introduced domain data.
  - **KL-Divergence Regularization (Optional):** Optional penalty to prevent catastrophic forgetting of general-domain knowledge by aligning student activations with the original base model's outputs.
- **FLOPs Scaling Formula:**
  - Total training FLOPs: $\text{FLOPs} \approx 6 \times N \times D$ (where $N$ is the parameter count (~3 Billion) and $D$ is the token count (1B to 5B)).
  - Compute per token (forward pass): $2 \times N$ FLOPs.
- **Precision Profile:**
  - **Weights and Activations:** BF16/FP16 (matching the precision format of the base model).
  - **Optimizer States:** FP32 (using mixed-precision AdamW to optimize GPU memory footprint).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 8x H100 (SXM5, 80GB) Spot instances.
- **Provider:** Vast.ai (or equivalent spot instance marketplaces).
- **Execution Duration Limit:** < 8 hours (optimized for quick, cost-contained injection runs).
- **Target Token/Batch Scale:** 1B to 5B ultra-curated tokens. Batch size dynamically scaled to fit memory limits with rapid gradient accumulation.
- **Estimated Budget:** ~$80 for compute, ~$5 for Cloudflare R2 storage, ~$15 buffer for retry runs. Total: ~$100.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** Multi-node clusters of 8x H100 (or H200/B200) nodes.
- **Interconnect Requirements:** NVLink (intra-node) and InfiniBand (inter-node) for high-throughput distributed parallel training (e.g., FSDP or DeepSpeed ZeRO-3).
- **Persistent Storage Topography:** High-speed NVMe local SSD scratch drives for local checkpoints, synchronized asynchronously via a background task to Cloudflare R2 or AWS S3.

## 3. Data Topography
- **Token Window Length:** Up to the maximum supported by the base model (e.g., 32,768 or 131,072 tokens).
- **Preprocessing Requirements:** 
  - Strict curation of target domain documents.
  - Line-by-line and document-level deduplication to remove fluff and duplicates.
- **Tokenizer Handling:** Lock the base model's tokenizer (e.g., Qwen/Llama tokenizer). **Do not** train a new tokenizer, as keeping the original token vocabulary intact is crucial to maintaining the foundational weights and internal embeddings of the base model.

## 4. Run Execution Script
```bash
#!/usr/bin/env bash
# Actionable script using LitGPT or Unsloth for Continual Pre-training.

set -eo pipefail

LOCAL_NVME_DIR="/mnt/nvme/checkpoints/cpt-3b"
R2_BUCKET_URI="r2://sovereign-ai-checkpoints/cpt-3b"

mkdir -p "${LOCAL_NVME_DIR}"

# Background async backup daemon
sync_to_r2() {
  while true; do
    echo "[Sync] Scanning for new local checkpoints..."
    rclone sync "${LOCAL_NVME_DIR}" "${R2_BUCKET_URI}" --fast-list --exclude "*.tmp"
    sleep 300
  done
}

# Run the sync daemon in the background
sync_to_r2 &
SYNC_PID=$!

# Terminate the background sync daemon when script exits
trap 'kill ${SYNC_PID} || true' EXIT

# Run pre-training execution
litgpt pretrain \
  --config configs/3b_dense_model.yaml \
  --initial_checkpoint_dir "Qwen/Qwen2.5-3B" \
  --data_dir "./my_curated_data" \
  --train.max_epochs 1 \
  --train.learning_rate 5e-5 \
  --train.warmup_steps 1000 \
  --train.checkpoint_dir "${LOCAL_NVME_DIR}" \
  2>&1 | tee -a "/var/log/cpt_run.log"
```

## 5. Failure Modes & Recovery
- **Weight Shattering (Catastrophic Loss of Foundation Knowledge):**
  - *Indicator:* Spikes in perplexity or loss at the very beginning of the run; the model starts outputting gibberish or repetitive sequences.
  - *Mitigation:* Ensure a strict cosine learning rate schedule with a learning rate warm-up period (at least 500–1000 steps) to allow the pre-trained weights to adjust to the new data distribution. Reduce the peak learning rate if shattering persists.
- **Flat or Spiking Loss Curves:**
  - *Indicator:* Training loss stops descending, stays completely flat, or diverges mid-run.
  - *Mitigation:* Immediately terminate the execution. Check learning rate configs, reduce the learning rate, check for dataset corruption (empty chunks, malformed inputs), and restart from the last successful checkpoint.
- **Preemption Recovery (Spot Instances):**
  - *Indicator:* Spot instance terminated by the provider.
  - *Mitigation:* Restore the last saved local checkpoint from Cloudflare R2 using the sync daemon:
    ```bash
    rclone copy "${R2_BUCKET_URI}" "${LOCAL_NVME_DIR}" --fast-list
    ```
    Resume training with the standard `--initial_checkpoint_dir` pointing to the recovered checkpoint directory.
