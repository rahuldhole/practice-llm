# Blueprint: Sparse MoE Pre-training

## 1. Core Physics & Mechanics
- **Mathematical Objective:** 
  - Cross-Entropy Loss for autoregressive language modeling.
  - **Auxiliary-Loss-Free Balancing:** Dynamic bias updates to enforce load balancing without interfering with the task loss. For each expert $i$, the router updates its routing bias $b_i$ dynamically:
    $$b_i \leftarrow b_i + \gamma \cdot (\text{target\_fraction} - \text{observed\_fraction})$$
    where $\gamma$ is a step size, maintaining uniform expert utilization.
- **FLOPs Scaling Formula:** 
  - Total training FLOPs: $\text{FLOPs} \approx 6 \times N_{\text{active}} \times D$ (where $N_{\text{active}}$ is active parameters (~800M) and $D$ is the total token count).
  - Compute per token (forward pass): $2 \times N_{\text{active}}$ FLOPs for non-attention layers, plus standard attention FLOPs.
- **Precision Profile:** 
  - **Router / Gating Math:** FP32 (Critical to prevent precision truncation and subsequent gradient explosion).
  - **Attention & Expert Blocks (Weights & Activations):** BF16.
  - **Optimizer States:** FP32 (standard AdamW state representation).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x H100 (80GB PCIe/SXM5) on spot instance.
- **Provider:** Vast.ai or RunPod.
- **Execution Duration Limit:** 2–4 hours.
- **Target Token/Batch Scale:** ~1 Billion tokens debug run. Small batch size (e.g., micro-batch size of 1 or 2 with gradient accumulation to fit in memory).

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 (SXM5, 80GB) node cluster, scalable to multi-node configurations.
- **Interconnect Requirements:** NVLink (intra-node, 900 GB/s) and InfiniBand (inter-node, 400 Gb/s) for high-performance token-shuffling (`all-to-all` collective operations) across experts.
- **Persistent Storage Topography:** High-speed local NVMe scratch disk (minimum 2TB read/write throughput of >5GB/s) acting as a direct dump for checkpoints, coupled with an asynchronous background daemon syncing checkpoints via `rclone` or AWS CLI to a Cloudflare R2 bucket.

## 3. Data Topography
- **Token Window Length:** 32,768 (32K) context window, optimized for high throughput while retaining deep spatial coherence.
- **Preprocessing Requirements:** 
  - Deduplicate corpus using MinHash LSH (Jaccard similarity threshold > 0.8) to eliminate low-value duplicates.
  - Format training corpus into contiguous pre-shuffled memory-mapped binary arrays (`.bin` files).
- **Tokenizer Handling:** Bespoke Byte-Pair Encoding (BPE) tokenizer with a 50,000 vocabulary size.

## 4. Run Execution Script
```bash
#!/usr/bin/env bash
# Actionable torchrun script using Megatron-DeepSpeed / DeepSpeed MoE configurations.

set -eo pipefail

# Environment & Infrastructure variables
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"29500"}
export WORLD_SIZE=${WORLD_SIZE:-"8"}
export RANK=${RANK:-"0"}

# Cloudflare R2 Checkpointing details
R2_BUCKET_URI="r2://sovereign-ai-checkpoints/moe-3b"
LOCAL_CHECKPOINT_DIR="/mnt/nvme/checkpoints/moe-3b"

mkdir -p "${LOCAL_CHECKPOINT_DIR}"

# Asynchronous sync hook (triggered in background)
sync_to_r2() {
  local step=$1
  echo "[Sync] Offloading checkpoint step ${step} to Cloudflare R2..."
  rclone sync "${LOCAL_CHECKPOINT_DIR}/step_${step}" "${R2_BUCKET_URI}/step_${step}" --fast-list &
}

# Run execution via DeepSpeed
torchrun \
  --nproc_per_node="${WORLD_SIZE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train_moe.py \
  --total-params 3000000000 \
  --active-params 800000000 \
  --num-experts 8 \
  --top-k 2 \
  --shared-expert true \
  --context-length 32768 \
  --data-path "/mnt/nvme/data/sharded_corpus" \
  --save-interval 1000 \
  --save-dir "${LOCAL_CHECKPOINT_DIR}" \
  --deepspeed_config configs/deepspeed_moe_config.json \
  --router-fp32 true \
  --xavier-init true \
  --router-warmup-steps 1000 \
  --learning-rate 3e-4 \
  --warmup-steps 2000 \
  2>&1 | tee -a "/var/log/moe_pretrain_run.log"

# Example DeepSpeed MoE Config (`configs/deepspeed_moe_config.json`):
# {
#   "train_batch_size": 2048,
#   "train_micro_batch_size_per_gpu": 2,
#   "gradient_accumulation_steps": 128,
#   "bf16": {
#     "enabled": true
#   },
#   "zero_optimization": {
#     "stage": 2,
#     "allgather_partitions": true,
#     "reduce_scatter": true,
#     "overlap_comm": true,
#     "contiguous_gradients": true
#   },
#   "moe": {
#     "enabled": true,
#     "num_experts": 8,
#     "type": "standard",
#     "moe_token_dropping": false
#   }
# }
```

## 5. Failure Modes & Recovery
- **Gradient Explosion:**
  - *Indicator:* Global gradient norm spikes > 1.0 or loss outputs NaN.
  - *Mitigation:* Immediately kill the execution, adjust learning rate downward (scale factor 0.5), load the last stable checkpoint step from Cloudflare R2, increase the warm-up step count, and re-execute.
- **Router Collapse (Expert De-activation):**
  - *Indicator:* Gating entropy drops below threshold (observed expert usage drops to 0% for certain experts while others receive 100%).
  - *Mitigation:* The dynamic bias update logic will increase bias weights on inactive experts. If this fails, switch to standard Auxiliary Load Balancing Loss (`router_aux_loss` coefficient set to 0.01) to force uniform distribution.
- **Weight Shattering:**
  - *Indicator:* Abrupt loss spikes during early validation stages (loss increases by >1.5x in a few hundred steps).
  - *Mitigation:* Implement a "Router Warm-up" phase (first 1,000 steps training only routing layers on token syntax before opening expert routing gates fully). If weight shattering occurs, restart from the initial checkpoint with a longer warm-up curve (e.g., 2,500 steps) and reduced learning rate.
