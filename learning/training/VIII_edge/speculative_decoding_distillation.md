# Blueprint: Speculative Decoding Distillation

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - KL-Divergence minimization between the token probability distribution of the large base model $P(x)$ (the Teacher, e.g., 3B) and a hyper-lightweight speculative model $Q(x)$ (the Student/Drafter, e.g., 100M):
    $$\min_{\theta} \mathcal{D}_{\text{KL}}(P(x) \parallel Q(x; \theta)) = \sum_{w \in \mathcal{V}} P(w \mid x_{<t}) \log \left(\frac{P(w \mid x_{<t})}{Q(w \mid x_{<t}; \theta)}\right)$$
    where $\mathcal{V}$ is the vocabulary space. Aligns draft token acceptance probability to maximize the speculative decoding validation rate.
- **FLOPs Scaling Formula:**
  - Standard training cost is determined by the draft student size:
    $$\text{FLOPs} \approx 6 \times N_{\text{draft}} \times D$$
    where $N_{\text{draft}}$ is the parameter count of the draft model (<100M) and $D$ is the token count. Teacher inference forward pass cost is also included if generating targets on-the-fly.
- **Precision Profile:**
  - **Teacher Model:** INT4/INT8 (e.g., GGUF format for memory bandwidth optimization on edge hardware).
  - **Draft Model:** FP16 or INT4.
  - **Distillation Operations:** FP32.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x local NVIDIA RTX 4090 (24GB VRAM) or Apple Silicon M-series GPU.
- **Provider:** Local edge machine.
- **Execution Duration Limit:** 4–8 hours.
- **Target Token/Batch Scale:** 1B–2B tokens of teacher-labeled logits. Batch size 16 at 4K context length.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** Production deployment is local execution on consumer devices. The distillation process runs on a single node GPU cluster (e.g., 8x H100) for fast turnaround.
- **Interconnect Requirements:** Not applicable for deployment; standard PCI-e for training.
- **Persistent Storage Topography:** High-speed local NVMe storage.

## 3. Data Topography
- **Token Window Length:** 8,192 (8K) tokens (synced exactly with the teacher model's context window).
- **Preprocessing Requirements:** Feed datasets through the teacher model to export token-level log probabilities (logits) onto local storage to prevent active forward pass bottlenecks during student training.
- **Tokenizer Handling:** The draft model **must** share the identical vocabulary and tokenizer configuration as the teacher model.

## 4. Run Execution Script
```bash
#!/usr/bin/env bash
# Actionable bash script executing speculative decoding using llama.cpp

set -eo pipefail

# Target teacher model and distilled speculative draft model
TEACHER_MODEL_PATH="./models/qwen2.5-3b-instruct-q8_0.gguf"
DRAFT_MODEL_PATH="./models/qwen2.5-100m-draft-f16.gguf"

# Verify models exist
if [ ! -f "$TEACHER_MODEL_PATH" ] || [ ! -f "$DRAFT_MODEL_PATH" ]; then
  echo "Error: GGUF models not found in ./models/"
  exit 1
fi

echo "Launching llama.cpp Speculative Decoding execution..."

# Run inference with llama.cpp using the speculative draft model
./llama-cli \
  --model "$TEACHER_MODEL_PATH" \
  --draft "$DRAFT_MODEL_PATH" \
  --prompt "Write a Python script to optimize kernel fusions on Apple Silicon NPUs." \
  --n-predict 512 \
  --threads 8 \
  --ctx-size 8192 \
  --temp 0.2 \
  --top-k 40 \
  --top-p 0.95 \
  --n-gpu-layers 99 \
  --draft-max 5 \
  --metrics 1 \
  2>&1 | tee -a "./logs/speculative_inference.log"

# Note: The '--draft-max 5' parameter tells the speculator to draft up to 5 tokens 
# in advance per step. The '--metrics 1' logs the acceptance rate of drafted tokens.
```

## 5. Failure Modes & Recovery
- **Low Draft Acceptance Rate (Performance Regression):**
  - *Indicator:* The speculative validation rate falls below 30% (as logged by llama.cpp metrics). This causes inference speed to become slower than running the teacher model alone due to correction backtracks.
  - *Mitigation:* Re-evaluate the student model dataset. Ensure distillation covers the specific prompt domains used during inference. Increase student size (e.g. from 100M to 300M parameters) or add more training epochs with higher KL-divergence weight.
- **Vocabulary Divergence:**
  - *Indicator:* Execution fails instantly on startup with tokenizer size mismatch errors.
  - *Mitigation:* Ensure that the student model's output layer weight matrices align exactly with the teacher's vocabulary index layout.
- **Rollback Instructions:**
  - Re-run the distillation using the original target logits. Fall back to standard inference without draft parameters (`--draft` omitted) to restore base system functionality.
