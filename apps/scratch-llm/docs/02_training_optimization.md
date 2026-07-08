# SOTA Training & Optimization Process

Training a Large Language Model effectively requires careful optimization strategies to ensure stability, convergence, and hardware efficiency.

## 1. AdamW Optimizer
While `Adam` is popular, it handles L2 regularization improperly. `AdamW` (Adam with Weight Decay) separates weight decay from the gradient update step.
* **Why it's SOTA:** AdamW significantly improves generalization performance and is universally used for training LLMs. Typically, parameters like LayerNorm/RMSNorm weights and biases are excluded from weight decay.

## 2. Learning Rate Schedulers (Cosine with Warmup)
You cannot use a static learning rate for LLMs. SOTA models use a **Linear Warmup** followed by a **Cosine Decay**.
* **Warmup:** The LR linearly increases from 0 to its maximum value over the first few thousand steps. This prevents the model from diverging early on when gradients are chaotic.
* **Cosine Decay:** The LR slowly decreases following a cosine curve down to a small fraction (usually 10%) of the max LR. This helps the model settle into a fine-grained minimum.

## 3. Gradient Clipping
During training, certain batches can produce massive gradients (especially in early stages or due to anomalous data), which can cause the optimizer to take too large a step and "blow up" the model (producing NaNs).
* **Why it's SOTA:** `torch.nn.utils.clip_grad_norm_` caps the total norm of the gradients at a specific threshold (e.g., 1.0). This provides guaranteed stability.

## 4. Mixed Precision Training (FP16 / BF16)
Training in full 32-bit floating point (FP32) is too slow and consumes too much VRAM. SOTA models are trained in FP16 or, more commonly now, BFloat16 (BF16).
* **Why it's SOTA:** BF16 provides the same dynamic range as FP32 (preventing underflow/overflow issues common in FP16) but uses half the memory and computes much faster on modern hardware (like NVIDIA Ampere+ GPUs or TPUs).

## 5. Distributed Training (FSDP / DeepSpeed)
To train models larger than what fits on a single GPU (e.g., a 7B model requires ~14GB just for weights, not including optimizer states and activations), frameworks use Fully Sharded Data Parallel (FSDP) or DeepSpeed ZeRO.
* **Why it's SOTA:** These frameworks shard the model parameters, gradients, and optimizer states across hundreds or thousands of GPUs, communicating only when necessary, allowing massive scale-up.
