# SOTA Training & Optimization Process

Training a Large Language Model effectively requires careful optimization strategies to ensure stability, convergence, and hardware efficiency. Our `scratch-llm` training pipeline now implements these industry-standard techniques.

## 1. Mixed Precision Training (BFloat16 / AMP)
Training in full 32-bit floating point (FP32) consumes too much VRAM. Our pipeline uses PyTorch Automatic Mixed Precision (AMP) with `torch.amp.autocast`. 
* **Why it's SOTA:** BFloat16 (BF16) provides the same dynamic range as FP32 (preventing underflow/overflow issues common in FP16) but uses half the memory and computes much faster on modern hardware (NVIDIA Ampere+, Apple Silicon MPS).

## 2. Advanced AdamW Optimizer Groups
We use `AdamW` (Adam with Weight Decay), which separates weight decay from the gradient update step. Crucially, our implementation categorizes parameters into distinct optimization groups.
* **Why it's SOTA:** We strictly apply weight decay *only* to 2D weight matrices (like linear layers). We explicitly disable weight decay for 1D parameters, including biases, RMSNorm scaling weights, and embedding layers, which prevents degradation of the model's normalization and representation capacity.

## 3. Learning Rate Schedulers (Cosine with Linear Warmup)
You cannot use a static learning rate for LLMs. Our pipeline utilizes a custom `LambdaLR` scheduler implementing a **Linear Warmup** followed by a **Cosine Decay**.
* **Warmup:** The LR linearly increases from 0 to its maximum value over the first 10% of training steps. This prevents the model from diverging early on when gradients are chaotic.
* **Cosine Decay:** The LR slowly decreases following a cosine curve down to 10% of the max LR, helping the model settle into a fine-grained minimum.

## 4. Byte-Pair Encoding (BPE) Subword Tokenizer
Instead of naive word splitting, we use Hugging Face's `tokenizers` library to train and deploy a Byte-Pair Encoding (BPE) subword tokenizer.
* **Why it's SOTA:** BPE eliminates out-of-vocabulary (`[UNK]`) errors by breaking unknown words down into subword or byte components, allowing the model to handle morphologically rich languages and code seamlessly.

## 5. Gradient Clipping
During training, certain batches can produce massive gradients (especially in early stages), which can cause the optimizer to take too large a step and "blow up" the model (producing NaNs).
* **Why it's SOTA:** `torch.nn.utils.clip_grad_norm_` caps the total norm of the gradients at a specific threshold (e.g., 1.0). This provides guaranteed stability.
