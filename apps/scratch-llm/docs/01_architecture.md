# SOTA LLM Architecture Process

This document explains the core architectural components that differentiate a modern State-of-the-Art (SOTA) Large Language Model (like LLaMA 3 or Mistral) from a standard vanilla Transformer. Our `scratch-llm` codebase now actively implements all of these components!

## 1. Grouped Query Attention (GQA) & FlashAttention
Instead of a single attention mechanism, MHA splits queries, keys, and values into multiple "heads". We have implemented **Grouped Query Attention (GQA)**, an evolution of MHA where multiple query heads share a single key/value head (e.g., 4 Q-heads, 2 KV-heads), which significantly reduces the size of the KV cache and memory bandwidth during inference.
* **FlashAttention:** Our codebase utilizes PyTorch's native `F.scaled_dot_product_attention`, automatically dispatching to FlashAttention-2 or Memory-Efficient Attention kernels for rapid, memory-optimized computation.

## 2. Inference KV Cache
During generation, feeding the entire sequence of tokens repeatedly into the model yields an $O(N^2)$ bottleneck. We have implemented a state-of-the-art **KV Cache** mechanism. By storing the keys and values from previous tokens, the model only passes the single new generated token through the attention mechanism, improving generation complexity to $O(N)$.

## 3. Rotary Positional Embeddings (RoPE)
Original transformers added a static sine/cosine positional encoding to the token embeddings. RoPE instead applies a mathematical rotation to the Queries and Keys *during* the attention computation.
* **Why it's SOTA:** RoPE provides a better mathematical inductive bias for relative positions (token distance matters more than absolute position) and extrapolates better to longer sequences than it was trained on.

## 4. RMSNorm (Root Mean Square Normalization)
Instead of `LayerNorm` which centers the data (subtracts mean) and scales it (divides by variance), `RMSNorm` only scales the data by its Root Mean Square.
* **Why it's SOTA:** It is computationally cheaper and performs identically to or slightly better than LayerNorm.

## 5. Pre-Norm vs Post-Norm
Original transformers applied normalization *after* the residual connection. Modern models apply normalization *before* the attention/FFN block.
* **Why it's SOTA:** Pre-Norm keeps the residual pathway completely clean, leading to significantly better training stability and allowing for much deeper networks without vanishing gradients.

## 6. SwiGLU Activation Function
The Feed-Forward Network (FFN) uses SwiGLU. SwiGLU uses the Swish activation function and a gating mechanism.
* **Why it's SOTA:** SwiGLU yields better performance across benchmarks compared to standard ReLU or GELU.
