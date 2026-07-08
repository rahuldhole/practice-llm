# SOTA LLM Architecture Process

This document explains the core architectural components that differentiate a modern State-of-the-Art (SOTA) Large Language Model (like LLaMA 3 or Mistral) from a standard vanilla Transformer.

## 1. Multi-Head Attention (MHA)
Instead of a single attention mechanism, MHA splits the queries, keys, and values into multiple "heads." Each head can learn to pay attention to different aspects of the sequence (e.g., one head for syntax, another for semantics). 
* **Grouped Query Attention (GQA):** An evolution of MHA where multiple query heads share a single key/value head, significantly reducing memory bandwidth during inference.

## 2. Rotary Positional Embeddings (RoPE)
Original transformers added a static sine/cosine positional encoding to the token embeddings. RoPE instead applies a mathematical rotation to the Queries and Keys *during* the attention computation.
* **Why it's SOTA:** RoPE provides a better mathematical inductive bias for relative positions (token distance matters more than absolute position) and extrapolates better to longer sequences than it was trained on.

## 3. RMSNorm (Root Mean Square Normalization)
Instead of `LayerNorm` which centers the data (subtracts mean) and scales it (divides by variance), `RMSNorm` only scales the data by its Root Mean Square.
* **Why it's SOTA:** It is computationally cheaper and performs identically to or slightly better than LayerNorm.

## 4. Pre-Norm vs Post-Norm
Original transformers applied normalization *after* the residual connection (`x = LayerNorm(x + Attention(x))`). Modern models apply normalization *before* the attention/FFN block (`x = x + Attention(RMSNorm(x))`).
* **Why it's SOTA:** Pre-Norm keeps the residual pathway completely clean, leading to significantly better training stability and allowing for much deeper networks without vanishing gradients.

## 5. SwiGLU Activation Function
The Feed-Forward Network (FFN) in a SOTA model usually replaces the standard ReLU activation with SwiGLU. SwiGLU uses the Swish activation function and a gating mechanism.
* **Why it's SOTA:** SwiGLU has empirically proven to yield better performance across multiple benchmarks compared to standard ReLU or GELU implementations, though it requires three weight matrices instead of two.
