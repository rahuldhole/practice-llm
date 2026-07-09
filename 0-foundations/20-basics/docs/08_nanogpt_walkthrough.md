# ⛰️ Tutorial 08: nanoGPT Walkthrough

**TLDR:** A step-by-step code walkthrough of NanoGPT implementation.

Now that you have built a Decoder-only Transformer (Tiny GPT), you have all the conceptual knowledge required to read and understand Andrej Karpathy's `nanoGPT` repository. 

This guide connects our educational modules to the optimizations used in a production-ready codebase.

---

## 1. model.py in nanoGPT

Our [TinyGPT](../src/gpt.py#L40) matches the architecture of nanoGPT. However, nanoGPT implements several key optimizations:

### A. Vectorized Attention (QKV Packing)
In our educational code, we created individual heads in a list and ran loops. In `nanoGPT`, all queries, keys, and values are computed in a single large matrix multiplication for speed:
```python
# Project input x to Q, K, V all at once
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
# Split columns by head, then transpose to (B, nh, T, hs)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

### B. Weight Tying
In GPT models, the input token embedding layer and the final output project head (LM Head) share the exact same weights. This:
- Cuts parameter counts in half.
- Encourages the model to place representations in a space that matches output distribution.
In PyTorch, this is configured as:
```python
self.transformer.wte.weight = self.lm_head.weight
```

### C. Regularization (Dropout)
To prevent the model from overfitting, nanoGPT places `nn.Dropout` layers after attention weight calculations and feed-forward networks, randomly silencing activations during training.

---

## 2. train.py in nanoGPT

The training script in `nanoGPT` is optimized to scale to millions of tokens across large clusters of GPUs.

### A. Mixed Precision Training (AMP)
Floating point numbers are normally represented using 32 bits (float32). Mixed precision performs calculations in float16 or bfloat16, which is twice as fast and uses half the GPU memory.
```python
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(X, Y)
```

### B. Gradient Accumulation
If you don't have enough GPU memory to train with a batch size of 64, you can run 4 forward passes with a batch size of 16, accumulate the gradients, and perform the optimizer step once. This simulates a larger virtual batch size:
```python
loss = loss / gradient_accumulation_steps
loss.backward()
if step % gradient_accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

### C. Cosine Learning Rate Decay
Instead of keeping the learning rate constant, we increase it slowly during training (warmup) and then decay it following a cosine curve to a minimum value. This ensures fast early progress and stable convergence at the end of training.

---

## 💡 Practical Challenge
You are now ready to tackle nanoGPT! Clone the repo and look at `model.py` and `train.py`. You will recognize LayerNorm, Block, FeedForward, and Causal Attention. Try modifying the model config in nanoGPT, train it on Shakespeare, and see your customized AI model generate text!
