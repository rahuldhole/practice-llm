# Blueprint: Multi-Token Prediction (MTP) Drafting

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Multi-Token Prediction extends standard autoregressive training by predicting multiple future tokens in parallel from a shared representations layer. The total loss is:
    $$\mathcal{L} = \mathcal{L}_{\text{next}} + \sum_{k=1}^{K} \lambda_k \mathcal{L}_{t+k+1}$$
    where $\mathcal{L}_{\text{next}}$ is the standard next-token cross-entropy loss and each $\mathcal{L}_{t+k+1}$ is the cross-entropy loss for predicting the token $k$-steps ahead using specialized MTP predictor heads.
- **FLOPs Scaling Formula:**
  - Standard autoregressive FLOPs scale as $6 \times N \times D$. Training with $K$ additional heads adds projection overhead:
    $$\text{FLOPs} \approx (6 + 0.15 \cdot K) \times N \times D$$
    where $K$ is the number of speculation tokens (usually $K=2$ or $K=4$).
- **Precision Profile:**
  - **Shared Trunk Layers:** FP16/BF16.
  - **MTP Prediction Heads:** FP16/BF16.
  - **Loss & Routing Logic:** FP32.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x Apple Mac Studio or MacBook Pro (M3 Max, minimum 32GB Unified Memory).
- **Provider:** Local Apple Silicon.
- **Execution Duration Limit:** 4–8 hours.
- **Target Token/Batch Scale:** 500M–1B tokens. Batch size of 8 or 16 at 8K context.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 4x node cluster of Apple Silicon Mac Studio devices or edge-servers equipped with native NPU/GPU arrays.
- **Interconnect Requirements:** Unified memory architecture (high bandwidth LPDDR5, up to 800 GB/s per node).
- **Persistent Storage Topography:** High speed local NVMe SSD (minimum Read/Write >7 GB/s).

## 3. Data Topography
- **Token Window Length:** 8,192 (8K) tokens.
- **Preprocessing Requirements:** Data pipelines must package token inputs along with multiple offset target lists ($y_{t+1}, y_{t+2}, \dots$) to compute secondary losses efficiently without redundant index lookup.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Actionable Python script using MLX to define and train an MTP model.

import mlx.core as mx
import mlx.nn as nn

class MTPModel(nn.Module):
    def __init__(self, vocab_size: int, dims: int, num_heads: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.dims = dims
        self.num_heads = num_heads
        
        # Base language model layers
        self.emb = nn.Embedding(vocab_size, dims)
        self.transformer_layer = nn.TransformerEncoderLayer(dims, 8)
        self.lm_head = nn.Linear(dims, vocab_size)
        
        # MTP prediction heads
        self.mtp_heads = [nn.Linear(dims, vocab_size) for _ in range(num_heads)]

    def __call__(self, x):
        h = self.emb(x)
        h = self.transformer_layer(h)
        
        # Autoregressive next token logit
        logits = self.lm_head(h)
        
        # MTP future token logits
        mtp_logits = [head(h) for head in self.mtp_heads]
        return logits, mtp_logits

# Loss function mapping multiple future offsets
def loss_fn(model, x, targets, lambdas):
    # targets shape: [num_heads + 1, batch, seq_len]
    logits, mtp_logits = model(x)
    
    # 1. Base autoregressive loss
    loss = nn.losses.cross_entropy(logits, targets[0])
    loss = mx.mean(loss)
    
    # 2. MTP future losses
    for i, mtp_log in enumerate(mtp_logits):
        h_loss = nn.losses.cross_entropy(mtp_log, targets[i+1])
        loss = loss + lambdas[i] * mx.mean(h_loss)
        
    return loss

# Training loop simulation
model = MTPModel(vocab_size=1000, dims=256, num_heads=2)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
optimizer = nn.Adam(learning_rate=1e-4)

# Mock token sequence and offset targets
x = mx.random.randint(0, 1000, (4, 128))
targets = [
    mx.random.randint(0, 1000, (4, 128)), # t + 1
    mx.random.randint(0, 1000, (4, 128)), # t + 2
    mx.random.randint(0, 1000, (4, 128))  # t + 3
]
lambdas = [0.3, 0.1]

for step in range(50):
    loss, grads = loss_and_grad_fn(model, x, targets, lambdas)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if step % 10 == 0:
        print(f"MTP Step {step}: Loss = {loss.item():.4f}")
```

## 5. Failure Modes & Recovery
- **Primary Head Divergence (Baseline Quality Drop):**
  - *Indicator:* The model's baseline next-token reasoning and perplexity collapse when MTP training starts.
  - *Mitigation:* Reduce the hyperparameter weight values ($\lambda_k$). Set $\lambda_1 = 0.2$ and $\lambda_2 = 0.05$. Freeze base transformer layers during early epochs and only train the projection heads first.
- **Latency Bloat during Execution (Inference Slowdowns):**
  - *Indicator:* Local edge tokens per second (t/s) drop due to activation serialization of the MTP heads.
  - *Mitigation:* Ensure that the MTP heads are only compiled for execution during speculative execution passes. Combine heads into a single batched GEMM layer to run concurrent outputs on the Apple Silicon GPU/NPU cores.
- **Rollback Instructions:**
  - Save checkpoints every 1,000 steps locally. If divergence is observed in the primary loss, restore from local NVMe storage, halve the learning rate of the head projections, and restart training.
