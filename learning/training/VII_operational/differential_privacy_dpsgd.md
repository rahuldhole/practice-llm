# Blueprint: Differential Privacy Training (DP-SGD)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Prevent the model from memorizing sensitive individual samples (e.g. personal customer logs or medical keys) during adaptation.
  - Aligns training with Differential Privacy guarantees by clipping per-sample gradients and adding Gaussian noise before optimization:
    1. For each sample $i$, compute gradient $g_i$.
    2. Clip gradients using threshold $C$:
       $$\bar{g}_i = g_i / \max\left(1, \frac{\|g_i\|_2}{C}\right)$$
    3. Sum clipped gradients and add Gaussian noise:
       $$\tilde{g} = \sum_{i=1}^B \bar{g}_i + \mathcal{N}\left(0, \sigma^2 C^2 I\right)$$
    4. Update weights using $\tilde{g}$.
- **FLOPs Scaling Formula:**
  - PyTorch standard backpropagation accumulates gradients, which is fast. DP-SGD requires computing gradients *separately for each sample* in a batch, which bypasses hardware parallel optimization, adding ~2-3x compute overhead:
    $$\text{FLOPs} \approx 2.5 \times (6 \times N \times D)$$
- **Precision Profile:**
  - **Model weights & forward pass:** BF16/FP16.
  - **Per-sample gradient clipping & noise addition:** FP32 (to prevent precision degradation during gradient manipulation).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM) or 1x A100.
- **Provider:** RunPod.
- **Execution Duration Limit:** < 8 hours.
- **Target Token/Batch Scale:** 1B tokens. Batch size of 32 at 2k context.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 4x H100 (SXM5, 80GB) node cluster.
- **Interconnect Requirements:** NVLink (intra-node) for high-speed weights sync.
- **Persistent Storage Topography:** High speed local scratch NVMe storage.

## 3. Data Topography
- **Token Window Length:** 2,048 tokens.
- **Preprocessing Requirements:** General dataset contains private keys or personal identification records that must not leak.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Conceptual script using Opacus to wrap PyTorch model for DP-SGD training.

from opacus import PrivacyEngine
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def run_dpsgd_training():
    # 1. Setup simple model
    model = nn.Sequential(
        nn.Embedding(1000, 128),
        nn.Linear(128, 1000)
    ).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    # Create dummy dataset
    x = torch.randint(0, 1000, (64, 16))
    y = torch.randint(0, 1000, (64, 16))
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=8)
    
    # 2. Attach Privacy Engine (Opacus)
    # Automatically wraps model and optimizer to compute per-sample gradients, clip them, and add noise.
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.0,  # Controls DP epsilon budget
        max_grad_norm=1.0,     # Gradient clipping threshold C
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training step
    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        logits = model(batch_x)
        loss = criterion(logits.view(-1, 1000), batch_y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Output current privacy budget consumed (epsilon)
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"Batch Loss: {loss.item():.4f} | Privacy Spent (Epsilon): {epsilon:.2f}")
        break

if __name__ == "__main__":
    run_dpsgd_training()
```

## 5. Failure Modes & Recovery
- **Catastrophic Accuracy Drop:**
  - *Indicator:* The training loss refuses to descend, or the model outputs gibberish due to excessive noise overriding actual gradients.
  - *Mitigation:* Epsilon budget may be too strict. Reduce the `noise_multiplier` or increase batch size (large batch sizes yield more stable gradient sums, reducing relative noise distortion).
- **GPU Out of Memory (OOM) due to Per-Sample Gradients:**
  - *Indicator:* Immediate OOM during first backward pass.
  - *Mitigation:* Reduce physical batch size while keeping logical batch size large via gradient accumulation. Verify that no custom layer implementation is caching unneeded intermediate tensors.
