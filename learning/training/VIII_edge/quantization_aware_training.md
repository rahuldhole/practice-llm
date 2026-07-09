# Blueprint: Quantization-Aware Training (QAT)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Auto-regressive Cross-Entropy Loss combined with a Quantization MSE loss to match the distribution of simulated low-bit weights:
    $$\min_{W, s} \mathcal{L}_{\text{CE}}(Y, f(X; \hat{W})) + \lambda \|W - \hat{W}\|_2^2$$
    where the quantized weight is defined as $\hat{W} = s \cdot \text{clamp}\left(\text{round}\left(\frac{W}{s}\right), -q_{\max}, q_{\max}\right)$.
  - **Straight-Through Estimator (STE):** Since the rounding operation has a gradient of 0 almost everywhere, we approximate:
    $$\frac{\partial \hat{W}}{\partial W} \approx I$$
    allowing gradients to flow directly back to the latent FP32 weight tensors during optimization.
- **FLOPs Scaling Formula:**
  - Quantization emulation adds small computational overhead to the forward pass:
    $$\text{FLOPs} \approx 6.5 \times N \times D$$
    where $N$ is the parameter count and $D$ is the token count.
- **Precision Profile:**
  - **Latent Master Weights:** FP32 (retained for gradient updates).
  - **Simulated Quantized Weights:** INT4 or INT2 projection.
  - **Activations and Gradients:** FP16/BF16 (optimized for Apple M-series Unified Memory or GPU tensor cores).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x local Apple Mac Studio (M2 Ultra / M3 Max with unified memory, minimum 64GB RAM).
- **Provider:** Local Apple Silicon hardware.
- **Execution Duration Limit:** 4–8 hours.
- **Target Token/Batch Scale:** 500M–1B tokens calibration corpus. Batch size scaled to match unified memory bandwidth (e.g., batch size 16 at 8K context).

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** Local farm or private cloud of 16x Mac Studio (M2 Ultra 192GB) nodes orchestrated via local cluster management, or private server nodes utilizing tensor-core GPUs executing INT4 scale checks.
- **Interconnect Requirements:** 10 GbE local LAN.
- **Persistent Storage Topography:** High-speed PCIe Gen 4 NVMe arrays directly connected to local developer systems.

## 3. Data Topography
- **Token Window Length:** 8,192 (8K) tokens (standard constraint for consumer edge deployments to prevent KV-cache explosion).
- **Preprocessing Requirements:** High-quality representative subset of the pre-training corpus to avoid bias collapse under severe quantization.
- **Tokenizer Handling:** Freeze model-native tokenizer (e.g., Llama/Qwen native tokenizer).

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Actionable Python script using MLX (Apple Silicon) to perform QAT calibration.

import mlx.core as mx
import mlx.nn as nn
import numpy as np

class QATLinear(nn.Module):
    def __init__(self, input_dims: int, output_dims: int, bits: int = 4):
        super().__init__()
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = (2 ** (bits - 1)) - 1
        
        # Initialize latent FP32 weight
        self.weight = mx.random.normal((output_dims, input_dims)) * 0.02
        self.scale = mx.array([1.0])

    def get_quantized_weight(self):
        # Simulated Quantization with Straight-Through Estimator simulation
        w_scaled = self.weight / self.scale
        w_clipped = mx.clip(w_scaled, self.qmin, self.qmax)
        w_rounded = mx.round(w_clipped)
        
        # MLX passes gradients through operations automatically, 
        # but to mimic STE we bypass gradient truncation:
        w_qat = self.scale * (w_rounded + (self.weight / self.scale - mx.stop_gradient(self.weight / self.scale)))
        return w_qat

    def __call__(self, x):
        w = self.get_quantized_weight()
        return mx.matmul(x, w.T)

# Example Training Loop
model = QATLinear(1024, 1024, bits=4)

def loss_fn(model, x, y):
    pred = model(x)
    return mx.mean(mx.square(pred - y))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# Mock input data
x = mx.random.normal((32, 1024))
y = mx.random.normal((32, 1024))

optimizer = nn.Adam(learning_rate=1e-4)

# Training step
for step in range(100):
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss.item():.6f}")
```

## 5. Failure Modes & Recovery
- **Quantization Shattering (NaN/Inf Losses):**
  - *Indicator:* Output activations explode, training loss turns to NaN or Inf instantly.
  - *Mitigation:* Reduce scale factor learning rates. Implement a "gradual quantization" scheduler where weights are partially quantized (e.g. starting with FP16 gradients, moving from INT8 down to INT4 over the first 2,000 steps).
- **Steep Accuracy Degradation (Poor Core Reasoning):**
  - *Indicator:* Evaluation perplexity spikes heavily on downstream reasoning tasks despite low training loss.
  - *Mitigation:* Set the coefficient $\lambda$ of weight distribution distance loss higher to ensure the quantized weight distribution stays close to the original floating-point distribution.
- **Rollback Instructions:**
  - Automatically checkpoint local state every 500 steps. If NaN or anomaly is detected, revert to the last local checkpoint on local NVMe storage and scale down the optimizer step size.
