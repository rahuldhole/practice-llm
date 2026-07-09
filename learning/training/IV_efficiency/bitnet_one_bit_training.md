# Blueprint: BitNet / 1-bit Pre-training

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Train foundation models from scratch where weight matrices are restricted to ternary/1.58-bit values ($-1, 0, 1$), replacing expensive floating-point multiplications with simple additions.
  - During the forward pass, weights are quantized to ternary values using a scaling factor $\beta$:
    $$W_{\text{quant}} = \text{Round}\left(\text{Clip}\left(\frac{W}{\beta}, -1, 1\right)\right)$$
    where $\beta = \frac{1}{d_{\text{in}} \times d_{\text{out}}} \sum_{i,j} |W_{i,j}|$.
  - In the backward pass, gradients are computed relative to full-precision latent weights (using the Straight-Through Estimator, STE) which are updated by the optimizer.
- **FLOPs Scaling Formula:**
  - Replacing matrix multiplications with addition/subtraction operations reduces energy footprint by up to 10x.
  - Aligns with an equivalent FLOP count of:
    $$\text{FLOPs} \approx 0.1 \times (6 \times N \times D)$$
- **Precision Profile:**
  - **Ternary Weights (Forward Pass):** INT2 (packed representation).
  - **Latent Master Weights & Gradients:** FP32/BF16.
  - **Activations:** INT8 (quantized dynamically).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM) or 1x A10G.
- **Provider:** RunPod.
- **Execution Duration Limit:** < 8 hours.
- **Target Token/Batch Scale:** 100M-500M tokens.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 SXM5 node.
- **Interconnect Requirements:** NVLink (intra-node) for high speed gradient accumulation.
- **Persistent Storage Topography:** High speed local scratch NVMe.

## 3. Data Topography
- **Token Window Length:** 8,192 tokens.
- **Preprocessing Requirements:** Standard pre-training corpus (e.g. FineWeb).
- **Tokenizer Handling:** Frozen base model tokenizer (e.g., Llama-3).

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# PyTorch implementation of the BitLinear (1.58-bit) layer using STE.

import torch
import torch.nn as nn

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Round/sign mapping
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator: pass gradients unchanged
        return grad_output

class BitLinear(nn.Linear):
    def forward(self, x):
        # 1. Quantize weights to {-1, 0, 1}
        w = self.weight
        beta = w.abs().mean()
        w_quant = w / (beta + 1e-9)
        # Apply STE rounding to keep gradients flowing
        w_bit = SignSTE.apply(torch.clamp(w_quant, -1.0, 1.0)) * beta
        
        # 2. Quantize activations to 8-bit integers
        x_scale = 127.0 / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x_quant = torch.clamp(x * x_scale, -128, 127).round() / x_scale
        
        # 3. Perform forward pass linear operation
        return nn.functional.linear(x_quant, w_bit, self.bias)

if __name__ == "__main__":
    linear = BitLinear(in_features=512, out_features=512)
    x = torch.randn(8, 64, 512)
    y = linear(x)
    print("BitLinear output shape:", y.shape)
```

## 5. Failure Modes & Recovery
- **Gradient Explosion / Instability:**
  - *Indicator:* Loss jumps to NaN or gradients vanish/explode during the first few thousand training steps.
  - *Mitigation:* Ensure activations are scaled properly at each layer boundary. Introduce LayerNorm layers before every quantization operation.
- **Low Capacity Representation Saturation:**
  - *Indicator:* Training loss plateaus early at a significantly higher value compared to FP16 baselines.
  - *Mitigation:* BitNet requires larger intermediate dimension widths (e.g. SwiGLU hidden dimensions scaled up by 1.3x) to match FP16 model capacity.
