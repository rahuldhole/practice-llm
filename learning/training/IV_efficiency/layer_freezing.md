# Blueprint: Layer Freezing (Progressive Freezing)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Speed up training and reduce GPU memory (VRAM) footprint by freezing the weights of early transformer blocks during fine-tuning.
  - Alleviates optimizer state storage requirements. Standard AdamW optimizer requires 8 bytes per parameter; freezing $M$ out of $L$ layers saves $M \times 8 \text{ bytes per parameter}$ of memory.
  - Gradients are only calculated and backpropagated through the active (unfrozen) layers:
    $$\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta \nabla_{\theta^{(l)}} \mathcal{L} \quad \text{for } l \ge M$$
    $$\theta_{t+1}^{(l)} = \theta_t^{(l)} \quad \text{for } l < M$$
- **FLOPs Scaling Formula:**
  - Autoregressive training FLOPs (forward + backward) scale as $6 \times N \times D$. Freezing layers reduces backward pass requirements (which is normally $4 \times N \times D$).
  - For $f$ fraction of frozen layers:
    $$\text{FLOPs} \approx (2 + (1 - f) \times 4) \times N \times D$$
    If 50% of the layers are frozen ($f=0.5$), FLOPs drop to $4 \times N \times D$ (a 33% total compute saving).
- **Precision Profile:**
  - **Frozen Layers:** FP16/BF16 (no gradient calculations or optimizer states).
  - **Active Layers:** BF16/FP16 weights, FP32 optimizer states.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM).
- **Provider:** RunPod or local machine.
- **Execution Duration Limit:** < 4 hours.
- **Target Token/Batch Scale:** 10M-50M tokens.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 SXM5 node.
- **Interconnect Requirements:** NVLink (intra-node).
- **Persistent Storage Topography:** High speed local scratch NVMe storage.

## 3. Data Topography
- **Token Window Length:** 8,192 tokens.
- **Preprocessing Requirements:** Standard QA/SFT formatting.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# PyTorch script demonstrating how to freeze early layers of a transformer model.

import torch
from transformers import AutoModelForCausalLM

def load_and_freeze_model(model_name, freeze_fraction=0.5):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    
    # Access layers (e.g. model.model.layers for Llama/Qwen)
    layers = model.model.layers
    num_layers = len(layers)
    num_to_freeze = int(num_layers * freeze_fraction)
    
    print(f"Total layers: {num_layers}. Freezing the first {num_to_freeze} layers...")
    
    # Freeze early layers
    for i in range(num_to_freeze):
        for param in layers[i].parameters():
            param.requires_grad = False
            
    # Keep later layers active
    for i in range(num_to_freeze, num_layers):
        for param in layers[i].parameters():
            param.requires_grad = True
            
    # Verify parameter gradients status
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {total_params} ({trainable_params/total_params:.1%})")
    
    return model

if __name__ == "__main__":
    load_and_freeze_model("Qwen/Qwen2.5-3B", freeze_fraction=0.5)
```

## 5. Failure Modes & Recovery
- **Catastrophic Forgetting of High-Level Semantics:**
  - *Indicator:* Fine-tuning performance is poor, or the model loses domain reasoning capacity.
  - *Mitigation:* Do not freeze the last 2-3 layers of the model or the final output projection head. If alignment fails, decrease the freeze fraction (e.g., from 0.8 to 0.5).
- **Gradient Stagnation / Bad Convergence:**
  - *Indicator:* Training loss converges slowly or gets stuck at a high plateau.
  - *Mitigation:* Ensure learning rate is slightly higher than standard full-parameter fine-tuning (e.g., 2e-5 instead of 1e-5) since fewer parameters are adjusting to absorb loss signals.
