# Blueprint: Uptraining & Model Layer Stacking (Dense to Larger Dense)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Transition a fully trained small base model checkpoint (e.g., 3B parameters) into a larger target model size (e.g., 7B parameters) by repeating or interpolating intermediate layers.
  - Alleviates the compute cost of training the larger architecture from scratch.
  - **Loss Target:** Standard Cross-Entropy autoregressive pre-training loss on a small stabilization dataset to blend the repeated layers.
- **FLOPs Scaling Formula:**
  - Stabilization requires running the scaled model on a subset of tokens.
  - $\text{FLOPs} \approx 6 \times N_{\text{target}} \times D_{\text{stabilize}}$ where $N_{\text{target}}$ is the parameter count of the target model (~7 Billion) and $D_{\text{stabilize}}$ is 10B–20B tokens.
- **Precision Profile:**
  - **Weights and Activations:** BF16/FP16 (matching base model precision).
  - **Optimizer States:** FP32 (mixed-precision optimization).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 8x H100 (SXM5, 80GB) Spot instances.
- **Provider:** Vast.ai.
- **Execution Duration Limit:** < 4 hours.
- **Target Token/Batch Scale:** 1B-5B tokens. Batch size scaled to 4M tokens.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** Multi-node 8x H100 nodes (e.g., 32x H100 total).
- **Interconnect Requirements:** NVLink (intra-node) and InfiniBand (inter-node) for FSDP or DeepSpeed ZeRO-3.
- **Persistent Storage Topography:** High speed local scratch NVMe with automated background backup to Cloudflare R2.

## 3. Data Topography
- **Token Window Length:** Matching target model maximum window (e.g. 8k-32k).
- **Preprocessing Requirements:** High-quality general pre-training mixture (e.g., SlimPajama, FineWeb-Edu) to stabilize representation alignment.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Script to duplicate/stack layers of a transformer model to increase capacity.

import torch
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer

def stack_model_layers(base_model_path, output_path, target_layers=40):
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    config = model.config
    
    # Identify the stackable transformer blocks (e.g., model.model.layers for Llama)
    layers = model.model.layers
    num_base_layers = len(layers)
    print(f"Base model has {num_base_layers} layers. Stacking to {target_layers} layers...")
    
    # Calculate mapping (e.g., repeat layers uniformly)
    # Simple tiling: repeat layers to fill up target
    new_layers = torch.nn.ModuleList()
    for i in range(target_layers):
        base_idx = int(i * (num_base_layers / target_layers))
        print(f"Mapping new layer {i} from base layer {base_idx}")
        copied_layer = copy.deepcopy(layers[base_idx])
        new_layers.append(copied_layer)
        
    model.model.layers = new_layers
    config.num_hidden_layers = target_layers
    
    print(f"Saving stacked model to {output_path}...")
    model.save_pretrained(output_path)
    print("Done! Ready for stabilization training.")

if __name__ == "__main__":
    # Example stack of a 3B model (e.g., 28 layers) to a 5B model (e.g., 40 layers)
    stack_model_layers("Qwen/Qwen2.5-3B", "./stacked_qwen_5b", target_layers=40)
```

## 5. Failure Modes & Recovery
- **Representation Mismatch / High Initial Loss:**
  - *Indicator:* The loss spike at step 0 is extremely high compared to the base model.
  - *Mitigation:* Apply a small scaling noise to repeated layer weights or run the first 500 steps with a frozen trunk (only training block boundaries) before unfreezing all parameters.
- **Catastrophic Representation Collapse:**
  - *Indicator:* Activations saturate or explode to NaN rapidly in the deeper duplicated layers.
  - *Mitigation:* Implement weight decay and lower the peak learning rate of stabilization (e.g., 1e-5 to 3e-5).
