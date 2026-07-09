# Blueprint: Attention Mechanism Modification (GQA/MLA Conversion)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Upgrades or alters the attention layer topology of an existing pre-trained model (e.g. converting Multi-Head Attention, MHA, to Grouped-Query Attention, GQA, or Multi-Latent Attention, MLA).
  - GQA reduces the size of the key-value (KV) cache by grouping query heads together to share a single KV head.
  - Convert existing projection weights by average-pooling the weight arrays of group heads:
    $$W_{K\text{-GQA}}^{(g)} = \frac{1}{H/G} \sum_{i=1}^{H/G} W_{K\text{-MHA}}^{(g \cdot (H/G) + i)}$$
    where $H$ is total heads and $G$ is number of target groups.
  - Followed by recovery training on general text to stabilize the new attention mechanics.
- **FLOPs Scaling Formula:**
  - Requires training the model for a short recovery calibration run:
    $$\text{FLOPs} \approx 6 \times N \times D_{\text{stabilize}}$$
    where $D_{\text{stabilize}}$ is 5B–10B tokens.
- **Precision Profile:**
  - **Conversion Script:** FP32 (to maintain weight average precision).
  - **Recovery Training:** BF16.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 8x H100 (SXM5, 80GB) Spot instances.
- **Provider:** Vast.ai.
- **Execution Duration Limit:** < 10 hours.
- **Target Token/Batch Scale:** 5B tokens. Batch size 4M tokens.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 32x H100 SXM5 nodes.
- **Interconnect Requirements:** NVLink (intra-node) and InfiniBand (inter-node) for high speed model checkpoints recovery tuning.
- **Persistent Storage Topography:** High speed local scratch NVMe storage.

## 3. Data Topography
- **Token Window Length:** 8,192 tokens.
- **Preprocessing Requirements:** Standard high-quality general pre-training text (SlimPajama).
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Script converting MHA weights to GQA format by pooling head parameters.

import torch

def convert_mha_weights_to_gqa(state_dict, mha_prefix, num_heads, num_kv_groups):
    # mha_prefix: e.g. "model.layers.0.self_attn"
    # num_heads: e.g. 32
    # num_kv_groups: e.g. 8 (meaning group size = 4 heads per group)
    group_size = num_heads // num_kv_groups
    print(f"Converting MHA projection weights. Group size: {group_size} heads per KV head.")
    
    # 1. Access Key Projection weight
    k_weight = state_dict[f"{mha_prefix}.k_proj.weight"] # Shape: [HiddenDim, HiddenDim]
    hidden_dim = k_weight.shape[0]
    head_dim = hidden_dim // num_heads
    
    # Reshape key weight: [NumHeads, HeadDim, HiddenDim]
    k_weight_reshaped = k_weight.view(num_heads, head_dim, hidden_dim)
    
    # 2. Pool weights across groups
    gqa_k_heads = []
    for g in range(num_kv_groups):
        start_idx = g * group_size
        end_idx = start_idx + group_size
        # Average pooling weights in group
        group_avg = k_weight_reshaped[start_idx:end_idx].mean(dim=0)
        gqa_k_heads.append(group_avg)
        
    gqa_k_weight = torch.stack(gqa_k_heads).view(num_kv_groups * head_dim, hidden_dim)
    
    # Update state dict key projection
    state_dict[f"{mha_prefix}.k_proj.weight"] = gqa_k_weight
    print("Key projection weight successfully converted to GQA shape.")
    
    return state_dict

if __name__ == "__main__":
    # Create mock MHA key projection weight (32 heads, head_dim 128, hidden_dim 4096)
    mock_weight = torch.randn(4096, 4096)
    state = {"model.layers.0.self_attn.k_proj.weight": mock_weight}
    
    converted = convert_mha_weights_to_gqa(
        state_dict=state,
        mha_prefix="model.layers.0.self_attn",
        num_heads=32,
        num_kv_groups=8
    )
    print("New key projection weight shape:", converted["model.layers.0.self_attn.k_proj.weight"].shape)
```

## 5. Failure Modes & Recovery
- **Catastrophic Performance Degradation:**
  - *Indicator:* The model outputs gibberish immediately after conversion, and recovery loss starts extremely high.
  - *Mitigation:* Ensure that query head configurations are left untouched, and only Key and Value projections are grouped. If loss remains high, initialize the GQA projections with random values instead of average pooling, and train for longer.
- **KV Cache Memory Overflow:**
  - *Indicator:* Out of memory during local high-concurrency evaluation steps.
  - *Mitigation:* Verify compiler configuration paths and ensure that sequence length allocation checks account for the group-sharing ratio reduction.
