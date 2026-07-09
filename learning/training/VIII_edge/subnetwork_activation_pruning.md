# Blueprint: Sub-Network Activation Pruning (Read-ME Strategy)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Cluster and refactor the feedforward network (FFN) layers of a pre-trained dense LLM into sparse, non-overlapping experts based on activation sparsity metrics collected over a calibration dataset $D_c$:
    $$\min_{E, \theta_R} \sum_{x \in D_c} \left\| FFN(x; W_{\text{dense}}) - \sum_{i \in E_{\text{active}}} \text{expert}_i(x; W_{e_i}) \right\|_2^2$$
    where $W_{\text{dense}}$ represents the original dense FFN weights, and $W_{e_i}$ represents the split expert weights clustered from $W_{\text{dense}}$ based on historical neuron activation patterns.
  - **Read-ME Clustering:** Run calibration data through the model to monitor the intermediate post-activation values. Neurons that fire together are grouped into a common expert. Zero-weight paths or low-frequency activation channels are pruned to compress the total parameter footprint.
- **FLOPs Scaling Formula:**
  - Converts a dense $N$-parameter model into a sparse MoE with active parameter footprint $N_{\text{active}}$ (e.g. 3B Dense $\rightarrow$ 3B Total / 800M Active MoE), reducing active parameters and execution FLOPs:
    $$\text{FLOPs/token} \approx 6 \times N_{\text{active}} \times D$$
- **Precision Profile:**
  - **Calibration & Profiling:** FP32 (to preserve tiny activation differences).
  - **Pruned Experts:** INT4/INT2 GGUF (optimized for edge memory footprint).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x local Apple Mac Studio (minimum 32GB Unified RAM) or 1x RTX 4090 GPU.
- **Provider:** Local edge machine.
- **Execution Duration Limit:** 4–8 hours calibration and clustering.
- **Target Token/Batch Scale:** 1 Billion token calibration corpus.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 1x node server with 8x H100 (80GB) GPUs to accelerate high-dimensional activation clustering sweeps over larger base models (e.g., 70B Dense $\rightarrow$ 70B MoE).
- **Interconnect Requirements:** PCI-e Gen 5 interfaces.
- **Persistent Storage Topography:** High-speed NVMe storage array.

## 3. Data Topography
- **Token Window Length:** 8,192 (8K) tokens.
- **Preprocessing Requirements:** Curate a highly representative calibration subset spanning code, reasoning, and conversational text to prevent over-pruning core model features.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Actionable Python script using MLX to profile neuron activations and split FFN weights.

import mlx.core as mx
import mlx.nn as nn
import numpy as np

class DenseFFN(nn.Module):
    def __init__(self, dims: int, hidden_dims: int):
        super().__init__()
        self.w1 = nn.Linear(dims, hidden_dims)
        self.w2 = nn.Linear(hidden_dims, dims)

    def __call__(self, x):
        return self.w2(mx.sigmoid(self.w1(x)))

# Mock a pre-trained dense FFN
dense_ffn = DenseFFN(dims=128, hidden_dims=1024)
mx.eval(dense_ffn.parameters())

# 1. Profile activation frequencies over calibration data
calibration_inputs = mx.random.normal((100, 32, 128)) # [batches, seq_len, dims]
activation_frequencies = mx.zeros((1024,))

for batch in calibration_inputs:
    # Forward pass up to intermediate activation layer
    h = mx.sigmoid(dense_ffn.w1(batch)) # [seq_len, hidden_dims]
    # Check which neurons are active (above a threshold, e.g., 0.1)
    is_active = mx.mean(mx.clip(h, 0.0, 1.0), axis=0) > 0.1
    activation_frequencies += is_active

mx.eval(activation_frequencies)

# 2. Identify dead or low-frequency neurons to prune (e.g., bottom 10%)
pruning_threshold = np.percentile(np.array(activation_frequencies), 10)
active_indices = mx.array([i for i, freq in enumerate(activation_frequencies) if freq > pruning_threshold])

print(f"Original hidden dimensions: 1024")
print(f"Pruned hidden dimensions: {active_indices.shape[0]}")

# 3. Create a pruned sparse weight matrix representation
pruned_w1_weight = dense_ffn.w1.weight[active_indices, :]
pruned_w2_weight = dense_ffn.w2.weight[:, active_indices]

# Define new pruned FFN module
pruned_ffn = DenseFFN(dims=128, hidden_dims=active_indices.shape[0])
pruned_ffn.w1.weight = pruned_w1_weight
pruned_ffn.w2.weight = pruned_w2_weight
mx.eval(pruned_ffn.parameters())

print("Refactored pruned FFN initialized successfully.")
```

## 5. Failure Modes & Recovery
- **Core Skill Loss (Over-pruning critical neurons):**
  - *Indicator:* The model's perplexity on general validation tasks (like reasoning or coding) spikes heavily, indicating critical knowledge neurons were mistakenly pruned.
  - *Mitigation:* Increase the active threshold limits or broaden the diversity of the calibration dataset. Keep a minimum active neuron count (at least 85% of original hidden dimensions) to prevent performance degradation.
- **Router Load Imbalance (Expert Clustering Collision):**
  - *Indicator:* After splitting FFN columns into experts, a subset of experts receives 95% of active tokens, causing severe memory queuing.
  - *Mitigation:* Re-run the clustering algorithm using balanced clustering constraints (K-means with size restrictions) to enforce uniform expert volume sizing.
- **Rollback Instructions:**
  - Revert the model configurations to the original dense architecture and decrease pruning parameters (e.g. prune 5% instead of 15%).
