# Blueprint: Data Mixing & Domain Weighting Optimization (DoReMi)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Optimizes the proportional weights of data domains (e.g., code, web, books, math) to maximize overall model capabilities and minimize average training losses.
  - Aligns with the DoReMi framework:
    1. Train a reference model $\pi_{\text{ref}}$ on an initial baseline mixture.
    2. Train a small proxy model $\pi_{\text{proxy}}$ that learns to minimize the excess loss relative to the reference model over each domain $d$:
       $$\mathcal{L}_d(\theta) = \mathbb{E}_{x \sim \mathcal{D}_d} \left[ \log P_{\pi_{\text{ref}}}(x) - \log P_{\pi_{\text{proxy}}}(x) \right]$$
    3. Update domain weights $\alpha_d$ dynamically using exponential gradient updates:
       $$\alpha_d^{(t+1)} \propto \alpha_d^{(t)} \exp \left( \eta \cdot \mathcal{L}_d(\theta) \right)$$
- **FLOPs Scaling Formula:**
  - Runs proxy model iterations. Total compute is cheap because proxy size is small (e.g. 100M-300M parameters) compared to the target model (e.g. 7B+):
    $$\text{FLOPs} \approx R \times (6 \times N_{\text{proxy}} \times D_{\text{slice}})$$
    where $R$ is the number of optimization rounds and $D_{\text{slice}}$ is the calibration slice volume.
- **Precision Profile:**
  - **Proxy & Reference Models:** BF16.
  - **Domain weights calculations:** FP32.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x RTX 4090 (24GB VRAM) or 1x A100 (80GB).
- **Provider:** RunPod.
- **Execution Duration Limit:** < 8 hours.
- **Target Token/Batch Scale:** 10 proxy training runs of 100M model on 1B tokens.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** 8x H100 SXM5 node.
- **Interconnect Requirements:** NVLink (intra-node).
- **Persistent Storage Topography:** High speed local scratch NVMe storage to enable rapid switching between domain data pipelines.

## 3. Data Topography
- **Token Window Length:** 4,096 tokens.
- **Preprocessing Requirements:** Data inputs split into distinct domain folders (e.g., `/code`, `/web`, `/medical`, `/math`) for structured sampling.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Conceptual script updating domain mixing weights based on loss differentials.

import torch
import math

def update_domain_weights(current_weights, domain_losses, eta=0.1):
    # Aligns domain weights using exponential gradient ascent on relative losses
    # current_weights: dict of {domain: weight}
    # domain_losses: dict of {domain: excess_loss} (relative loss of proxy vs ref)
    
    new_weights = {}
    sum_weights = 0.0
    
    for domain, w in current_weights.items():
        loss = domain_losses.get(domain, 0.0)
        # Exp update rule
        new_w = w * math.exp(eta * loss)
        new_weights[domain] = new_w
        sum_weights += new_w
        
    # Renormalize weights
    for domain in new_weights:
        new_weights[domain] /= sum_weights
        
    return new_weights

if __name__ == "__main__":
    baseline = {"code": 0.25, "web": 0.40, "math": 0.15, "books": 0.20}
    # Excess losses (Proxy loss - Reference loss)
    # Higher excess loss indicates the model is struggling on this domain, so increase weight
    losses = {"code": 0.45, "web": -0.10, "math": 0.80, "books": 0.05}
    
    updated = update_domain_weights(baseline, losses, eta=0.5)
    print("Baseline:", baseline)
    print("Updated Mixture:", updated)
```

## 5. Failure Modes & Recovery
- **Domain Starvation / Degenerate Collapse:**
  - *Indicator:* The optimizer assigns a weight near zero to a domain (e.g. medical texts), causing the model to completely forget capabilities in that domain.
  - *Mitigation:* Enforce a minimum domain weight threshold (e.g. $\alpha_d \ge 0.02$) to guarantee that all domains remain present in the training mixture.
- **Noisy Gradient Steps:**
  - *Indicator:* Domain weights fluctuate wildly between calibration rounds without converging.
  - *Mitigation:* Reduce the learning rate parameter $\eta$ or increase the calibration token slice size $D_{\text{slice}}$ to get more stable, averaged loss metrics.
