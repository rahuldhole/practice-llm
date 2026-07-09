# Blueprint: Prompt-Level Routing ("Static Gating")

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Standard autoregressive Cross-Entropy Loss with static expert locking:
    $$\min_{\theta_{\text{trunk}}, \theta_{R}} \mathcal{L}_{\text{CE}}(Y, f(X; \theta_{\text{trunk}}, E_{\text{selected}}))$$
    where the set of active experts $E_{\text{selected}} = \operatorname{Top-K}\left(R\left(\text{Embedding}(X_{\text{prompt}})\right)\right)$ is computed once at the start of the sequence (generation step 0) and pinned in memory, preventing per-token weight-swapping over subsequent generation ticks.
  - **Expert Locking:** For a prompt input sequence $P$, the routing block $R$ scores all $N$ experts. The top $K$ experts are loaded into high-speed unified RAM, while the remaining $N-K$ experts remain in local flash storage (NAND).
- **FLOPs Scaling Formula:**
  - Since expert selection occurs only once during prompt evaluation, there is no per-token routing latency or token-swapping cost. Compute per token is identical to a standard dense execution using active experts:
    $$\text{FLOPs/token} \approx 6 \times N_{\text{active}} \times D$$
- **Precision Profile:**
  - **Static Gating Router:** FP32.
  - **Base Transformer & Locked Expert Weights:** INT4 or INT8 (to minimize RAM footprint on consumer devices).
  - **Unified Activations:** FP16/BF16.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x Apple MacBook Pro (M-Series, minimum 16GB Unified RAM).
- **Provider:** Local edge machine.
- **Execution Duration Limit:** 2–4 hours calibration.
- **Target Token/Batch Scale:** 100M–500M tokens domain prompt dataset.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** Deployable to client devices natively (e.g., iPhone 15 Pro+, Apple M-series iPads/Macs). Training/fine-tuning requires a single GPU node (e.g., 1x A100/H100) for routing optimization.
- **Interconnect Requirements:** High-bandwidth on-device unified memory (e.g., LPDDR5 at >150 GB/s).
- **Persistent Storage Topography:** Local NAND flash storage (from which unselected expert weights are excluded).

## 3. Data Topography
- **Token Window Length:** 8,192 (8K) tokens.
- **Preprocessing Requirements:** Group training datasets by topic or prompt categories to teach the router distinct domain separations.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Actionable Python script using MLX implementing prompt-level static expert selection.

import mlx.core as mx
import mlx.nn as nn

class PromptRouter(nn.Module):
    def __init__(self, embedding_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(embedding_dim, num_experts)

    def __call__(self, prompt_embedding):
        # prompt_embedding shape: [batch, embedding_dim]
        scores = self.router(prompt_embedding)
        # Average over sequence dimension if input is 3D
        if len(scores.shape) == 3:
            scores = mx.mean(scores, axis=1)
        
        # Select static expert indices once
        expert_weights = mx.softmax(scores, axis=-1)
        selected_experts = mx.argpartition(expert_weights, -self.top_k, axis=-1)[:, -self.top_k:]
        return selected_experts

class EdgeStaticMoE(nn.Module):
    def __init__(self, num_experts: int, dims: int):
        super().__init__()
        self.num_experts = num_experts
        self.dims = dims
        self.router = PromptRouter(dims, num_experts, top_k=2)
        
        # Initialize experts
        self.experts = [nn.Linear(dims, dims) for _ in range(num_experts)]
        self.shared_expert = nn.Linear(dims, dims) # fallback expert

    def forward_generate(self, x, selected_expert_indices):
        # x shape: [batch, seq_len, dims]
        # selected_expert_indices shape: [batch, top_k]
        
        # Standard shared trunk
        h = self.shared_expert(x)
        
        # Route tokens ONLY through the pre-selected pinned experts
        batch_size = x.shape[0]
        out = mx.zeros_like(h)
        for b in range(batch_size):
            indices = selected_expert_indices[b].tolist()
            # Execute only the locked experts
            for idx in indices:
                out[b] += self.experts[idx](x[b])
        
        return h + out

# Simulation
moe = EdgeStaticMoE(num_experts=8, dims=128)
prompt_tokens = mx.random.normal((1, 10, 128))
gen_token = mx.random.normal((1, 1, 128))

# 1. Compute static expert routing mask once at step 0
selected_indices = moe.router(mx.mean(prompt_tokens, axis=1))
print("Pinned Expert Indices for Generation:", selected_indices.tolist())

# 2. Run generation steps utilizing ONLY the locked expert weights in memory
out = moe.forward_generate(gen_token, selected_indices)
mx.eval(out)
print("Generation Step Output Shape:", out.shape)
```

## 5. Failure Modes & Recovery
- **Domain Mismatch Route Failures (Wrong Expert Loaded):**
  - *Indicator:* Output quality falls sharply (hallucinations, loss of subject context) when a user transitions topics mid-chat.
  - *Mitigation:* Pin a "generalist fallback" or "shared" expert permanently in RAM alongside the routed experts. If prompt routing confidence is below a threshold, fall back to loading the generalist experts.
- **Memory Thrashing (Prompt Boundary Overhead):**
  - *Indicator:* High latency spikes at the start of a chat session when swapping out selected experts for the new prompt.
  - *Mitigation:* Implement expert cache recycling. Keep the previously loaded experts in a ring-buffer cache; if the new prompt overlaps with currently loaded experts, reuse them to bypass NAND reads.
- **Rollback Instructions:**
  - If the static router outputs NaN or fails to select valid index arrays, bypass static selection and default to loading a pre-configured baseline dense model or the generalist shared expert group.
