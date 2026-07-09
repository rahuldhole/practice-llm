# Blueprint: Router-Decoupled Pre-Gating (Lookahead Scheduling)

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Decouple the expert routing decisions from the active layer compute pointer. The lookahead router $R_{\text{lookahead}}$ predicts expert routing indices $L$ layers in advance based on intermediate token states:
    $$\hat{e}_{l+d} = R_{\text{lookahead}}(h_l)$$
    where $h_l$ is the token activation at layer $l$, and $\hat{e}_{l+d}$ designates the predicted experts needed at layer $l+d$.
  - **Asynchronous Pre-fetching:** While the GPU is calculating attention and FFN operations for layers $l$ through $l+d-1$, an asynchronous CPU/OS background thread issues flash-to-RAM copy operations (DMA transfers) to stream the weights of expert $\hat{e}_{l+d}$ into unified RAM.
- **FLOPs Scaling Formula:**
  - Overlapping compute and I/O hides the latency of expert weight swaps:
    $$\text{Latency/token} = \max\left(\text{ComputeTime}(l \to l+d-1), \text{I/O\_TransferTime}(\hat{e}_{l+d})\right)$$
    This ensures that memory bandwidth bottlenecks do not stall execution loops.
- **Precision Profile:**
  - **Lookahead Router:** FP32.
  - **Expert Weights:** INT4 or INT2 GGUF (crucial for keeping transfer sizes smaller than unified memory bandwidth limits).
  - **Attention/Trunk:** FP16.

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x Apple MacBook (M-Series, minimum 16GB RAM) or local Snapdragon NPU system.
- **Provider:** Local edge hardware.
- **Execution Duration Limit:** 2–4 hours.
- **Target Token/Batch Scale:** 500M tokens calibration dataset.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** On-device consumer target configurations (Apple Silicon unified memory chips with high-performance solid-state drives).
- **Interconnect Requirements:** Direct Memory Access (DMA) interfaces linking NAND storage flash directly to unified GPU/NPU memory channels.
- **Persistent Storage Topography:** High-speed NVMe/UFS 4.0 flash storage.

## 3. Data Topography
- **Token Window Length:** 8,192 (8K) tokens.
- **Preprocessing Requirements:** Track expert activations over standard runs to build target lookahead prediction targets.
- **Tokenizer Handling:** Frozen base model tokenizer.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Actionable Python script using MLX showing concurrent pre-fetching of expert weights.

import mlx.core as mx
import mlx.nn as nn
import threading
import queue
import time

class DecoupledRouter(nn.Module):
    def __init__(self, dims: int, num_experts: int, lookahead_layers: int = 4):
        super().__init__()
        self.lookahead = lookahead_layers
        self.proj = nn.Linear(dims, num_experts)

    def __call__(self, h):
        scores = self.proj(h)
        return mx.argmax(scores, axis=-1)

# Mock expert class containing disk loading simulation
class EdgeExpert:
    def __init__(self, idx: int, dims: int):
        self.idx = idx
        self.dims = dims
        self.weight = None

    def load_from_disk(self):
        # Simulate local flash to RAM load latency (e.g. 10ms)
        time.sleep(0.01)
        self.weight = mx.random.normal((self.dims, self.dims))

    def unload(self):
        self.weight = None

# Pre-fetching queue manager
prefetch_queue = queue.Queue()
experts = [EdgeExpert(i, 256) for i in range(8)]

# Background worker thread simulating OS DMA transfer
def dma_worker():
    while True:
        expert_idx = prefetch_queue.get()
        if expert_idx is None:
            break
        if experts[expert_idx].weight is None:
            # Simulate background NAND read
            experts[expert_idx].load_from_disk()
        prefetch_queue.task_done()

# Start background pre-fetch worker
worker_thread = threading.Thread(target=dma_worker, daemon=True)
worker_thread.start()

# Decoupled Routing execution loop
router = DecoupledRouter(dims=256, num_experts=8, lookahead_layers=4)
token_state = mx.random.normal((1, 256))

print("Starting generation loop with lookahead routing...")

for layer in range(12):
    # 1. At layer L, predict expert for layer L + 4
    if layer + 4 < 12:
        future_expert_idx = router(token_state).item()
        # Trigger background load without stalling current thread
        prefetch_queue.put(future_expert_idx)
        print(f"[Layer {layer}] Triggered pre-fetch of Expert {future_expert_idx} for Layer {layer + 4}")

    # 2. Simulate attention compute for current layer L (e.g. 8ms)
    time.sleep(0.008)
    
    # 3. Check if layer L expert is loaded
    if layer >= 4:
        # Resolve target expert
        target_expert = experts[0] # mock index lookup
        if target_expert.weight is None:
            print(f"[Warning] Layer {layer} stalled! Expert weights not loaded in time.")
            target_expert.load_from_disk() # Block thread to load
        else:
            print(f"[Layer {layer}] Execution overlap SUCCESS. Expert is active in Unified RAM.")

# Cleanup worker
prefetch_queue.put(None)
worker_thread.join()
```

## 5. Failure Modes & Recovery
- **Pre-fetch Lag (Compute stalls I/O):**
  - *Indicator:* High frame drop rates or stuttering during local text generation due to the GPU waiting on flash read access.
  - *Mitigation:* Dynamically redirect calculations to generalist/shared experts already permanently pinned in memory, skipping the un-loaded expert forward pass entirely.
- **Unified RAM Bloat (Over-fetching):**
  - *Indicator:* The host system terminates the process due to out-of-memory errors caused by loading too many experts concurrently.
  - *Mitigation:* Implement a strict expert lease/release cache scheduler. Unload expert weights from unified RAM as soon as their target layer executions are completed.
- **Rollback Instructions:**
  - If lookahead predictions fail or cache misses exceed 20%, disable background threads and fall back to executing standard token-by-token dense projection, bypassing speculative expert swaps.
