# Blueprint: NPU-Native Kernel Compilations

## 1. Core Physics & Mechanics
- **Mathematical Objective:**
  - Optimization of the computational graph to maximize operator fusion and minimize data movement between Unified Memory and Neural Processing Units (NPUs):
    $$\min_{\text{graph}} \text{Latency}(\text{Graph}) \propto \sum_{i \in \text{layers}} (\text{Compute}_i + \text{MemoryIO}_i)$$
  - Graph compilation merges consecutive memory-bound operations (e.g. LayerNorm, SwiGLU activations, Scale-Dot-Product attention scaling) into unified hardware execution kernels to prevent thread scheduling overhead and cache misses on edge chips.
- **FLOPs Scaling Formula:**
  - Hardware execution optimization does not change the theoretical FLOP count ($6 \times N \times D$). Instead, it optimizes hardware execution efficiency:
    $$\text{Effective TFLOPS} = \frac{\text{Theoretical FLOPs}}{\text{Real-world Execution Latency}}$$
- **Precision Profile:**
  - **Apple Silicon ANE/GPU:** FP16/INT8.
  - **Intel Lunar Lake / Snapdragon X NPUs:** INT8/INT4 (requiring strict quantization scales mapping).

## 2. Infrastructure Specs (Scale Matrix)
### SandBox (Under $100)
- **Hardware:** 1x Apple Silicon M-series Mac or 1x Snapdragon X Elite Laptop.
- **Provider:** Local edge hardware.
- **Execution Duration Limit:** 1–2 hours.
- **Target Token/Batch Scale:** Validation on a 1,000-prompt performance test set.

### Production Scale (Unlimited Budget / Venture-Scale)
- **Cluster Size:** Single target device deployments (phones, laptops). The compilation and validation are performed locally on host developer rigs before distribution.
- **Interconnect Requirements:** Not applicable (local PCIe/unified memory interfaces).
- **Persistent Storage Topography:** High-speed local SSD.

## 3. Data Topography
- **Token Window Length:** 8,192 (8K) tokens.
- **Preprocessing Requirements:** Export model weights to intermediate formats (ONNX, CoreML, or MLX `.npz`) prior to graph lowering.
- **Tokenizer Handling:** Tokenizer stays native and runs entirely on the host CPU.

## 4. Run Execution Script
```python
#!/usr/bin/env python3
# Actionable Python script utilizing ONNX Runtime / MLX graph compilation routines.

import mlx.core as mx
import mlx.nn as nn
import time

class FusedAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.dims = dims
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(self, x):
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        
        # Scale-Dot-Product-Attention math
        scale = 1.0 / (self.dims ** 0.5)
        scores = mx.matmul(q, k.T) * scale
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)
        return self.out_proj(out)

# Create model
model = FusedAttention(dims=512, num_heads=8)

# 1. Compile execution graph to combine ops on GPU/NPU
compiled_fn = mx.compile(model)

# Mock input data
x = mx.random.normal((1, 512, 512))

# Dry run to compile the graph (JIT compilation overhead)
start = time.perf_counter()
res = compiled_fn(x)
mx.eval(res)
print(f"JIT Compilation & First Run Time: {(time.perf_counter() - start) * 1000:.2f} ms")

# Hot execution loops showing compiled performance gains
runs = 100
start = time.perf_counter()
for _ in range(runs):
    res = compiled_fn(x)
    mx.eval(res)
avg_time = (time.perf_counter() - start) / runs * 1000
print(f"Compiled Average Run Time: {avg_time:.2f} ms")
```

## 5. Failure Modes & Recovery
- **Kernel Compilation Divergence (NaNs or Numerical Outliers):**
  - *Indicator:* Output tensor values suddenly collapse to NaN during fused execution but remain healthy during standard interpretation mode.
  - *Mitigation:* Isolate the divergent sub-graph using execution profiling. Disable fusion compiler flags for the target layer (e.g. softmax or rotary embedding) and fall back to standard uncompiled MLX/PyTorch operations.
- **Memory Allocation Overflow (Out of Unified Memory):**
  - *Indicator:* The operating system kills the process during compilation due to out-of-memory errors.
  - *Mitigation:* Reduce compiler optimization levels. Compile the graph layer-by-layer rather than compiling the entire model trunk at once.
- **Rollback Instructions:**
  - Clear local compiler cash directories (`~/.cache/mlx` or ONNX runtime model caches), revert to uncompiled model code, and run diagnostics checks.
