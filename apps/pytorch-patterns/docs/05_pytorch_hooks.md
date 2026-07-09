# 🧬 Tutorial 05: PyTorch Hooks & Debugging

**TLDR:** Attaching non-intrusive hooks to inspect layer activations and gradients at runtime.

When debugging neural networks, logging variables inside the `forward` function of a model can quickly clutter your codebase, especially when using pre-built networks (like torchvision models or Hugging Face Transformers).

PyTorch solves this with **Hooks**—callbacks that can be registered on any `nn.Module` to execute custom code during the forward or backward pass without altering the module's source code.

---

## 1. Types of Hooks

### A. Forward Hooks (`register_forward_hook`)
- Executes after a module's `forward` function finishes.
- Signature: `hook_fn(module, input, output)`
- Useful for extracting feature activations, logging representation statistics (mean/std), or caching values for visualization.

### B. Backward Hooks (`register_full_backward_hook`)
- Executes when gradients are computed during backpropagation.
- Signature: `hook_fn(module, grad_input, grad_output)`
- Useful for checking if gradients are vanishing (approaching 0) or exploding, and auditing layer updates.

*Code reference*: [ModelInspector hooks setup in hooks_debug.py](../src/hooks_debug.py#L32-L61)

---

## 2. Preventing Memory Leaks
When working with hooks, it is easy to accidentally cause memory leaks:
1. **Detaching Tensors**: Inside hooks, always call `.detach()` on inputs, outputs, or gradients if you intend to store them outside the pass (e.g. in a list). Storing active tensors keeps their entire computational graph in GPU memory, preventing it from being garbage collected.
2. **Removing Handles**: Registering a hook returns a hook handle. Once debugging is complete, call `handle.remove()` to clean up. If left registered, the hook runs on every step, degrading performance.

*Code reference*: [Hook cleanup in hooks_debug.py](../src/hooks_debug.py#L99-L106)

---

## 💡 Practical Challenge
Run the inspector demo with `task pytorch-patterns:run -- src/hooks_debug.py`. Try modifying `ModelInspector` to track the percentage of zero activations (dead neurons) in ReLU layers. This is done by checking what fraction of values in the `output` tensor of a layer are exactly 0.0.
