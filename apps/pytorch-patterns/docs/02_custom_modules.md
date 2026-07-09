# 🧬 Tutorial 02: Custom Modules, Buffers & Initialization

**TLDR:** Managing trainable parameters, state tracking buffers, and custom weight initializations in nn.Module.

Every layer in PyTorch inherits from `nn.Module`. While high-level layers (like `nn.Linear` or `nn.Conv2d`) manage their parameters automatically, advanced applications require writing custom layers where you manage parameters and states manually.

---

## 1. Parameters vs. Buffers
There are two main types of state variables in a PyTorch module:

### A. Trainable Parameters (`nn.Parameter`)
- Wraps a tensor and marks it as a learnable parameter.
- Automatically registered under the module's `.parameters()` generator.
- Gradients are calculated for it during `.backward()` (`requires_grad = True` by default).

### B. State Buffers (`register_buffer`)
- A tensor containing state that is *not* learnable (does not compute gradients) but is still a key part of the model.
- Example: running mean and variance in Batch Normalization, position embeddings (like sinusoidal position tables), or step counters.
- By registering it as a buffer, PyTorch:
  - Saves it in the module's `state_dict()` (for checkpoint loading/saving).
  - Automatically moves it when calling `.to(device)` (e.g. sending the model to CUDA).

*Code reference*: [CustomLinear parameter & buffer setup in custom_module.py](../src/custom_module.py#L22-L35)

---

## 2. Weights Initialization
Starting weights significantly impact a model's convergence. PyTorch provides initialization functions in the `torch.nn.init` module:
- **Xavier (Glorot) Initialization**: Keeps variance of activations and gradients uniform across layers. Best for symmetric activations (tanh, sigmoid).
- **Kaiming (He) Initialization**: Tailored for asymmetric non-linearities like ReLU or LeakyReLU.
- **Constant Initialization**: Fills tensors with a single scalar value (often used for biases).

*Code reference*: [initialize_parameters in custom_module.py](../src/custom_module.py#L37-L57)

---

## 💡 Practical Challenge
Run the module using `task pytorch-patterns:run -- src/custom_module.py`. Modify `CustomLinear` to track and log the maximum activation magnitude seen so far across all steps inside `forward`, and save it using a registered buffer named `max_activation`. Check if it appears in `state_dict()`.
