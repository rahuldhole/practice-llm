# 🧬 Tutorial 04: Custom Autograd & Loss Functions

**TLDR:** Extending PyTorch Autograd with custom forward/backward passes and writing custom loss modules.

PyTorch is powered by its **Autograd** engine, which builds a dynamic computational graph during forward execution and traces it backward to calculate gradients. Sometimes, you need to define operations with custom gradient calculations that aren't natively supported, or modify gradients during backpropagation (e.g. for gradient reversal).

---

## 1. Custom Autograd Functions
To create a custom math operation, subclass `torch.autograd.Function` and implement two static methods:
- `forward(ctx, inputs, ...)`: Computes the output tensor. Use `ctx.save_for_backward` or store constants on `ctx` to cache variables needed for backpropagation.
- `backward(ctx, grad_output)`: Receives the incoming gradients w.r.t the outputs, and computes the gradients w.r.t each input.

### Example: Gradient Reversal Layer (GRL)
In domain-adversarial networks, we want to extract features that are domain-invariant. We do this by reversing the gradients flowing into the feature extractor.
- **Forward pass**: $f(x) = x$ (Identity)
- **Backward pass**: $\frac{df}{dx} = -\alpha \cdot \text{grad\_output}$ (Reverses and scales gradients)

*Code reference*: [GradientReversalFunction in custom_autograd.py](../src/custom_autograd.py#L5-L42)

---

## 2. Custom Loss Functions
Loss functions are standard `nn.Module` subclasses. You override the `forward` method to compute a scalar tensor representing the model's error.

### Example: Binary Focal Loss
For highly imbalanced datasets, standard Binary Cross Entropy (BCE) loss can be overwhelmed by easy-to-classify background examples. Focal Loss addresses this by multiplying the BCE loss by a scale factor $(1 - p_t)^\gamma$:
- When the model is confident and correct ($p_t \approx 1$), $(1 - p_t)^\gamma \approx 0$, heavily discounting the loss.
- When the model is incorrect or uncertain ($p_t \approx 0$), $(1 - p_t)^\gamma \approx 1$, keeping the loss magnitude intact.

*Code reference*: [FocalLoss in custom_autograd.py](../src/custom_autograd.py#L52-L99)

---

## 💡 Practical Challenge
Run the code with `task pytorch-patterns:run -- src/custom_autograd.py`. Implement a custom autograd function `Square` that computes $y = x^2$ in the forward pass, and manually computes the derivative $dy/dx = 2x$ in the backward pass. Verify that your custom backward pass yields the same gradients as PyTorch's default autograd.
