# 🧬 Tutorial 04: Custom Autograd & Loss Functions

**TLDR:** Extending PyTorch Autograd with custom forward/backward passes and writing custom loss modules.

PyTorch tracks all math operations using its **Autograd** engine to calculate gradients automatically. Sometimes, you need to define operations with custom mathematical gradients or modify gradients during backpropagation.

---

## 🪞 The Visual Metaphor: Gradient Reversal
Think of **Gradient Reversal** like a mirror that lets light pass through normally (forward pass) but reflects it back in the opposite direction with a different tint (backward pass). This is useful in domain adaptation tasks where we want features to align by *confusing* a domain discriminator.

```mermaid
flowchart LR
    subgraph Forward Pass (Identity)
        X[Input x] -->|Passes through| Y["Output y = x"]
    end
    
    subgraph Backward Pass (Reversal)
        dY[grad_output] -->|Multiplied by -alpha| dX["grad_input = -alpha * grad_output"]
    end
```

---

## 📊 Loss Function Comparison

| Feature / Detail | Standard BCE Loss | Binary Focal Loss |
|---|---|---|
| **Goal** | Measures overall error between predictions and target labels. | Addresses extreme class imbalances by focusing on hard examples. |
| **How it treats Easy Samples** | Still accumulates small errors from millions of easy background samples. | 📉 Downweights easy samples to zero using a factor: $(1 - p_t)^\gamma$. |
| **Best Used For** | Balanced target classes. | Severely imbalanced targets (e.g., detecting fraud, anomalies, or tiny objects). |

---

<details>
<summary>💡 Read about custom Autograd Functions and Focal Loss Math</summary>

### 1. Custom Autograd Functions (`torch.autograd.Function`)
In a custom autograd function, you must write both the forward step and the backward step manually.
- `forward(ctx, x, alpha)`: Calculates the output. We save `alpha` in `ctx` (context object).
- `backward(ctx, grad_output)`: Reads `alpha` from `ctx` and returns `-alpha * grad_output`. Because we had two inputs (`x` and `alpha`), we must return two outputs. Since `alpha` is a constant scalar, we return `None` for its gradient.

*Code reference*: [autograd_grl.py](../src/autograd_grl.py)

### 2. Custom Loss Modules (`nn.Module`)
A loss module is a standard `nn.Module`.
- In Binary Focal Loss, we get the probabilities $p = \text{sigmoid}(logits)$.
- We find the probability of the true label: $p_t = p \cdot y + (1 - p) \cdot (1 - y)$.
- We scale down the loss for high-confidence predictions: $Loss = - (1 - p_t)^\gamma \cdot \log(p_t)$.

*Code reference*: [loss_focal.py](../src/loss_focal.py)

</details>

---

## 💡 Practical Challenge
Run the code with `task pytorch-patterns:run -- src/autograd_grl.py`. Implement a custom autograd function `Square` that computes $y = x^2$ in the forward pass, and manually computes the derivative $dy/dx = 2x$ in the backward pass. Verify that your custom backward pass yields the same gradients as PyTorch's default autograd.
