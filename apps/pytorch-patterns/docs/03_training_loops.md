# 🧬 Tutorial 03: Robust Training Loops & Optimization

**TLDR:** Designing a safe, structured training loop with mode toggling, gradient clipping, and scheduler integrations.

A training loop in PyTorch requires coordinating several moving parts. A poorly configured loop can lead to silent bugs (e.g. updating validation stats, forgetting to zero gradients, or experiencing training divergence due to gradient explosions).

This guide walks through the components of a robust, production-grade training cycle.

---

## 1. Core Training Cycle Elements

### A. Train/Eval Mode Toggling
- `model.train()` puts the model in training mode. This enables layers like Dropout (which randomly drops connections) and Batch Normalization (which updates running statistics).
- `model.eval()` freezes these layers for inference so behavior is deterministic.

### B. Gradient Zeroing
Before running a backward pass, you must call `optimizer.zero_grad()`. In PyTorch, gradients accumulate (add up) on every `.backward()` call by default. Accumulating gradients is useful for virtual batch sizes, but if you do not clear them, subsequent training steps will mix gradients across steps, leading to incorrect updates.

### C. Gradient Norm Clipping
Exploding gradients (massive gradients) can destabilize training. Using `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` computes the total L2 norm of the model's gradients and scales them down if they exceed `max_norm`.

### D. Optimizer & Scheduler Steps
The optimizer updates weights using `optimizer.step()`. Schedulers (like `StepLR` or `CosineAnnealingLR`) decay learning rates over time to fine-tune final convergence. Remember to call `scheduler.step()` at the correct interval (e.g., at the end of each epoch).

### E. Inference under `torch.no_grad()`
During evaluation, wrapping computations inside `with torch.no_grad():` disables autograd tracking. This eliminates the memory overhead of storing intermediate activations, making inference significantly faster and memory-efficient.

*Code reference*: [Trainer class in training_loop.py](../src/training_loop.py#L42-L113)

---

## 💡 Practical Challenge
Run the training loop script using `task pytorch-patterns:run -- src/training_loop.py`. Try changing the optimizer from `Adam` to `SGD` with momentum, and change the learning rate scheduler to a cosine annealing scheduler (`torch.optim.lr_scheduler.CosineAnnealingLR`). Observe how the learning rate decays differently and compare the final validation accuracy.
