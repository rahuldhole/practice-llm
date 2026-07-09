# 🎓 PyTorch Patterns: Advanced Deep Learning Workflows

**TLDR:** High-level overview and getting started guide for the project.

Welcome to the **PyTorch Patterns** learning project! This directory is designed to help you master advanced PyTorch workflows and patterns that are critical when writing custom neural network layers, custom training setups, and debugging large models.

Before building large-scale deep learning models (like Transformers or custom LLMs), mastering these PyTorch API patterns will make writing modular, clean, and bug-free code much easier.

## 🚀 The Implementation Ladder

Each file in this project covers a separate PyTorch concept in depth, complete with detailed comments and a standalone execution demo at the bottom.

1. **Custom Datasets & DataLoaders (`src/custom_dataset.py`)**: Custom indexing and dynamic padding. Learn how to tokenize raw sequences and write a custom collate function (`collate_fn`) to pad variable-length sequences dynamically per batch.
2. **Custom Modules & Initialization (`src/custom_module.py`)**: Subclassing `nn.Module`. Learn parameter registration (`nn.Parameter`), non-trainable state buffers (`register_buffer`), and custom weight initialization methods (Xavier/Kaiming).
3. **Robust Training Loops (`src/training_loop.py`)**: Standard training cycles. Learn how to toggle train/eval states correctly, run backpropagation, perform gradient clipping, and decay learning rates with schedulers.
4. **Custom Autograd & Loss (`src/custom_autograd.py`)**: Extending PyTorch math. Subclass `torch.autograd.Function` to write custom forward/backward passes (Gradient Reversal Layer) and write a custom loss function (Binary Focal Loss).
5. **PyTorch Hooks & Debugging (`src/hooks_debug.py`)**: Non-intrusive model inspection. Use forward and backward hooks to dynamically monitor activation statistics and gradient norms during training.

---

## 🛠️ Getting Started

### 1. Installation
To install this package in editable mode and configure the shared virtual environment, run:
```bash
task install
```

### 2. Run the Unit Tests
A comprehensive test suite verifies each code concept implementation. Run:
```bash
task test
```

### 3. Run Individual Demos
Each file in `src/` contains a self-contained demonstration at the bottom (which runs when the file is executed directly). You can run them using:
```bash
task run -- src/custom_dataset.py
task run -- src/custom_module.py
task run -- src/training_loop.py
task run -- src/custom_autograd.py
task run -- src/hooks_debug.py
```

## 📖 The Tutorial Roadmap
For detailed, step-by-step guides on each of these concepts, explore the local tutorials in the [docs/](docs/00_overview.md) directory.
