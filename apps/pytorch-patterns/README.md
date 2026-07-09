# 🎓 PyTorch Patterns: Advanced Deep Learning Workflows

**TLDR:** High-level overview and getting started guide for the project.

Welcome to the **PyTorch Patterns** learning project! This directory is designed to help you master advanced PyTorch workflows and patterns that are critical when writing custom neural network layers, custom training setups, and debugging large models.

Before building large-scale deep learning models (like Transformers or custom LLMs), mastering these PyTorch API patterns will make writing modular, clean, and bug-free code much easier.

## 🚀 The Implementation Ladder

Each file in this project covers a separate PyTorch concept in depth, complete with simplified, kid-friendly comments.

1. **Custom Datasets (`src/dataset_text.py`)**: Learn how to write a simple character index dataset.
2. **Dynamic Padding (`src/collate_padding.py`)**: Learn to pad batch sequences dynamically to the longest sequence in the batch.
3. **Module Parameters (`src/module_params.py`)**: Learn the difference between learning weights (`nn.Parameter`) and tracking states (`register_buffer`).
4. **Weight Initializations (`src/module_init.py`)**: Learn how to initialize layer weights using Xavier, Kaiming, or constant distributions.
5. **MLP Models (`src/model_mlp.py`)**: Learn to stack layers to form a Multi-Layer Perceptron.
6. **Training Loops (`src/trainer_loop.py`)**: Learn how to build a training and evaluation runner loop step-by-step.
7. **Custom Autograd (`src/autograd_grl.py`)**: Subclass `torch.autograd.Function` to write custom forward/backward logic (Gradient Reversal Layer).
8. **Custom Loss (`src/loss_focal.py`)**: Write a custom Binary Focal Loss module targeting class imbalance.
9. **Hooks Inspection (`src/hooks_inspector.py`)**: Attach forward/backward hooks to monitor activations and gradients non-intrusively.

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
task run -- src/dataset_text.py
task run -- src/collate_padding.py
task run -- src/module_params.py
task run -- src/module_init.py
task run -- src/model_mlp.py
task run -- src/trainer_loop.py
task run -- src/autograd_grl.py
task run -- src/loss_focal.py
task run -- src/hooks_inspector.py
```

## 📖 The Tutorial Roadmap
For detailed, step-by-step guides on each of these concepts, explore the local tutorials in the [docs/](docs/00_overview.md) directory.
