# 🐍 Python Concepts: Core Python for Deep Learning & Library Design

**TLDR:** High-level overview and getting started guide for the project.

Welcome to the **Python Concepts** learning project! This directory is designed to help you master advanced Python concepts that are foundational to deep learning framework implementations, PyTorch usage, and general library design.

Before diving deep into writing Transformer networks, neural network layers, or autograd engines, understanding how Python handles magic methods, custom iterators, decorators, context managers, and metaclass-based class registries will make reading and writing library code much more natural.

## 🚀 The Learning Ladder

Each file in this project covers a separate concept in depth, complete with comments and a standalone execution demo at the bottom.

1. **Dunder (Magic) Methods (`src/dunder_methods.py`)**: Customizing object behavior. Learn operator overloading (`+`, `*`), custom representations, list/indexing behavior, and callable objects (`__call__`) using a mathematical `Vector` class.
2. **Decorators & Closures (`src/decorators.py`)**: Dynamically extending function capabilities. Implement custom decorators for performance timing, caching/memoization, and parameterizable logging.
3. **Generators & Iterators (`src/generators.py`)**: Lazy evaluation and pipeline data streaming. Implement the Iterator protocol (`__iter__`/`__next__`) via a `BatchLoader` that streams mini-batches for ML models, and construct generator pipelines.
4. **Context Managers (`src/context_managers.py`)**: Automated setup/teardown and execution state control. Implement context managers for block execution timing, safe file logging, and setting global runtime configuration flags (similar to `torch.no_grad()`).
5. **OOP & Metaprogramming (`src/oop_meta.py`)**: Abstract interfaces, properties, and dynamic class creation. Implement Abstract Base Classes (ABCs), getters/setters, and a custom Metaclass class registry pattern to auto-register network layers.

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
task run -- src/dunder_methods.py
task run -- src/decorators.py
task run -- src/generators.py
task run -- src/context_managers.py
task run -- src/oop_meta.py
```

## 📖 The Tutorial Roadmap
For detailed, step-by-step guides on each of these concepts, explore the local tutorials in the [docs/](docs/00_overview.md) directory.
