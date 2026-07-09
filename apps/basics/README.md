# 🎓 Basics: Foundations of Deep Learning & Transformers

Welcome to the **Basics** learning project! This directory is designed to help you build the conceptual and mathematical foundation needed to understand nanoGPT easily.

Instead of jumping directly into complex LLMs, you will build the stack from the ground up, starting with raw Python, basic matrix operations, building an autograd engine, constructing layers and MLPs, writing tokenizers, understanding RNNs, implementing single-head and multi-head attention, and finally assembling a Decoder-Only Transformer (Tiny GPT).

## 🚀 The Implementation Ladder

Follow this order of implementation:

1. **Matrix Library (`src/matrix.py`)**: Matrix operations from scratch (addition, dot product, transpose, mean, variance).
2. **Autograd Engine (`src/autograd.py`)**: A mini-PyTorch backward-propagation engine for automatic differentiation (computational graph tracking).
3. **Neural Network Library (`src/nn.py`)**: Custom layers, neurons, and a Multilayer Perceptron (MLP) built from scratch using our autograd engine.
4. **Tokenizer (`src/tokenizer.py`)**: Character-level tokenization and Byte Pair Encoding (BPE).
5. **RNN Sequence Model (`src/sequence.py`)**: A character-level Recurrent Neural Network (RNN) language model.
6. **Attention (`src/attention.py`)**: Scaled dot-product attention, causal masking, and Multi-Head Attention using PyTorch.
7. **Tiny GPT (`src/gpt.py`)**: A mini Decoder-Only Transformer model built, trained, and evaluated on sample text.

---

## 🛠️ Getting Started

### 1. Installation
To install the package in editable mode and set up the shared virtual environment, run:
```bash
task install
```

### 2. Run the Unit Tests
We have provided a robust test suite to verify that your mathematical and structural implementations are correct. Run:
```bash
task test
```

### 3. Run Individual Demos
Each file in `src/` contains a self-contained demonstration at the bottom (run when the file is executed directly). You can run them using:
```bash
task run -- src/autograd.py
task run -- src/nn.py
task run -- src/gpt.py
```

## 📖 The Tutorial Roadmap
For a detailed guide on the learning ladder, start with the local [Course Overview](docs/00_overview.md) or read the shared roadmap [05-basics-roadmap.md](../../learning/tutorials-llm/05-basics-roadmap.md).
