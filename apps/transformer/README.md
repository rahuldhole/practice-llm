# Educational Transformer from Scratch

This project is a step-by-step educational guide to building a Transformer model from scratch using PyTorch. Before diving into full Large Language Models (LLMs), it's crucial to understand the fundamental building blocks of the Transformer architecture (Attention Is All You Need, 2017).

## How to use this project

The codebase is split into digestible, bite-sized modules. Read them in order:

1. **`modules/embeddings_and_positional.py`**: Learn how tokens are turned into continuous vectors, and how position information is injected.
2. **`modules/attention.py`**: Understand Scaled Dot-Product Attention and Multi-Head Attention.
3. **`modules/feed_forward.py`**: See the simple Position-wise Feed-Forward Network.
4. **`modules/encoder_decoder.py`**: See how attention and feed-forward layers are assembled into Encoder and Decoder blocks.
5. **`modules/transformer.py`**: The complete Transformer model combining the Encoder and Decoder.

Finally, check out **`train_example.py`** to see how the model is trained on a simple sequence reversal task.

## Running the code

This project uses `Taskfile.yaml` to manage running the code in a virtual environment.

First, install the project and its dependencies into a virtual environment:
```bash
task install
```

You can run individual modules to see small demonstrations of shapes and outputs:
```bash
task run -- modules/embeddings_and_positional.py
```

Or run the full training loop:
```bash
task train
```
