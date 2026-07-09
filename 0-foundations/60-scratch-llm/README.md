# Scratch LLM

**TLDR:** High-level overview and getting started guide for the project.

This is a tiny, from-scratch implementation of a Transformer-based Large Language Model, written entirely in PyTorch.

## Features
- Custom word-level tokenizer
- Positional encodings
- Single-head self-attention
- Simple Multi-Layer Perceptron (FFN)
- Overfitting on a tiny dataset

## How to use

1. Set up your virtual environment and install dependencies using Task:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install taskipy  # Or install go-task
   task install
   ```

2. Train the model:
   ```bash
   task train
   ```

3. Run the API:
   ```bash
   task api
   ```

## Endpoints

- `POST /generate`: Pass `{"prompt": "Hello", "max_new_tokens": 5}` to get a prediction based on the trained model.
