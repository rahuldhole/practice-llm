import argparse
import torch
import os
from llm.model import ScratchLLM
from llm.generate import load_tokenizer, generate_text

def run_inference():
    parser = argparse.ArgumentParser(description="Run ScratchLLM Inference")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu, cuda, mps)")
    args = parser.parse_args()

    # Parameters must match training
    d_model = 32
    n_heads = 4
    n_layers = 2
    
    # Device configuration
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    
    print(f"Running inference on device: {device}")
    
    # Load tokenizer
    tokenizer = load_tokenizer("dist/tokenizer.json")
    vocab_size = tokenizer.vocab_size
    
    # Initialize model and load weights
    model = ScratchLLM(vocab_size, d_model, n_heads=n_heads, n_layers=n_layers)
    model.load_state_dict(torch.load("dist/model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Test generation
    prompt = "User: Hello! AI:"
    print(f"Prompt: '{prompt}'")
    
    generated = generate_text(model, tokenizer, prompt, max_new_tokens=20, temperature=0.7, top_k=10, device=device)
    print(f"Generated output: '{generated}'")

if __name__ == "__main__":
    run_inference()
