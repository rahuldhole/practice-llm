import torch
from llm.model import TinyLLM
from llm.generate import load_tokenizer, generate_text

def run_inference():
    # Parameters must match training
    d_model = 16
    
    # Load tokenizer
    tokenizer = load_tokenizer("dist/vocab.json")
    vocab_size = tokenizer.vocab_size
    
    # Initialize model and load weights
    model = TinyLLM(vocab_size, d_model)
    model.load_state_dict(torch.load("dist/model.pth"))
    
    # Test generation
    prompt = "Hello"
    print(f"Prompt: '{prompt}'")
    
    generated = generate_text(model, tokenizer, prompt, max_new_tokens=4)
    print(f"Generated output: '{generated}'")

if __name__ == "__main__":
    run_inference()
