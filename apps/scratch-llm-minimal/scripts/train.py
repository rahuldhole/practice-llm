import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from llm.dataset import get_dataloader
from llm.model import ScratchLLM

def main():
    parser = argparse.ArgumentParser(description="Train ScratchLLM Minimal")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu, cuda, mps)")
    args = parser.parse_args()

    # Hyperparameters
    d_model = 32
    n_heads = 4
    n_layers = 2
    seq_length = 16
    batch_size = 16
    epochs = 50
    learning_rate = 2e-3
    vocab_size = 2000

    # Device configuration
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available. Falling back to CPU.")
        device = torch.device("cpu")
    
    print(f"Training on device: {device}")

    # Load dataloader and tokenizer
    dataloader, tokenizer = get_dataloader("data/dataset.txt", seq_length=seq_length, batch_size=batch_size, vocab_size=vocab_size)
    actual_vocab_size = tokenizer.vocab_size
    print(f"Actual vocab size: {actual_vocab_size}")

    # Initialize model
    model = ScratchLLM(actual_vocab_size, d_model, n_heads=n_heads, n_layers=n_layers)
    model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Simple Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # Simple Forward Pass
            logits = model(x)
            logits = logits.view(-1, actual_vocab_size)
            y = y.view(-1)
            
            # Loss and Backward Pass
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
                
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {learning_rate:.6f}")

    print("Training complete!")
    
    os.makedirs("dist", exist_ok=True)
    
    # Save the model and tokenizer
    torch.save(model.state_dict(), "dist/model.pth")
    tokenizer.save("dist/tokenizer.json")
    print("Model saved to dist/model.pth")
    print("Tokenizer saved to dist/tokenizer.json")

if __name__ == "__main__":
    main()
