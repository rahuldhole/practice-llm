import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from llm.dataset import get_dataloader
from llm.model import ScratchLLM
import os

def train():
    # Hyperparameters
    d_model = 16
    n_heads = 4
    n_layers = 2
    seq_length = 4
    batch_size = 2
    epochs = 200
    learning_rate = 0.01
    weight_decay = 0.1
    max_grad_norm = 1.0

    # Get data
    dataloader, tokenizer = get_dataloader("data/dataset.txt", seq_length=seq_length, batch_size=batch_size)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Initialize model
    model = ScratchLLM(vocab_size, d_model, n_heads=n_heads, n_layers=n_layers)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # SOTA Practice: Use AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # SOTA Practice: Cosine Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            # Forward pass
            logits = model(x) # (batch_size, seq_len, vocab_size)
            
            # Reshape for CrossEntropyLoss
            # It expects inputs of shape (N, C) and targets of shape (N)
            logits = logits.view(-1, vocab_size)
            y = y.view(-1)
            
            # Compute loss
            loss = criterion(logits, y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # SOTA Practice: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    print("Training complete!")
    
    os.makedirs("dist", exist_ok=True)
    
    # Save the model and tokenizer
    torch.save(model.state_dict(), "dist/model.pth")
    # For simplicity in this tiny example, we just save the word2id
    import json
    with open("dist/vocab.json", "w") as f:
        json.dump(tokenizer.word2id, f)
    print("Model and vocabulary saved.")

if __name__ == "__main__":
    train()
