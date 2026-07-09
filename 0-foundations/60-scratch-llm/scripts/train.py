import argparse
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from llm.dataset import get_dataloader
from llm.model import ScratchLLM, RMSNorm

def get_optimizer_params(model, weight_decay):
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear,)
    blacklist_weight_modules = (RMSNorm, nn.Embedding)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                no_decay.add(fpn)
                
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {param_dict.keys() - union_params} were not categorized"
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # cosine decay to 10%
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 0.1 + 0.9 * cosine_decay
    return LambdaLR(optimizer, lr_lambda)

def main():
    parser = argparse.ArgumentParser(description="Train ScratchLLM")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu, cuda, mps)")
    args = parser.parse_args()

    # Hyperparameters
    d_model = 32
    n_heads = 4
    n_kv_heads = 2 # GQA
    n_layers = 2
    seq_length = 16
    batch_size = 16
    epochs = 50
    learning_rate = 2e-3
    weight_decay = 0.1
    max_grad_norm = 1.0
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
    model = ScratchLLM(actual_vocab_size, d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, n_layers=n_layers)
    model.to(device)

    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with weight decay groupings
    optim_groups = get_optimizer_params(model, weight_decay)
    optimizer = optim.AdamW(optim_groups, lr=learning_rate)
    
    # LR Scheduler (warmup + cosine)
    total_steps = epochs * len(dataloader)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    # AMP Config
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    logits = logits.view(-1, actual_vocab_size)
                    y = y.view(-1)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                logits = logits.view(-1, actual_vocab_size)
                y = y.view(-1)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
            scheduler.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

    print("Training complete!")
    
    os.makedirs("dist", exist_ok=True)
    
    # Save the model and BPE tokenizer
    torch.save(model.state_dict(), "dist/model.pth")
    tokenizer.save("dist/tokenizer.json")
    print("Model saved to dist/model.pth")
    print("Tokenizer saved to dist/tokenizer.json")

if __name__ == "__main__":
    main()
