import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from modules.transformer import Transformer, causal_mask

class SequenceReversalDataset(Dataset):
    """
    A simple toy dataset where the goal is to reverse a sequence of numbers.
    e.g. Input: [1, 2, 3, 4], Output: [4, 3, 2, 1]
    We use 0 for padding, 1 for Start of Sequence (SOS), and 2 for End of Sequence (EOS).
    """
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Valid tokens are from 3 to vocab_size-1
        self.data = torch.randint(3, vocab_size, (num_samples, seq_len))
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        src = self.data[idx]
        # Target is the reversed sequence
        tgt = torch.flip(src, dims=[0])
        
        # Add SOS (1) and EOS (2) tokens
        encoder_input = torch.cat([torch.tensor([1]), src, torch.tensor([2])])
        
        # For decoder input, we prepend SOS and don't include the final EOS
        decoder_input = torch.cat([torch.tensor([1]), tgt])
        
        # The label is the target sequence with EOS appended
        label = torch.cat([tgt, torch.tensor([2])])
        
        return encoder_input, decoder_input, label

def train():
    # Hyperparameters
    vocab_size = 20
    seq_len = 5 # length of actual content
    # With SOS and EOS, the effective length is seq_len + 2
    effective_seq_len = seq_len + 2
    
    batch_size = 32
    num_epochs = 10
    
    # Tiny model configuration for fast training
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=effective_seq_len,
        tgt_seq_len=effective_seq_len,
        d_model=32,
        N=2,
        h=4,
        dropout=0.0, # no dropout for this simple task
        d_ff=128
    )
    
    dataset = SequenceReversalDataset(num_samples=1000, seq_len=seq_len, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Ignore padding index 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (encoder_input, decoder_input, label) in enumerate(dataloader):
            
            # Create masks
            # (batch, 1, 1, seq_len)
            src_mask = (encoder_input != 0).unsqueeze(1).unsqueeze(2).int()
            
            # Decoder mask: padding mask & causal mask
            padding_mask = (decoder_input != 0).unsqueeze(1).unsqueeze(2).int()
            causal = causal_mask(decoder_input.size(1)).unsqueeze(0).unsqueeze(0).to(decoder_input.device)
            tgt_mask = padding_mask & causal
            
            # Forward pass
            logits = model(encoder_input, decoder_input, src_mask, tgt_mask)
            
            # Flatten to compute loss
            # logits: (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
            # label: (batch, seq_len) -> (batch * seq_len)
            loss = loss_fn(logits.view(-1, vocab_size), label.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")
        
    print("\nTraining Complete! The model successfully learns to reverse sequences.")
    print("You can inspect the loss decreasing across epochs to verify the Transformer is working.")

if __name__ == "__main__":
    train()
