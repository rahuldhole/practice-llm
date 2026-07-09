import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basics.attention import MultiHeadAttention
from basics.tokenizer import CharTokenizer

class FeedForward(nn.Module):
    """
    A simple linear projection followed by a non-linearity (GELU) and another projection.
    This acts as the token-wise position-wise Feed Forward Network (FFN) in a Transformer.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    A single Transformer Block.
    Combines Causal Multi-Head Attention and a Feed Forward network,
    using Layer Normalization and Residual Connections (Pre-LN architecture).
    """
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # sa_out is (out, weights_list); we only pass output up the residual stream
        sa_out, _ = self.sa(self.ln1(x))
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    """
    A minimal, Decoder-only (GPT-style) Transformer language model.
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        
        # Token and position embedding tables
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size) for _ in range(n_layer)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Logits prediction head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. Look up token embeddings: (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)
        
        # 2. Look up learned position embeddings: (T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        
        # 3. Sum embeddings
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # 4. Pass through blocks
        x = self.blocks(x)    # (B, T, n_embd)
        x = self.ln_f(x)      # (B, T, n_embd)
        
        # 5. Project to logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Reshape for cross entropy loss
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the block size (context window)
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step logits
            logits = logits[:, -1, :] # (B, vocab_size)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx


if __name__ == "__main__":
    print("--- Tiny GPT Decoder-Only Transformer Demo ---")
    torch.manual_seed(42)
    
    # 1. Dataset
    corpus = "hello world! hello from scratch transformer! let's build gpt!"
    tokenizer = CharTokenizer(corpus)
    print(f"Dataset Vocab Size: {tokenizer.vocab_size}")
    
    # We will generate training batches
    block_size = 8
    encoded_corpus = tokenizer.encode(corpus)
    
    inputs = []
    targets = []
    for i in range(len(encoded_corpus) - block_size):
        inputs.append(encoded_corpus[i : i + block_size])
        targets.append(encoded_corpus[i + 1 : i + block_size + 1])
        
    X = torch.tensor(inputs)  # (N, block_size)
    Y = torch.tensor(targets) # (N, block_size)
    
    # 2. Model configuration
    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=32,
        n_head=2,
        n_layer=2,
        block_size=block_size
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    
    # 3. Training Loop
    print("\nTraining Tiny GPT model on text...")
    for step in range(120):
        # Forward pass
        logits, loss = model(X, Y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 20 == 0:
            print(f"Step {step+1:03d} | Loss: {loss.item():.4f}")
            
    # 4. Generate from prompt
    prompt = "hello "
    prompt_tensor = torch.tensor([tokenizer.encode(prompt)])
    generated_tensor = model.generate(prompt_tensor, max_new_tokens=25)
    result = tokenizer.decode(generated_tensor[0].tolist())
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated output: '{result}'")
