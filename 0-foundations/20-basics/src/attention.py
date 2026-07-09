import torch
import torch.nn as nn
import math

class CausalSelfAttentionHead(nn.Module):
    """
    A single head of causal self-attention.
    Processes inputs by projecting to Query, Key, and Value vectors,
    computing scaled dot-product attention scores, applying a causal mask,
    and outputting the context-rich vectors.
    """
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Lower-triangular matrix for causal masking (autoregressive generation)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x: input tensor of shape (batch, seq_len, n_embd)
        B, T, C = x.shape
        
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)
        
        # 1. Compute raw attention scores ("affinities"): q @ k^T / sqrt(d_k)
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.shape[-1]))
        
        # 2. Apply causal mask (prevent tokens from looking into the future)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 3. Softmax activation to compute attention weights
        weights = torch.softmax(scores, dim=-1) # (B, T, T)
        
        # 4. Perform weighted sum of values: weights @ v
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = weights @ v
        
        return out, weights

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Attention.
    Runs multiple CausalSelfAttentionHeads in parallel, concatenates their outputs,
    and applies a linear projection back to the embedding dimension.
    """
    def __init__(self, n_embd, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(n_embd, head_size, block_size) 
            for _ in range(num_heads)
        ])
        # Projection layer to mix head outputs
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        # x: (B, T, n_embd)
        # Concatenate outputs of all heads along the last dimension
        head_outputs = []
        attn_weights = []
        for head in self.heads:
            out, weights = head(x)
            head_outputs.append(out)
            attn_weights.append(weights)
            
        out = torch.cat(head_outputs, dim=-1) # (B, T, num_heads * head_size)
        out = self.proj(out)                  # (B, T, n_embd)
        
        return out, attn_weights


if __name__ == "__main__":
    print("--- Self-Attention Demo ---")
    torch.manual_seed(42)
    
    # Define sequence dimensions: Batch=1, SeqLen=3 (e.g. "I love coding"), EmbeddingDim=4
    B, T, C = 1, 3, 4
    x = torch.randn(B, T, C)
    
    # 1. Single Causal Attention Head
    head_size = 2
    single_head = CausalSelfAttentionHead(n_embd=C, head_size=head_size, block_size=8)
    out, weights = single_head(x)
    print(f"Input X shape: {x.shape}")
    print(f"Single Head output shape: {out.shape} (expected: ({B}, {T}, {head_size}))")
    print(f"Attention weights matrix:\n{weights[0]}")
    print("Note that row 0 has [1.0, 0.0, 0.0] - the first word cannot look at the second or third word!")
    
    # 2. Multi-Head Attention
    num_heads = 2
    mha = MultiHeadAttention(n_embd=C, num_heads=num_heads, head_size=head_size, block_size=8)
    mha_out, _ = mha(x)
    print(f"\nMulti-Head Attention output shape: {mha_out.shape} (expected: ({B}, {T}, {C}))")
