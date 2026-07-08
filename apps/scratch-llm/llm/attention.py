import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from llm.embedding import apply_rotary_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.size()
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention: (batch_size, seq_len, n_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE
        Q, K = apply_rotary_emb(Q, K, freqs_cis)
        
        # Transpose to (batch_size, n_heads, seq_len, head_dim) for attention computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of values
        context = torch.matmul(attn_weights, V) # (batch_size, n_heads, seq_len, head_dim)
        
        # Reshape back to (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.o_proj(context)
