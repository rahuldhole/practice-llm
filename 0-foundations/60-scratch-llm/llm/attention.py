import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from llm.embedding import apply_rotary_emb

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = d_model // n_heads
        
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.num_queries_per_kv = self.n_heads // self.n_kv_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis, kv_cache=None):
        batch_size, seq_len, _ = x.size()
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head/GQA:
        # Q: (batch_size, seq_len, n_heads, head_dim)
        # K, V: (batch_size, seq_len, n_kv_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply RoPE
        Q, K = apply_rotary_emb(Q, K, freqs_cis)
        
        # Update/retrieve from KV cache if provided
        if kv_cache is not None:
            if kv_cache.is_empty(self):
                kv_cache.update(self, K, V)
            else:
                cached_K, cached_V = kv_cache.get(self)
                K = torch.cat([cached_K, K], dim=1)
                V = torch.cat([cached_V, V], dim=1)
                kv_cache.update(self, K, V)
        
        # If GQA is active (different Q vs KV heads), duplicate KV heads
        if self.n_kv_heads != self.n_heads:
            K = K.repeat_interleave(self.num_queries_per_kv, dim=2)
            V = V.repeat_interleave(self.num_queries_per_kv, dim=2)
            
        # Transpose to (batch_size, n_heads, seq_len, head_dim) for attention
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute FlashAttention (scaled dot-product attention)
        # is_causal=True applies the lower-triangular causal mask. Only causal for sequence inputs.
        is_causal = seq_len > 1
        
        context = F.scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=None, 
            dropout_p=0.0, 
            is_causal=is_causal
        )
        
        # Reshape back to (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        return self.o_proj(context)
