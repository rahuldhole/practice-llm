import torch
import torch.nn as nn
from llm.embedding import EmbeddingLayer, precompute_freqs_cis
from llm.attention import MultiHeadAttention

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class ScratchLLMBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        hidden_dim = int(4 * d_model * 2 / 3)
        self.ffn = SwiGLU(d_model, hidden_dim)
        
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x, freqs_cis):
        # Pre-Norm architecture
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.ffn(self.ffn_norm(h))
        return out

class ScratchLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads=4, n_layers=2, max_seq_len=2048):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            ScratchLLMBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(d_model // n_heads, max_seq_len)

    def forward(self, x):
        """
        x: Input token IDs, shape (batch_size, seq_len)
        """
        _, seq_len = x.shape
        x = self.embedding(x)
        
        # Get frequencies for the current sequence length
        freqs_cis = self.freqs_cis[:seq_len].to(x.device)
        
        for layer in self.layers:
            x = layer(x, freqs_cis)
            
        x = self.norm(x)
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)
        
        return logits
