import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # Reshape xq and xk to match the complex representation
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs_cis to match sequence length
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    
    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        x: Tensor, shape (batch_size, seq_len)
        """
        # Positional encodings (RoPE) are applied during attention, 
        # so this layer only does token embeddings now.
        return self.word_embeddings(x)
