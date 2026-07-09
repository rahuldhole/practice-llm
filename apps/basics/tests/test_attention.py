import torch
import pytest
from basics.attention import CausalSelfAttentionHead, MultiHeadAttention

def test_single_head():
    torch.manual_seed(42)
    B, T, C = 2, 5, 8
    head_size = 4
    head = CausalSelfAttentionHead(n_embd=C, head_size=head_size, block_size=10)
    
    x = torch.randn(B, T, C)
    out, weights = head(x)
    
    # Assert correct dimensions
    assert out.shape == (B, T, head_size)
    assert weights.shape == (B, T, T)
    
    # Assert causal masking (upper triangle should be exactly 0)
    for b in range(B):
        for i in range(T):
            for j in range(i + 1, T):
                assert weights[b, i, j].item() == 0.0
                
    # Assert softmax normalization (each row sums to 1.0)
    for b in range(B):
        for i in range(T):
            assert abs(weights[b, i, :i+1].sum().item() - 1.0) < 1e-6

def test_multi_head_attention():
    torch.manual_seed(42)
    B, T, C = 2, 5, 8
    num_heads = 3
    head_size = 4
    mha = MultiHeadAttention(n_embd=C, num_heads=num_heads, head_size=head_size, block_size=10)
    
    x = torch.randn(B, T, C)
    out, weights_list = mha(x)
    
    # Output projection should project back to C (n_embd)
    assert out.shape == (B, T, C)
    assert len(weights_list) == num_heads
    assert weights_list[0].shape == (B, T, T)
