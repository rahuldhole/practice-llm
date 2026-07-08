import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Linear layers for Query, Key, Value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Compute Queries, Keys, and Values
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, d_model)
        V = self.v_proj(x)  # (batch_size, seq_len, d_model)
        
        # Compute scaled dot-product attention scores
        # Q @ K^T
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model) # (batch_size, seq_len, seq_len)
        
        # Create a causal mask (lower triangular matrix)
        # to prevent looking ahead into the future tokens
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0)
        
        # Apply the mask (set upper triangle to -inf)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1) # (batch_size, seq_len, seq_len)
        
        # Multiply weights by Values
        context = torch.bmm(attn_weights, V) # (batch_size, seq_len, d_model)
        
        return context
