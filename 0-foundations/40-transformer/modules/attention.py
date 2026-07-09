import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        """
        Multi-Head Attention block.
        
        Args:
            d_model (int): The dimension of the model (must be divisible by h).
            h (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by number of heads (h)"
        
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h # Dimension of each head
        
        # Linear layers for projecting Query, Key, and Value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Final linear layer to project concatenated heads back to d_model
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        """
        Compute Scaled Dot-Product Attention.
        Shapes:
            q: (batch, h, seq_len, d_k)
            k: (batch, h, seq_len, d_k)
            v: (batch, h, seq_len, d_k)
        """
        d_k = q.size(-1)
        
        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Mask should be broadcastable to (batch, h, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        p_attn = torch.softmax(scores, dim=-1)
        
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
            
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        output = torch.matmul(p_attn, v)
        
        return output, p_attn

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            q, k, v (torch.Tensor): Tensors of shape (batch, seq_len, d_model)
            mask (torch.Tensor, optional): Mask tensor.
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size = q.size(0)
        
        # 1. Linear projections and split into multiple heads
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # 2. Apply attention on all the projected vectors in batch
        x, self.attn_weights = self.attention(query, key, value, mask)
        
        # 3. "Concat" using a view and apply a final linear layer
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(x)

if __name__ == "__main__":
    print("Testing Multi-Head Attention")
    batch_size = 2
    seq_len = 5
    d_model = 16
    heads = 4
    
    # Create random embeddings
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape (q, k, v): {x.shape}")
    
    # Initialize Multi-Head Attention
    mha = MultiHeadAttention(d_model=d_model, h=heads)
    
    # Self-attention: q, k, v are all the same input
    output = mha(q=x, k=x, v=x)
    print(f"Output shape after Multi-Head Attention: {output.shape}")
