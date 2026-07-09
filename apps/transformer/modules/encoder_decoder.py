import torch
import torch.nn as nn

# In a real project, you would import these from the previous files:
# from modules.02_attention import MultiHeadAttention
# from modules.03_feed_forward import FeedForwardBlock
# For this script to be standalone/executable easily, we rely on the same modules or assume they are available.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.attention import MultiHeadAttention
from modules.feed_forward import FeedForwardBlock

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        """
        Implements a residual connection followed by Layer Normalization.
        Note: The original paper puts LayerNorm *after* the addition (Post-LN),
        but modern architectures often put LayerNorm *before* the sub-layer (Pre-LN).
        We will use Pre-LN as it generally trains more stably.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
            sublayer (Callable): The sublayer function (e.g. self-attention or FFN).
        """
        # Pre-LN: Norm -> Sublayer -> Dropout -> Add
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        A single Encoder Block consisting of Self-Attention and a Feed-Forward network.
        """
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        
        # We need two residual connections (one for attention, one for FFN)
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Sublayer 1: Self-Attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Sublayer 2: Feed-Forward
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        """
        A single Decoder Block consisting of Masked Self-Attention, Cross-Attention, and a Feed-Forward network.
        """
        super().__init__()
        self.self_attention_block = MultiHeadAttention(d_model, h, dropout)
        self.cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        
        # We need three residual connections here
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # Sublayer 1: Masked Self-Attention
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        # Sublayer 2: Cross-Attention (Query from Decoder, Keys and Values from Encoder)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        
        # Sublayer 3: Feed-Forward
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


if __name__ == "__main__":
    print("Testing Encoder and Decoder Blocks")
    batch_size = 2
    seq_len = 5
    d_model = 16
    heads = 4
    d_ff = 64
    
    # Random inputs
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # Initialize Blocks
    encoder_block = EncoderBlock(d_model, heads, d_ff)
    decoder_block = DecoderBlock(d_model, heads, d_ff)
    
    # Encoder Forward
    # Note: src_mask=None means no masking for padding in this simple test
    enc_out = encoder_block(x, src_mask=None)
    print(f"Encoder output shape: {enc_out.shape}")
    
    # Decoder Forward
    # In reality tgt_mask would be a causal mask to prevent looking ahead
    dec_out = decoder_block(x, enc_out, src_mask=None, tgt_mask=None)
    print(f"Decoder output shape: {dec_out.shape}")
