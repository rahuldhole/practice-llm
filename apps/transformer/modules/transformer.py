import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.embeddings_and_positional import InputEmbeddings, PositionalEncoding
from modules.encoder_decoder import EncoderBlock, DecoderBlock

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(layers[0].self_attention_block.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = nn.LayerNorm(layers[0].self_attention_block.d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We output logits; CrossEntropyLoss expects logits
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 src_seq_len: int, 
                 tgt_seq_len: int, 
                 d_model: int = 512, 
                 N: int = 6, 
                 h: int = 8, 
                 dropout: float = 0.1, 
                 d_ff: int = 2048):
        """
        The full Transformer architecture.
        """
        super().__init__()
        
        # 1. Embeddings and Positional Encodings
        self.src_embed = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
        self.src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        
        # 2. Encoder and Decoder Stacks
        encoder_blocks = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.encoder = Encoder(encoder_blocks)
        
        decoder_blocks = nn.ModuleList([DecoderBlock(d_model, h, d_ff, dropout) for _ in range(N)])
        self.decoder = Decoder(decoder_blocks)
        
        # 3. Final Projection
        self.projection = ProjectionLayer(d_model, tgt_vocab_size)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.projection(decoder_output)


def causal_mask(size: int) -> torch.Tensor:
    """Creates a triangular causal mask for the decoder."""
    # upper triangular matrix of 1s
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.int)
    # return True where 0 (i.e. lower triangular part is kept)
    return mask == 0


if __name__ == "__main__":
    print("Testing Full Transformer")
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 10
    vocab_size = 100
    
    # Initialize tiny model
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        d_model=16,
        N=2,
        h=4,
        d_ff=64
    )
    
    # Mock data
    src = torch.randint(0, vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_len))
    
    # Masks
    # (batch, seq_len, seq_len)
    src_mask = torch.ones(batch_size, 1, 1, src_seq_len)
    # Target mask needs both padding mask and causal mask
    tgt_mask = causal_mask(tgt_seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, tgt_seq_len, tgt_seq_len)
    
    # Forward pass
    out = model(src, tgt, src_mask, tgt_mask)
    print(f"Final output logits shape: {out.shape} -> Expected: (batch, tgt_seq_len, vocab_size)")
