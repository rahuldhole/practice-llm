import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        Input embeddings convert token IDs into dense vectors of size d_model.
        
        Args:
            d_model (int): The dimension of the embeddings.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of token IDs, shape (batch_size, seq_len)
        Returns:
            torch.Tensor: Embedded tensor of shape (batch_size, seq_len, d_model)
        """
        # In the original paper, embeddings are multiplied by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.1):
        """
        Positional encoding injects information about the relative or absolute
        position of the tokens in the sequence.
        
        Args:
            d_model (int): The dimension of the embeddings.
            max_seq_len (int): The maximum length of the sequence.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        
        # Create a vector of shape (max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the sinusoidal formulas
        # shape: (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension -> (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register pe as a buffer so it's saved in the state_dict but not trained
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Embedded input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: Tensor with positional encodings added, same shape as input.
        """
        seq_len = x.size(1)
        # Add the positional encoding to the input (broadcasting over batch)
        # We only take the encoding up to the current sequence length
        x = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(x)


if __name__ == "__main__":
    print("Testing Embeddings and Positional Encoding")
    batch_size = 2
    seq_len = 5
    vocab_size = 100
    d_model = 16
    
    # Create random token IDs
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape (token IDs): {x.shape}")
    
    # Initialize modules
    emb_layer = InputEmbeddings(d_model, vocab_size)
    pe_layer = PositionalEncoding(d_model, max_seq_len=50)
    
    # Forward pass
    embeddings = emb_layer(x)
    print(f"Embeddings shape: {embeddings.shape}")
    
    output = pe_layer(embeddings)
    print(f"Output shape after Positional Encoding: {output.shape}")
