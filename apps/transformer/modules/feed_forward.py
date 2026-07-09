import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Position-wise Feed-Forward Network.
        This block applies two linear transformations with a ReLU activation in between.
        
        Args:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the hidden layer in the feed-forward network.
            dropout (float): Dropout probability.
        """
        super().__init__()
        # First linear transformation: projects from d_model to d_ff
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        # Second linear transformation: projects from d_ff back to d_model
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)
        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, d_model)
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        # The activation used in the original paper is ReLU.
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

if __name__ == "__main__":
    print("Testing Position-wise Feed-Forward Network")
    batch_size = 2
    seq_len = 5
    d_model = 16
    d_ff = 64
    
    # Create random embeddings
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # Initialize Feed-Forward Network
    ffn = FeedForwardBlock(d_model=d_model, d_ff=d_ff)
    
    # Forward pass
    output = ffn(x)
    print(f"Output shape after FFN: {output.shape}")
