import torch
import torch.nn as nn
from llm.embedding import EmbeddingLayer
from llm.attention import SingleHeadAttention

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.attention = SingleHeadAttention(d_model)
        
        # Feed-forward network (simple 2-layer MLP)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Final linear layer to project back to vocabulary size
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        x: Input token IDs, shape (batch_size, seq_len)
        """
        # 1. Get embeddings + positional encoding
        x = self.embedding(x)
        
        # 2. Self-Attention with residual connection & layer norm
        attn_out = self.attention(x)
        x = self.ln1(x + attn_out)
        
        # 3. Feed Forward with residual connection & layer norm
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        # 4. Predict logits for each token in the sequence
        logits = self.lm_head(x) # (batch_size, seq_len, vocab_size)
        
        return logits
