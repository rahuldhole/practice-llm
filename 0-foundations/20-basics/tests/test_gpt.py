import torch
import pytest
from basics.gpt import TinyGPT

def test_tiny_gpt_forward_shape():
    torch.manual_seed(42)
    vocab_size = 10
    n_embd = 16
    n_head = 2
    n_layer = 1
    block_size = 8
    
    model = TinyGPT(vocab_size, n_embd, n_head, n_layer, block_size)
    
    # Batch size = 2, Seq len = 5
    x = torch.randint(0, vocab_size, (2, 5))
    logits, loss = model(x)
    
    assert logits.shape == (2, 5, vocab_size)
    assert loss is None
    
    # Forward pass with targets
    y = torch.randint(0, vocab_size, (2, 5))
    logits, loss = model(x, y)
    assert logits.shape == (2, 5, vocab_size)
    assert loss is not None
    assert loss.item() > 0.0

def test_tiny_gpt_generation():
    torch.manual_seed(42)
    vocab_size = 10
    n_embd = 16
    n_head = 2
    n_layer = 1
    block_size = 8
    
    model = TinyGPT(vocab_size, n_embd, n_head, n_layer, block_size)
    
    x = torch.randint(0, vocab_size, (1, 3)) # seq_len = 3
    out = model.generate(x, max_new_tokens=5)
    
    assert out.shape == (1, 8) # 3 original + 5 new tokens

def test_tiny_gpt_overfitting():
    torch.manual_seed(42)
    vocab_size = 6
    n_embd = 16
    n_head = 2
    n_layer = 1
    block_size = 4
    
    model = TinyGPT(vocab_size, n_embd, n_head, n_layer, block_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Toy training instance
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, 4]]) # shift by 1
    
    # Get initial loss
    _, initial_loss = model(x, y)
    
    # Short optimization loop
    for _ in range(50):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    _, final_loss = model(x, y)
    assert final_loss.item() < initial_loss.item()
    assert final_loss.item() < 0.1 # Should fit easily
