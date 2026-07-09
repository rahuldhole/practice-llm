import pytest
import torch
import torch.nn as nn
from pytorch_patterns.hooks_inspector import ModelInspector

def test_hooks_inspector():
    model = nn.Sequential(
        nn.Linear(3, 2)
    )
    
    inspector = ModelInspector(model, target_type=nn.Linear)
    
    # Assert initial lists are empty
    layer_key = list(inspector.activations.keys())[0]
    assert len(inspector.activations[layer_key]) == 0
    assert len(inspector.gradients[layer_key]) == 0
    
    # Run forward pass
    x = torch.randn(2, 3)
    out = model(x)
    
    # Check activation captured
    assert len(inspector.activations[layer_key]) == 1
    assert inspector.activations[layer_key][0]["shape"] == [2, 2]
    
    # Run backward pass
    loss = out.sum()
    loss.backward()
    
    # Check gradient captured
    assert len(inspector.gradients[layer_key]) == 1
    assert "mean" in inspector.gradients[layer_key][0]
    assert "norm" in inspector.gradients[layer_key][0]
    
    # Remove hooks
    inspector.remove()
    assert len(inspector.hook_handles) == 0
