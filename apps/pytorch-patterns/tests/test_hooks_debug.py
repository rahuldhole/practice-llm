import pytest
import torch
import torch.nn as nn
from pytorch_patterns.hooks_debug import ModelInspector

def test_model_inspector_registration_and_stats():
    # Setup simple model with one target layer
    model = nn.Sequential(
        nn.Linear(5, 3),
        nn.ReLU()
    )
    
    inspector = ModelInspector(model, target_types=(nn.Linear,))
    
    # Assert initial lists are empty
    assert len(inspector.hook_handles) == 2  # 1 forward + 1 backward hook
    layer_key = list(inspector.activations.keys())[0]
    assert len(inspector.activations[layer_key]) == 0
    assert len(inspector.gradients[layer_key]) == 0
    
    # Run forward pass
    x = torch.randn(2, 5)
    out = model(x)
    
    # Check activation captured
    assert len(inspector.activations[layer_key]) == 1
    assert "mean" in inspector.activations[layer_key][0]
    assert "std" in inspector.activations[layer_key][0]
    assert inspector.activations[layer_key][0]["shape"] == [2, 3]
    
    # Run backward pass
    loss = out.sum()
    loss.backward()
    
    # Check gradient captured
    assert len(inspector.gradients[layer_key]) == 1
    assert "mean" in inspector.gradients[layer_key][0]
    assert "norm" in inspector.gradients[layer_key][0]
    
    # Remove hooks
    inspector.remove_hooks()
    assert len(inspector.hook_handles) == 0
    
    # Run another forward/backward step, confirm stats count does not increment
    out2 = model(x)
    loss2 = out2.sum()
    loss2.backward()
    
    assert len(inspector.activations[layer_key]) == 1
    assert len(inspector.gradients[layer_key]) == 1
