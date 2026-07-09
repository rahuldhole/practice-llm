import pytest
import torch
import torch.nn as nn
from pytorch_patterns.module_params import CustomLinear
from pytorch_patterns.module_init import initialize_weights

def test_custom_linear_parameters_and_buffers():
    layer = CustomLinear(in_features=3, out_features=2)
    
    assert layer.step_count.item() == 0
    
    x = torch.randn(2, 3)
    out = layer(x)
    
    assert out.shape == (2, 2)
    assert layer.step_count.item() == 1
    
    # Check backward pass computes gradients w.r.t parameters
    loss = out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert layer.step_count.grad is None


def test_initialize_weights():
    fc = nn.Linear(4, 2)
    
    initialize_weights(fc, method="constant")
    assert (fc.weight == 0.5).all()
    assert (fc.bias == 0.0).all()
    
    initialize_weights(fc, method="xavier")
    # check that values are set and not all equal
    assert not (fc.weight == 0.5).all()
    
    with pytest.raises(ValueError):
        initialize_weights(fc, method="invalid")
