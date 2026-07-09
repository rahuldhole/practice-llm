import pytest
import torch
from pytorch_patterns.custom_module import CustomLinear

def test_custom_linear_initialization():
    in_features, out_features = 8, 4
    
    # Test Xavier init
    fc_xavier = CustomLinear(in_features, out_features, init_method="xavier")
    assert fc_xavier.weight.shape == (out_features, in_features)
    assert fc_xavier.bias.shape == (out_features,)
    assert (fc_xavier.bias == 0.0).all()
    
    # Test Constant init
    fc_const = CustomLinear(in_features, out_features, init_method="constant")
    assert (fc_const.weight == 0.1).all()
    assert (fc_const.bias == 0.0).all()
    
    # Invalid init method
    with pytest.raises(ValueError):
        CustomLinear(in_features, out_features, init_method="invalid")


def test_custom_linear_forward_and_buffers():
    in_features, out_features = 4, 2
    fc = CustomLinear(in_features, out_features, init_method="xavier")
    
    # Check initial buffer states
    assert fc.step_count.item() == 0
    assert (fc.running_mean_activation == 0.0).all()
    
    # Simulate single input forward pass
    x = torch.randn(3, in_features)
    out = fc(x)
    
    assert out.shape == (3, out_features)
    assert fc.step_count.item() == 1
    # running_mean should be non-zero after update
    assert not (fc.running_mean_activation == 0.0).all()
    
    # Second forward pass
    _ = fc(x)
    assert fc.step_count.item() == 2
    
    # Check parameters have gradients after backward
    loss = out.sum()
    loss.backward()
    assert fc.weight.grad is not None
    assert fc.bias.grad is not None
    
    # Verify buffers do NOT require gradients
    assert fc.step_count.grad is None
    assert fc.running_mean_activation.grad is None
