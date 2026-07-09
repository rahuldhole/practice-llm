import pytest
import torch
from pytorch_patterns.autograd_grl import GradientReversal
from pytorch_patterns.loss_focal import FocalLoss

def test_gradient_reversal_layer():
    x = torch.tensor([4.0], requires_grad=True)
    grl = GradientReversal(alpha=3.0)
    
    y = grl(x)
    assert torch.equal(y, x)
    
    loss = y * 5.0  # dy/dx would normally be 5.0
    loss.backward()
    
    # GRL backward formula: -alpha * grad_output -> -3.0 * 5.0 = -15.0
    assert x.grad.item() == -15.0


def test_focal_loss():
    loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
    
    logits = torch.tensor([[2.0], [-2.0]])
    targets = torch.tensor([[1.0], [0.0]])
    
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
