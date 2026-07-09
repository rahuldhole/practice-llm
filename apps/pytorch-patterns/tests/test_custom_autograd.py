import pytest
import torch
from pytorch_patterns.custom_autograd import GradientReversal, FocalLoss

def test_gradient_reversal():
    x = torch.tensor([5.0, -10.0], requires_grad=True)
    grl = GradientReversal(alpha=2.0)
    
    y = grl(x)
    # Forward pass: identity
    assert torch.equal(y, x)
    
    # Backward pass: reverse and scale gradient by 2.0
    loss = (y * 3.0).sum()  # dy/dx would normally be 3.0
    loss.backward()
    
    # GRL backward gradient: -alpha * grad_output -> -2.0 * 3.0 = -6.0
    assert torch.equal(x.grad, torch.tensor([-6.0, -6.0]))


def test_focal_loss_values():
    # Setup focal loss
    loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction="mean")
    
    logits = torch.tensor([[0.0], [2.0], [-2.0]])
    targets = torch.tensor([[1.0], [1.0], [0.0]])
    
    # Calculate loss manually/conceptually:
    # focal loss should compute positive non-nan values
    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    
    # Verify focal loss handles extreme values safely without NaNs
    logits_extreme = torch.tensor([[100.0], [-100.0]])
    targets_extreme = torch.tensor([[1.0], [0.0]])
    
    loss_extreme = loss_fn(logits_extreme, targets_extreme)
    # Confident and correct predictions should have loss near 0
    assert loss_extreme.item() < 1e-4
