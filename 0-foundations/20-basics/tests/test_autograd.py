import pytest
import torch
from basics.autograd import Value

def test_autograd_basic():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    
    x_val, y_val = x, y
    
    # Let's do the same in PyTorch
    x_t = torch.Tensor([-4.0]).double()
    x_t.requires_grad = True
    
    z_t = 2 * x_t + 2 + x_t
    q_t = torch.relu(z_t) + z_t * x_t
    h_t = torch.relu(z_t * z_t)
    y_t = h_t + q_t + q_t * x_t
    y_t.backward()
    
    assert abs(y_val.data - y_t.item()) < 1e-9
    assert abs(x_val.grad - x_t.grad.item()) < 1e-9

def test_autograd_activations():
    # Tanh test
    a = Value(0.5)
    b = a.tanh()
    b.backward()
    
    a_t = torch.Tensor([0.5]).double()
    a_t.requires_grad = True
    b_t = torch.tanh(a_t)
    b_t.backward()
    
    assert abs(b.data - b_t.item()) < 1e-9
    assert abs(a.grad - a_t.grad.item()) < 1e-9

    # Sigmoid test
    x = Value(-1.5)
    y = x.sigmoid()
    y.backward()
    
    x_t = torch.Tensor([-1.5]).double()
    x_t.requires_grad = True
    y_t = torch.sigmoid(x_t)
    y_t.backward()
    
    assert abs(y.data - y_t.item()) < 1e-9
    assert abs(x.grad - x_t.grad.item()) < 1e-9

def test_autograd_division_subtraction():
    a = Value(3.0)
    b = Value(4.0)
    c = a / b
    d = a - b
    e = c + d
    e.backward()
    
    a_t = torch.Tensor([3.0]).double()
    a_t.requires_grad = True
    b_t = torch.Tensor([4.0]).double()
    b_t.requires_grad = True
    
    c_t = a_t / b_t
    d_t = a_t - b_t
    e_t = c_t + d_t
    e_t.backward()
    
    assert abs(e.data - e_t.item()) < 1e-9
    assert abs(a.grad - a_t.grad.item()) < 1e-9
    assert abs(b.grad - b_t.grad.item()) < 1e-9
