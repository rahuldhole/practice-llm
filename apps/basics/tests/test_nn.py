import random
import pytest
from basics.autograd import Value
from basics.nn import Neuron, Layer, MLP

def test_neuron():
    random.seed(42)
    n = Neuron(3)
    assert len(n.parameters()) == 4  # 3 weights + 1 bias
    x = [Value(1.0), Value(-2.0), Value(0.5)]
    out = n(x)
    assert isinstance(out, Value)

def test_layer():
    random.seed(42)
    lay = Layer(3, 2)
    assert len(lay.parameters()) == 8  # 2 neurons * (3 weights + 1 bias)
    x = [1.0, -1.0, 0.0]
    out = lay(x)
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(val, Value) for val in out)

def test_mlp_forward_and_backward():
    random.seed(42)
    model = MLP(3, [4, 4, 1])
    assert len(model.parameters()) == (3 * 4 + 4) + (4 * 4 + 4) + (4 * 1 + 1)
    
    x = [1.0, 2.0, -1.0]
    out = model(x)
    assert isinstance(out, Value)
    
    # Check backpropagation propagates gradients to parameters
    out.backward()
    for p in model.parameters():
        assert p.grad != 0.0

def test_xor_training():
    random.seed(1337)
    model = MLP(2, [4, 1])
    
    xs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    ys = [-1.0, 1.0, 1.0, -1.0]
    
    # Check that a short training run reduces loss
    ypred = [model(x) for x in xs]
    initial_loss = sum((yp - y)**2 for y, yp in zip(ys, ypred)) / len(ys)
    
    for epoch in range(150):
        # Forward pass
        ypred = [model(x) for x in xs]
        loss = sum((yp - y)**2 for y, yp in zip(ys, ypred)) / len(ys)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Gradient update
        for p in model.parameters():
            p.data -= 0.1 * p.grad
            
    final_loss = sum((yp - y)**2 for y, yp in zip(ys, [model(x) for x in xs])) / len(ys)
    
    # Verify training has successfully reduced the MSE loss
    assert final_loss.data < initial_loss.data
    # Assert final loss is low (usually < 0.1 for a fitted XOR on 150 epochs)
    assert final_loss.data < 0.15
