import pytest
from python_concepts.oop_meta import Layer, Linear, Activation, LAYER_REGISTRY, create_layer

def test_abc_enforcement():
    # Attempting to instantiate Layer should fail because forward/get_params are abstract
    with pytest.raises(TypeError):
        Layer("base")  # Can't instantiate abstract class Layer with abstract methods


def test_properties_validation():
    lin = Linear(in_features=3, out_features=2, name="mylin")
    assert lin.name == "mylin"
    
    # Change name
    lin.name = "newlin"
    assert lin.name == "newlin"
    
    # Invalid name
    with pytest.raises(ValueError):
        lin.name = ""
        
    with pytest.raises(ValueError):
        lin.name = None
        
    # Check weight setter validation
    with pytest.raises(ValueError):
        lin.weight = [[1.0, 1.0]]  # out_features is 2, so list must have length 2


def test_registry_tracking():
    # Verify linear and relu are in register
    assert "linear" in LAYER_REGISTRY
    assert "relu" in LAYER_REGISTRY
    
    assert LAYER_REGISTRY["linear"] is Linear
    assert LAYER_REGISTRY["relu"] is Activation


def test_factory_creation():
    config = {
        "type": "linear",
        "args": {
            "in_features": 4,
            "out_features": 3,
            "name": "factory_lin"
        }
    }
    
    layer = create_layer(config)
    assert isinstance(layer, Linear)
    assert layer.name == "factory_lin"
    assert layer.in_features == 4
    assert layer.out_features == 3
    
    # Unknown layer type
    with pytest.raises(ValueError):
        create_layer({"type": "non_existent"})


def test_forward_pass():
    lin = Linear(in_features=2, out_features=2, name="lin")
    lin.weight = [[2.0, 0.0], [0.0, 3.0]]  # W
    lin._bias = [1.0, -1.0]                # b
    
    # y = Wx + b
    # x = [2.0, 2.0]
    # y[0] = 2.0 * 2.0 + 0.0 * 2.0 + 1.0 = 5.0
    # y[1] = 0.0 * 2.0 + 3.0 * 2.0 - 1.0 = 5.0
    out = lin.forward([2.0, 2.0])
    assert out == [5.0, 5.0]
    
    # ReLU layer
    relu = Activation("relu")
    assert relu.forward([5.0, -3.0]) == [5.0, 0.0]
