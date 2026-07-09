import torch
import torch.nn as nn
import torch.nn.init as init

def initialize_weights(layer, method="xavier"):
    """
    Fills a layer's weights and biases with strategic starting values.
    Like setting standard starting volume knobs on speakers.
    """
    if not hasattr(layer, "weight") or not hasattr(layer, "bias"):
        return
        
    if method == "xavier":
        # Glorot/Xavier initialization: Good for Tanh/Sigmoid activations
        init.xavier_uniform_(layer.weight)
        init.constant_(layer.bias, 0.0)
        
    elif method == "kaiming":
        # He/Kaiming initialization: Good for ReLU activations
        init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        init.constant_(layer.bias, 0.01)
        
    elif method == "constant":
        # Sets weights to a constant value for debugging
        init.constant_(layer.weight, 0.5)
        init.constant_(layer.bias, 0.0)
        
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    print("--- Weights Initialization Demo ---")
    fc = nn.Linear(3, 2)
    
    print("Before init weight:", fc.weight.data)
    initialize_weights(fc, method="constant")
    print("After constant init weight:", fc.weight.data)
