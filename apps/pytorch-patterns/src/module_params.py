import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    A custom layer that shows parameters (weights that learn)
    vs buffers (values that only track metrics, like a step counter).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Trainable Parameters (Weight and Bias)
        # PyTorch updates these automatically during training
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # 2. Tracking Buffer (Step Counter)
        # Stays saved in checkpoints, but is not updated by learning math
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        # Linear layer math: y = x * W^T + bias
        output = torch.matmul(x, self.weight.t()) + self.bias
        
        # Update our tracking step counter (detaching it from gradient calculations)
        with torch.no_grad():
            self.step_count += 1
            
        return output


if __name__ == "__main__":
    print("--- Parameters & Buffers Demo ---")
    layer = CustomLinear(in_features=3, out_features=2)
    
    # Check parameters and buffers
    print("Parameters (Trainable):", [n for n, _ in layer.named_parameters()])
    print("Buffers (Tracking only):", [n for n, _ in layer.named_buffers()])
    
    # Run a forward step
    x = torch.randn(1, 3)
    out = layer(x)
    print(f"Step Count after forward: {layer.step_count.item()}")
