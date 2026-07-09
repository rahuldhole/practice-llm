import torch
import torch.nn as nn
import torch.nn.init as init

class CustomLinear(nn.Module):
    """
    A custom implementation of a Linear layer (fully connected layer) that
    demonstrates advanced nn.Module patterns:
      1. Registering trainable parameters with nn.Parameter.
      2. Registering non-trainable state tracking buffers with register_buffer.
      3. Implementing manual initialization options for parameters.
      4. Updating state buffers in the forward pass.
    """
    def __init__(self, in_features, out_features, init_method="xavier"):
        """
        Args:
            in_features (int): Dimension of each input sample.
            out_features (int): Dimension of each output sample.
            init_method (str): Method to initialize weights ('xavier', 'kaiming', or 'constant').
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Define trainable parameters.
        # Wrappers around torch.Tensor that are registered as module parameters
        # and will appear in parameters() and state_dict() and receive gradients.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # 2. Define non-trainable state buffers.
        # Buffers are saved in the state_dict and moved with model.to(device),
        # but are not treated as trainable parameters (requires_grad = False).
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("running_mean_activation", torch.zeros(out_features))
        
        # 3. Apply weights initialization
        self.initialize_parameters(init_method)

    def initialize_parameters(self, method):
        """
        Demonstrates common parameter initialization techniques in PyTorch.
        """
        if method == "xavier":
            # Glorot/Xavier uniform initialization (good for tanh, sigmoid activation functions)
            init.xavier_uniform_(self.weight)
            init.constant_(self.bias, 0.0)
        elif method == "kaiming":
            # He/Kaiming normal initialization (good for ReLU, LeakyReLU activations)
            init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
            init.constant_(self.bias, 0.01)  # small positive bias for ReLU activation
        elif method == "constant":
            # Constant initialization (mostly for debugging or specific architectures)
            init.constant_(self.weight, 0.1)
            init.constant_(self.bias, 0.0)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def forward(self, x):
        """
        Defines the computation performed at every forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Linear layer calculation: y = x W^T + b
        output = torch.matmul(x, self.weight.t()) + self.bias
        
        # Update running buffers (not tracked in computational graph for gradients)
        with torch.no_grad():
            self.step_count += 1
            # Update running mean of activations (exponential moving average)
            batch_mean = output.mean(dim=0)
            alpha = 0.1
            self.running_mean_activation.copy_(
                (1.0 - alpha) * self.running_mean_activation + alpha * batch_mean
            )
            
        return output


if __name__ == "__main__":
    print("--- PyTorch Custom Modules & Initialization Demo ---")
    
    # Instantiate custom linear layers with different initializations
    print("\n1. Instantiating layers with different weight initialization methods:")
    fc_xavier = CustomLinear(4, 2, init_method="xavier")
    fc_kaiming = CustomLinear(4, 2, init_method="kaiming")
    fc_constant = CustomLinear(4, 2, init_method="constant")
    
    print("Xavier Weights:\n", fc_xavier.weight.data)
    print("Kaiming Weights:\n", fc_kaiming.weight.data)
    print("Constant Weights:\n", fc_constant.weight.data)
    
    # Check parameters and buffers in the state dict
    print("\n2. Checking state_dict contents (showing parameters and buffers):")
    for key, val in fc_xavier.state_dict().items():
        print(f" - {key}: shape={val.shape}, is_parameter={key in [n for n, _ in fc_xavier.named_parameters()]}")
        
    # Simulate a forward pass
    print("\n3. Simulating forward passes:")
    # Batch size 3, features 4
    x = torch.randn(3, 4)
    print("Input Tensor:\n", x)
    
    # Run the layer twice to see buffers change
    out1 = fc_xavier(x)
    print("\nForward Pass 1 Output:\n", out1)
    print(f"Buffers after 1 pass: step_count={fc_xavier.step_count.item()}, running_mean={fc_xavier.running_mean_activation.tolist()}")
    
    out2 = fc_xavier(x)
    print("\nForward Pass 2 Output:\n", out2)
    print(f"Buffers after 2 passes: step_count={fc_xavier.step_count.item()}, running_mean={fc_xavier.running_mean_activation.tolist()}")
    
    # Verify gradient computation works
    loss = out2.sum()
    loss.backward()
    print("\n4. Verifying gradient computation:")
    print("Weight gradient (dLoss/dWeight):\n", fc_xavier.weight.grad)
    print("Bias gradient (dLoss/dBias):\n", fc_xavier.bias.grad)
