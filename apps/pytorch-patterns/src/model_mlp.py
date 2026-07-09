import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    A simple Multilayer Perceptron (MLP) classification network.
    Like a stack of layers processing inputs step-by-step.
    """
    def __init__(self, in_features=2, hidden_dim=8):
        super().__init__()
        # A simple pipeline of linear connections and ReLU activations
        self.pipeline = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Raw score output (logit)
        )

    def forward(self, x):
        # Passes input coordinates through our layers pipeline
        return self.pipeline(x)


if __name__ == "__main__":
    print("--- Simple MLP Model Demo ---")
    model = SimpleMLP(in_features=2, hidden_dim=4)
    x = torch.randn(2, 2)
    out = model(x)
    print("Input coords:\n", x)
    print("Output score logits:\n", out)
