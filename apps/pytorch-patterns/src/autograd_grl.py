import torch
import torch.nn as nn

class GradientReversalFunction(torch.autograd.Function):
    """
    A custom math gate that passes values forward unchanged,
    but reverses and scales the gradients flowing backward.
    Like a one-way mirror changing gradient direction.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        # Save alpha constant for the backward math pass
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient direction (negate) and scale by alpha
        grad_input = grad_output.neg() * ctx.alpha
        
        # We must return one gradient per input parameter.
        # Since alpha is a scalar configuration, it does not learn, so we return None.
        return grad_input, None


class GradientReversal(nn.Module):
    """
    A simple wrapper layer to apply our custom autograd Function.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        # We invoke the custom autograd function using .apply
        return GradientReversalFunction.apply(x, self.alpha)


if __name__ == "__main__":
    print("--- Gradient Reversal Demo ---")
    x = torch.tensor([5.0], requires_grad=True)
    grl = GradientReversal(alpha=2.0)
    
    y = grl(x)
    loss = y * 10.0
    loss.backward()
    
    # Normally dy/dx is 10.0. With GRL (alpha=2.0), it should be -2.0 * 10.0 = -20.0
    print(f"Input: {x.item()} -> GRL Output: {y.item()}")
    print(f"Gradient (reversed & scaled): {x.grad.item()}")
