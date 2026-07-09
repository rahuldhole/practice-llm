import torch
import torch.nn as nn
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    """
    A custom PyTorch Autograd Function that implements a Gradient Reversal Layer (GRL).
    
    During the forward pass, it acts as an identity mapping (passes input through unchanged).
    During the backward pass, it multiplies the incoming gradient by a negative scalar (-alpha).
    This is commonly used in domain adaptation networks to align representations.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """
        Args:
            ctx: Context object to stash information for backward computation.
            x (torch.Tensor): Input tensor.
            alpha (float): Scalar scaling factor for the gradient reversal.
            
        Returns:
            torch.Tensor: Same as input x.
        """
        ctx.alpha = alpha
        # Return a copy or just the tensor itself (identity)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output (torch.Tensor): Gradients flowing back from the subsequent layer.
            
        Returns:
            torch.Tensor: Reversed and scaled gradients w.r.t the input x.
            None: No gradient for alpha (since it is a hyperparameter scalar).
        """
        # Reverse the gradient and scale by alpha
        grad_input = grad_output.neg() * ctx.alpha
        return grad_input, None


class GradientReversal(nn.Module):
    """
    A Module wrapper for GradientReversalFunction.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        # We invoke the custom autograd function using the .apply method
        return GradientReversalFunction.apply(x, self.alpha)


class FocalLoss(nn.Module):
    """
    A custom loss function implementing Binary Focal Loss.
    
    Focal Loss is designed to address class imbalance by down-weighting the loss 
    assigned to easy-to-classify (well-classified) examples, focusing instead 
    on hard examples.
    
    Formula:
        FL(p_t) = - alpha * (1 - p_t)^gamma * log(p_t)
        where p_t is the probability of the correct class.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (float): Balancing factor for class imbalance (weight of positive class).
            gamma (float): Focusing parameter to dynamically scale down loss of easy examples.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw predictions from the model (shape: Batch x 1).
            targets (torch.Tensor): Binary target labels (0 or 1, shape: Batch x 1).
        """
        # Calculate standard binary cross entropy loss per element (without reduction)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # Calculate probabilities of predictions
        probs = torch.sigmoid(logits)
        
        # p_t is the model's estimated probability for the true class label
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate the focal scale weight factor: (1 - p_t)^gamma
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # Compute individual focal loss elements
        loss = focal_weight * bce_loss
        
        # Apply class-balancing weight alpha
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_factor * loss
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    print("--- PyTorch Custom Autograd & Loss Functions Demo ---")
    
    # 1. Test Gradient Reversal Layer
    print("\n1. Testing Gradient Reversal Layer (GRL):")
    # Define an input tensor with gradient tracking enabled
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    print("Input Tensor (x):", x)
    
    # Run through GRL with alpha = 0.5
    grl = GradientReversal(alpha=0.5)
    y = grl(x)
    print("Forward Pass Output (y = grl(x)):", y)
    
    # Define dummy loss and calculate backward gradients
    loss = (y * y).sum()
    loss.backward()
    
    # Normally, d(x^2)/dx would be 2*x -> [4.0, -6.0].
    # Under GRL with alpha=0.5, it should reverse and scale: -0.5 * 2 * x -> [-2.0, 3.0].
    print("Computed gradient (x.grad):", x.grad)
    print("Expected gradient (reversed & scaled): [-2.0, 3.0]")
    
    # 2. Test Custom Focal Loss
    print("\n2. Testing Binary Focal Loss:")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Target is positive class (1.0)
    targets = torch.ones(3, 1)
    
    # Case A: Correct and confident predictions (logits = 5.0 -> prob ~ 0.99)
    logits_easy = torch.tensor([[5.0], [5.0], [5.0]])
    # Case B: Confident but incorrect prediction (logits = -5.0 -> prob ~ 0.006)
    logits_hard = torch.tensor([[-5.0], [-5.0], [-5.0]])
    
    loss_easy = focal_loss(logits_easy, targets)
    loss_hard = focal_loss(logits_hard, targets)
    
    print("Easy prediction loss (confidence=99% correct):", loss_easy.item())
    print("Hard prediction loss (confidence=99% incorrect):", loss_hard.item())
    print(f"Ratio of Hard Loss to Easy Loss: {loss_hard.item() / (loss_easy.item() + 1e-9):.2f}x")
