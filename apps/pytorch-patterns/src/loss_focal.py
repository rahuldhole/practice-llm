import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    A custom loss function that down-weights easy samples to focus
    gradient updates on harder, uncertain examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 1. Compute standard binary cross entropy loss for each sample
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        # 2. Get probabilities from logits
        probs = torch.sigmoid(logits)
        
        # 3. Find the probability of predicting the correct class label
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 4. Compute focal weight scale: (1 - p_t)^gamma
        # Highly confident correct predictions (p_t near 1) get near 0 weight
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # 5. Apply focal scale and class-imbalance weight alpha
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_factor * focal_weight * bce_loss
        
        # 6. Return combined loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


if __name__ == "__main__":
    print("--- Focal Loss Demo ---")
    loss_fn = FocalLoss()
    logits = torch.tensor([[5.0], [-5.0]]) # confident correct vs confident incorrect
    targets = torch.tensor([[1.0], [1.0]])
    print(f"Easy example loss: {loss_fn(logits[0:1], targets[0:1]).item():.6f}")
    print(f"Hard example loss: {loss_fn(logits[1:2], targets[1:2]).item():.6f}")
