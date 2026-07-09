import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def generate_synthetic_data(num_samples=100):
    """
    Creates random 2D coordinate points. Label is 1 if point is inside
    a circle of radius 1, otherwise 0.
    """
    X = (torch.rand(num_samples, 2) * 4) - 2
    y = (X.pow(2).sum(dim=1) < 1.0).float().unsqueeze(1)
    
    # Split into 80% train, 20% validation datasets
    split = int(num_samples * 0.8)
    return TensorDataset(X[:split], y[:split]), TensorDataset(X[split:], y[split:])


class Trainer:
    """
    A simple checklist runner that handles model training and validation steps.
    """
    def __init__(self, model, optimizer, scheduler, criterion, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.max_grad_norm = max_grad_norm

    def train_epoch(self, dataloader):
        # 1. Put model in learning mode (enables batch stats and dropout)
        self.model.train()
        total_loss = 0.0
        
        for X, y in dataloader:
            # 2. Clear old gradients so they don't pile up
            self.optimizer.zero_grad()
            
            # 3. Forward pass: compute predictions
            logits = self.model(X)
            loss = self.criterion(logits, y)
            
            # 4. Backward pass: compute weight correction scales
            loss.backward()
            
            # 5. Clip gradients to prevent them from growing too large (exploding)
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
            # 6. Apply corrections to updates
            self.optimizer.step()
            total_loss += loss.item() * X.size(0)
            
        # 7. Update learning rate decay schedule
        if self.scheduler is not None:
            self.scheduler.step()
            
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader):
        # 1. Put model in freeze mode (stops batch updates and dropout)
        self.model.eval()
        total_loss = 0.0
        
        # 2. Tell PyTorch not to waste memory tracking gradients during evaluation
        with torch.no_grad():
            for X, y in dataloader:
                logits = self.model(X)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * X.size(0)
                
        return total_loss / len(dataloader.dataset)
