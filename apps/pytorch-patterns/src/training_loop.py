import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class SimpleMLP(nn.Module):
    """
    A simple Multilayer Perceptron (MLP) for binary classification.
    """
    def __init__(self, in_features=2, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs raw logits for binary classification
        )

    def forward(self, x):
        return self.net(x)


def generate_synthetic_data(num_samples=600):
    """
    Generates a synthetic binary classification dataset (points inside vs outside a circle).
    """
    # Create random 2D points in range [-2, 2]
    X = (torch.rand(num_samples, 2) * 4) - 2
    # Label is 1 if point is inside a circle of radius 1, else 0
    y = (X.pow(2).sum(dim=1) < 1.0).float().unsqueeze(1)
    
    # Split into train and validation sets (80/20)
    split = int(num_samples * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    return train_dataset, val_dataset


class Trainer:
    """
    A robust trainer class implementing standard training loop patterns:
      1. Toggle train/eval mode (model.train() vs model.eval()).
      2. Forward pass, loss calculation, backpropagation.
      3. Optimizer gradient zeroing and stepping.
      4. Gradient norm clipping to prevent exploding gradients.
      5. Learning rate scheduler stepping.
      6. Evaluation with torch.no_grad().
    """
    def __init__(self, model, optimizer, scheduler, criterion, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.max_grad_norm = max_grad_norm

    def train_epoch(self, dataloader):
        """
        Trains the model for one epoch.
        """
        # 1. Ensure the model is in training mode (enables dropout, batch norm updates, etc.)
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for X, y in dataloader:
            # 2. Reset gradients from the previous step
            self.optimizer.zero_grad()
            
            # 3. Forward pass
            logits = self.model(X)
            loss = self.criterion(logits, y)
            
            # 4. Backward pass (computes gradients of loss w.r.t. parameters)
            loss.backward()
            
            # 5. Gradient Clipping
            # Clips gradients of the model parameters to prevent exploding gradients.
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                
            # 6. Optimizer Step
            # Updates parameters based on computed gradients.
            self.optimizer.step()
            
            # Accumulate statistics
            total_loss += loss.item() * X.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total_samples += X.size(0)
            
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        
        # 7. Scheduler Step (step once per epoch, or per iteration depending on scheduler type)
        if self.scheduler is not None:
            self.scheduler.step()
            
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader):
        """
        Evaluates the model on a validation/test dataset.
        """
        # 1. Set model to evaluation mode (disables dropout, fixes batch norm stats)
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # 2. Disable gradient tracking during inference to save memory and computations
        with torch.no_grad():
            for X, y in dataloader:
                logits = self.model(X)
                loss = self.criterion(logits, y)
                
                total_loss += loss.item() * X.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += (preds == y).sum().item()
                total_samples += X.size(0)
                
        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples
        return val_loss, val_acc


if __name__ == "__main__":
    print("--- PyTorch Robust Training Loop Demo ---")
    
    # 1. Generate synthetic binary classification datasets
    train_ds, val_ds = generate_synthetic_data(num_samples=1000)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 2. Initialize Model, Loss (BCEWithLogitsLoss combines sigmoid + BCE for stability), Optimizer, Scheduler
    model = SimpleMLP(in_features=2, hidden_dim=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Decays the learning rate by a factor of 0.5 every 3 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # 3. Instantiate Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        max_grad_norm=1.0  # limit gradient norms to 1.0
    )
    
    # 4. Run training for a few epochs
    epochs = 10
    print(f"\nTraining for {epochs} epochs on synthetic circle-bound dataset:")
    for epoch in range(1, epochs + 1):
        # Get current learning rate from optimizer group
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch:02d} | LR: {current_lr:.5f} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.1f}%")
