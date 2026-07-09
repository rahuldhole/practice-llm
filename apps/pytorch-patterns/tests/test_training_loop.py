import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_patterns.training_loop import SimpleMLP, generate_synthetic_data, Trainer

def test_generate_synthetic_data():
    train_ds, val_ds = generate_synthetic_data(num_samples=100)
    
    assert len(train_ds) == 80
    assert len(val_ds) == 20
    
    x, y = train_ds[0]
    assert x.shape == (2,)
    assert y.shape == (1,)
    assert y.item() in (0.0, 1.0)


def test_trainer_workflow():
    # Mini setup for quick testing
    train_ds, val_ds = generate_synthetic_data(num_samples=50)
    train_loader = DataLoader(train_ds, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=5, shuffle=False)
    
    model = SimpleMLP(in_features=2, hidden_dim=4)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        max_grad_norm=1.0
    )
    
    # Store initial model weights to verify updates
    initial_weights = model.net[0].weight.clone()
    
    # Train 1 epoch
    train_loss, train_acc = trainer.train_epoch(train_loader)
    assert isinstance(train_loss, float)
    assert 0.0 <= train_acc <= 1.0
    
    # Evaluate
    val_loss, val_acc = trainer.evaluate(val_loader)
    assert isinstance(val_loss, float)
    assert 0.0 <= val_acc <= 1.0
    
    # Check that model weights actually changed/updated
    updated_weights = model.net[0].weight
    assert not torch.equal(initial_weights, updated_weights)
    
    # Check learning rate scheduler decayed the rate
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.09)

