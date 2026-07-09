import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_patterns.model_mlp import SimpleMLP
from pytorch_patterns.trainer_loop import generate_synthetic_data, Trainer

def test_trainer_pipeline():
    train_ds, val_ds = generate_synthetic_data(num_samples=40)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    
    model = SimpleMLP(in_features=2, hidden_dim=4)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        max_grad_norm=1.0
    )
    
    initial_weights = model.pipeline[0].weight.clone()
    
    # Run 1 epoch of training
    loss = trainer.train_epoch(train_loader)
    assert isinstance(loss, float)
    assert loss > 0
    
    # Run 1 epoch of validation
    val_loss = trainer.evaluate(val_loader)
    assert isinstance(val_loss, float)
    assert val_loss > 0
    
    # Check that model weights actually changed (learning worked)
    assert not torch.equal(initial_weights, model.pipeline[0].weight)
    
    # Check that scheduler decayed learning rate
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.05)
