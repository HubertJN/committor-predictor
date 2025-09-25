import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from modules.architecture import CNN, GNN, fit, physics_func
from modules.dataset import prepare_subset, prepare_datasets
from modules.config import load_config

config = load_config("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"

grids, attrs, train_idx, valid_idx = prepare_subset(
    config.paths.data,
    test_size=config.dataset.test_size
)
train_dl, valid_dl, train_ds, valid_ds = prepare_datasets(
    grids, attrs, train_idx, valid_idx,
    config.model.type, device,
    config.dataset.batch_size,
    augment=config.dataset.augment
)

if config.model.type == "cnn":
    model = CNN(
        input_size=config.model.input_size,
        channels=config.model.channels,
        num_cnn_layers=config.model.num_cnn_layers,
        num_fc_layers=config.model.num_fc_layers,
        dropout=config.model.dropout
    ).to(device)
else:
    model = GNN().to(device)

decay, no_decay = [], []
for name, param in model.named_parameters():
    if "bn" in name or "bias" in name:
        no_decay.append(param)
    else:
        decay.append(param)

optimizer = torch.optim.AdamW([
    {'params': decay, 'weight_decay': config.training.weight_decay},
    {'params': no_decay, 'weight_decay': 0.0}
], lr=config.training.lr)

if config.training.loss.lower() == "smoothl1loss":
    loss_func = torch.nn.SmoothL1Loss()
elif config.training.loss.lower() == "mse":
    loss_func = torch.nn.MSELoss()
else:
    raise ValueError(f"Unsupported loss: {config.training.loss}")

print(f"Starting training for {config.model.type.upper()}...")
fit(config.training.epochs, model, loss_func, physics_func, optimizer,
    train_dl, valid_dl, device)
print("Training complete.")

save_path = f"{config.paths.save_dir}/{config.model.type}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")
