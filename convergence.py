#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from datetime import datetime
from modules.architecture import CNN, fit, physics_func
from modules.dataset import prepare_subset, prepare_datasets
from modules.config import load_config

config = load_config("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(config.paths.save_dir, exist_ok=True)

grids, attrs, train_idx, valid_idx = prepare_subset(config.paths.data, test_size=config.dataset.test_size)
np.randoms.seed(42)

# loss
if config.training.loss.lower() == "smoothl1loss":
    loss_func = torch.nn.SmoothL1Loss()
elif config.training.loss.lower() == "mse":
    loss_func = torch.nn.MSELoss()
else:
    raise ValueError(f"Unsupported loss: {config.training.loss}")

fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
results = []

for frac in fractions:
    n = max(1, int(round(len(train_idx) * frac)))
    subset_train_idx = np.random.choice(train_idx, size=n, replace=False).tolist()
    train_dl, valid_dl, _, _ = prepare_datasets(
        grids, attrs, subset_train_idx, valid_idx,
        config.model.type, device,
        config.dataset.batch_size,
        augment=config.dataset.augment
    )

    # model + optimizer
    model = CNN(
        input_size=config.model.input_size,
        channels=config.model.channels,
        num_cnn_layers=config.model.num_cnn_layers,
        num_fc_layers=config.model.num_fc_layers,
        dropout=config.model.dropout
    ).to(device)

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

    print(f"Train frac={frac:.2f} ({n} samples) -> training...")
    fit(config.training.epochs, model, loss_func, physics_func, optimizer, train_dl, valid_dl, device)

    # --- RMSE on validation set ---
    model.eval()
    y_valid = np.array([valid_ds[i][1].item() for i in range(len(valid_ds))])
    x_valid = torch.stack([valid_ds[i][0] for i in range(len(valid_ds))]).to(device)
    with torch.no_grad():
        predictions = model(x_valid).squeeze().cpu().numpy()
    rmse = np.sqrt(np.mean((predictions - y_valid) ** 2))
    print(f"Fraction {frac:.2f}, train size {n}, RMSE: {rmse:.6f}")

    # save model checkpoint
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    uid = np.random.randint(0, 1e6)
    fname = f"{config.model.type}_frac{int(frac*100)}_{ts}_{uid}.pth"
    save_path = os.path.join(config.paths.save_dir, fname)
    torch.save({'model_state_dict': model.state_dict(), 'fraction': frac, 'rmse': rmse}, save_path)

    # record results for NumPy array
    results.append([n, rmse])
    results_np = np.array(results)
    np.save(os.path.join(config.paths.save_dir, f"{config.model.type}_rmse_vs_train.npy"), results_np)
    print("Saved RMSE results to NumPy array:", results_np)
