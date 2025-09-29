import os
import sys

import torch
import numpy as np

from modules.architecture import CNN, fit, physics_func
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

if config.model.type.lower() != "cnn":
    raise RuntimeError("This hyperparameter script only supports config.model.type == 'cnn'")

# hyperparameter grid
channels_list = [1 << i for i in range(3, 7)]    # 1,2,4,8,16,32,64
cnn_layers_list = list(range(1, 9))              # 1..8
fc_layers_list = list(range(1, 9))               # 1..8

combos = []
for ch in channels_list:
    for nc in cnn_layers_list:
        for nf in fc_layers_list:
            combos.append((ch, nc, nf))

#print(len(combos))

# pick exactly one combo based on SLURM_ARRAY_TASK_ID
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
if task_id < 0 or task_id >= len(combos):
    raise IndexError(f"SLURM_ARRAY_TASK_ID {task_id} out of range (0..{len(combos)-1})")

channels, num_cnn_layers, num_fc_layers = combos[task_id]

print(f"[task {task_id}] Running config index {task_id}: channels={channels}, cnn_layers={num_cnn_layers}, fc_layers={num_fc_layers}")

# loss
if config.training.loss.lower() == "smoothl1loss":
    loss_func = torch.nn.SmoothL1Loss()
elif config.training.loss.lower() == "mse":
    loss_func = torch.nn.MSELoss()
else:
    raise ValueError(f"Unsupported loss: {config.training.loss}")

# instantiate model for this combo
model = CNN(
    input_size=config.model.input_size,
    channels=channels,
    num_cnn_layers=num_cnn_layers,
    num_fc_layers=num_fc_layers,
    dropout=config.model.dropout
).to(device)

# optimizer param groups
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

# train
print(f"Starting training for CNN (ch={channels}, nc={num_cnn_layers}, nf={num_fc_layers})...")
fit(config.training.epochs, model, loss_func, physics_func, optimizer,
    train_dl, valid_dl, device)
print("Training complete.")

model.eval()

# --- Predictions on validation set ---
y_valid = np.array([valid_ds[i][1].item() for i in range(len(valid_ds))])
x_valid = torch.stack([valid_ds[i][0] for i in range(len(valid_ds))]).to(device)

with torch.no_grad():
    predictions = model(x_valid).squeeze().cpu().numpy()

# --- Accuracy and RMSE ---
threshold = 0.05
accuracy = 100 * np.sum(np.abs(predictions - y_valid) < threshold) / len(y_valid)
rmse = np.sqrt(np.mean((predictions - y_valid) ** 2))
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("============================================")
print(f"Accuracy for validation set : {accuracy:.2f} %")
print("Number of parameters :", param_count)
print("RMSE: ", rmse)
print("============================================")

# --- Append a single CSV line atomically to a shared file ---
out_dir = config.paths.save_dir
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, "parameter_search.csv")

header = "channels,cnn_layers,fc_layers,param_count,rmse\n"
line = f"{channels},{num_cnn_layers},{num_fc_layers},{param_count},{rmse:.6g}\n"
bline = line.encode("utf-8")
bheader_and_line = (header + line).encode("utf-8")

# Try to create the file and write header+line atomically if it doesn't exist
if not os.path.exists(csv_path):
    try:
        fd = os.open(csv_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, bheader_and_line)
        finally:
            os.close(fd)
    except FileExistsError:
        # Someone else created it in the meantime — just append our line
        fd = os.open(csv_path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, bline)
        finally:
            os.close(fd)
else:
    fd = os.open(csv_path, os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, bline)
    finally:
        os.close(fd)

print(f"[task {task_id}] Appended results to {csv_path}: {line.strip()}")

# save unique file
os.makedirs(config.paths.save_dir, exist_ok=True)
fname = f"{config.model.type}_ch{channels}_cn{num_cnn_layers}_fc{num_fc_layers}.pth"
save_path = os.path.join(config.paths.save_dir, fname)
torch.save({
    'model_state_dict': model.state_dict(),
    'channels': channels,
    'num_cnn_layers': num_cnn_layers,
    'num_fc_layers': num_fc_layers,
}, save_path)
print(f"Model weights saved to {save_path}")
