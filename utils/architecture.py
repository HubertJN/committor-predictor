import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch_geometric.nn import GCNConv, global_mean_pool

def loss_batch(model, loss_func, spin_up, spin_down, xb, yb, opt=None):
    xb_all = torch.cat([xb, spin_up, spin_down], dim=0)
    pred_all = model(xb_all)

    n = xb.shape[0]
    pred_data = pred_all[:n]
    pred_phys = pred_all[n:]

    loss = loss_func(pred_data, yb)
    #loss += 0.01 * ((pred_phys[0] - 1).mean()) ** 2
    #loss += 0.01 * (pred_phys[1].mean()) ** 2

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, device="cpu", config=None, save_path=None, save_dir=None, save_interval=10):
    spin_up = torch.full([64, 64], 1.0, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
    spin_down = torch.full([64, 64], -1.0, dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)

    width = max(4, len(str(epochs)))  # determine width for epoch numbers

    # Create save directory if provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        # --- Training ---
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss_val, n = loss_batch(model, loss_func, spin_up, spin_down, xb, yb, opt)
            train_losses.append((loss_val, n))

        train_vals, train_counts = zip(*train_losses)
        train_loss = np.sum(np.multiply(train_vals, train_counts)) / np.sum(train_counts)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_losses = [
                loss_batch(model, loss_func, spin_up, spin_down, xb.to(device), yb.to(device))
                for xb, yb in valid_dl
            ]
            val_vals, val_counts = zip(*val_losses)
            val_loss = np.sum(np.multiply(val_vals, val_counts)) / np.sum(val_counts)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'model_state_dict': model.state_dict().copy(),
                'channels': config.model.channels if config else None,
                'num_cnn_layers': config.model.num_cnn_layers if config else None,
                'num_fc_layers': config.model.num_fc_layers if config else None,
                'epoch': epoch,
                'val_loss': val_loss
            }
            best_epoch = epoch
            
            # Checkpointing: save best model (legacy support)
            if save_path and config:
                torch.save(best_model_state, save_path)
                print(f"Checkpoint saved at epoch {epoch+1} with val_loss {val_loss:.5f}")

        # Periodic saving: save every N epochs if save_dir is provided
        if save_dir and ((epoch+1) % save_interval == 0 or epoch == 0):
            # Save current model
            current_model_path = save_dir / f"model_{epoch+1:04d}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'channels': config.model.channels if config else None,
                'num_cnn_layers': config.model.num_cnn_layers if config else None,
                'num_fc_layers': config.model.num_fc_layers if config else None,
                'epoch': epoch + 1,
                'val_loss': val_loss
            }, current_model_path)
            print(f"Current model saved to {current_model_path}")
            
            # Save best model if we have found one
            if best_model_state is not None:
                best_model_path = save_dir / f"model_{epoch+1:04d}_best_{best_epoch+1:04d}.pth"
                torch.save(best_model_state, best_model_path)
                print(f"Best model saved to {best_model_path}")

        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch {epoch+1:{width}}/{epochs:{width}} - "
              f"Train Loss: {train_loss:.5f} - Validation Loss: {val_loss:.5f} - "
              f"Time: {epoch_time:.2f}s")
        
    total_time = time.perf_counter() - total_start
    print(f"Total training time: {total_time:.2f}s")

class ResidualCNNLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.SiLU(),
            #nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            #nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode="circular")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        """
        res = x.clone() # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x) # (bs, c, h, w)

        # Second conv block
        x = self.block2(x) # (bs, c, h, w)

        # Add back residual
        x = x + res # (bs, c, h, w)

        return x
    
class ResidualLinearLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, channels)
        """
        res = x.clone()  # (bs, channels)

        x = self.block(x)  # (bs, channels)

        x = x + res  # (bs, channels)

        return x

class CNN(nn.Module):
    def __init__(self, channels=16,
                 num_cnn_layers=3, num_fc_layers=2):
        super().__init__()

        self.init_conv = nn.Sequential(nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, padding_mode="circular"), nn.BatchNorm2d(channels), nn.SiLU())

        self.cnn_blocks = nn.ModuleList([
            ResidualCNNLayer(channels) for _ in range(num_cnn_layers)
        ])

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc_blocks = nn.ModuleList([
            ResidualLinearLayer(channels) for _ in range(num_fc_layers)
        ])

        self.out = nn.Linear(channels, 1)

    def forward(self, x):

        x = self.init_conv(x)
        
        for layer in self.cnn_blocks:
            x = layer(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        for fc in self.fc_blocks:
            x = fc(x)

        x = self.out(x)

        return torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, channels=8, num_fc_layers=1):
        super().__init__()

        self.init_fc = nn.Sequential(nn.Linear(2, channels),  nn.SiLU())

        self.fc_blocks = nn.ModuleList([
            ResidualLinearLayer(channels) for _ in range(num_fc_layers)
        ])

        self.out = nn.Linear(channels, 1)

    def forward(self, x):

        x = self.init_fc(x)

        for fc in self.fc_blocks:
            x = fc(x)

        x = self.out(x)

        return torch.sigmoid(x)

class GNN(nn.Module):
    """
    Simple GCN for 2D Ising grids.
    Treat each spin as a node; connect to 4 nearest neighbors (up/down/left/right).
    """
    def __init__(self, channels=32, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(1, channels)
        self.conv2 = GCNConv(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, 1)

    def forward(self, data_batch):
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch
        x = x + nn.SiLU()(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = x + nn.SiLU()(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))
