# =======================
# --- Color Palette ---
# =======================
colors = {
    "steel_blue": "#1F77B4",
    "light_steel_blue": "#AEC7E8",
    "orange": "#FF7F0E",
    "light_orange": "#FFBB78",
    "forest_green": "#2CA02C",
    "light_green": "#98DF8A",
    "firebrick_red": "#D62728",
    "soft_red": "#FF9896",
    "lavender": "#9467BD",
    "light_lavender": "#C5B0D5",
    "brown": "#8C564B",
    "tan": "#C49C94",
    "orchid": "#E377C2",
    "light_orchid": "#F7B6D2",
    "gray": "#7F7F7F",
    "light_gray": "#C7C7C7",
    "yellow_green": "#BCBD22",
    "light_yellow_green": "#DBDB8D",
    "turquoise": "#17BECF",
    "light_turquoise": "#9EDAE5"
}

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import label
import matplotlib.gridspec as gridspec
from modules.architecture import CNN
from modules.dataset import prepare_subset, prepare_datasets
from modules.config import load_config

# =======================
# --- Plotting Parameters ---
# =======================
FONT_SIZE = 16  # Global font size for all plots

plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE * 0.9,
    'ytick.labelsize': FONT_SIZE * 0.9,
    'legend.fontsize': FONT_SIZE * 0.9,
    'figure.titlesize': FONT_SIZE * 1.1
})


# --- Load Config ---
config = load_config("config.yaml")

# --- parse optional inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Override beta value")
parser.add_argument("--h", type=float, help="Override h value")
args = parser.parse_args()

# --- apply overrides ---
beta = args.beta if args.beta is not None else config.parameters.beta
h = args.h if args.h is not None else config.parameters.h

device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = config.model.type.lower()

# --- Dataset ---
h5path = f"../data/gridstates_training_{beta:.3f}_{h:.3f}.hdf5"
batch_size = config.dataset.batch_size
test_size = config.dataset.test_size

grids, attrs, train_idx, valid_idx, test_idx = prepare_subset(h5path, test_size=test_size)
train_dl, valid_dl, test_dl, train_ds, valid_ds, test_ds = prepare_datasets(
    grids, attrs, train_idx, valid_idx, test_idx,
    model_type, device,
    batch_size,
    augment=False
)

# --- Load trained model ---
checkpoint_path = f"{config.paths.save_dir}/{config.model.type}_ch{config.model.channels}_cn{config.model.num_cnn_layers}_fc{config.model.num_fc_layers}_{beta:.3f}_{h:.3f}.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
channels = checkpoint['channels']
num_cnn_layers = checkpoint['num_cnn_layers']
num_fc_layers = checkpoint['num_fc_layers']

model = CNN(
    input_size=config.model.input_size,
    channels=channels,
    num_cnn_layers=num_cnn_layers,
    num_fc_layers=num_fc_layers,
    dropout=config.model.dropout
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"{config.model.type.upper()} model loaded.")

# =======================
# --- Sigmoid Fit ---
# =======================
train_cluster_size = attrs[:,1][train_idx]
train_committor = attrs[:,2][train_idx]
train_error = attrs[:,3][train_idx]

def sigmoid(x, k, x0):
    base = 1 / (1 + np.exp(-k * (x - x0)))
    base0 = 1 / (1 + np.exp(k * x0))
    return (base - base0) / (1 - base0 + 1e-10)

try:
    with np.errstate(divide='ignore', invalid='ignore'):
        popt, _ = curve_fit(sigmoid, train_cluster_size, train_committor,
                            p0=[0.1, np.median(train_cluster_size)], maxfev=10000)
except RuntimeError as e:
    print(f"Sigmoid fit failed: {e}, using default parameters")
    popt = [0.1, np.median(train_cluster_size)]
x_fit = np.linspace(train_cluster_size.min(), train_cluster_size.max(), 200)
y_fit = sigmoid(x_fit, *popt)

plt.figure(figsize=(6,6))
plt.errorbar(
    train_cluster_size, train_committor, yerr=train_error, 
    fmt='o', ms=3, alpha=0.5, label="Training Data", ecolor='gray', capsize=2
)
plt.plot(x_fit, y_fit, color=colors['firebrick_red'], linewidth=2, label="Sigmoid Fit", zorder=10)
#plt.vlines(7, 0, 1, linestyle="dashed", color="grey")
plt.xlabel("Largest Cluster Size")
plt.ylabel("Target Committor")
plt.title(f"β = {beta:.3f}, h = {h:.3f}")
#plt.legend()
plt.tight_layout()
plt.xlim(-25, 800)
plt.savefig(f"{config.paths.plot_dir}/sigmoid_cluster.pdf")
plt.close()
print(f"Saved cluster size vs committor plot with error bars to {config.paths.plot_dir}/sigmoid_cluster.pdf")

# =======================
# --- CNN Predictions ---
# =======================
y_train = np.array([train_ds[i][1].item() for i in range(len(train_ds))])
y_train_err = attrs[:,3][train_idx]

# Process in chunks to avoid GPU memory issues
chunk_size = 1024
predictions = []
for i in range(0, len(train_ds), chunk_size):
    end_idx = min(i + chunk_size, len(train_ds))
    batch = torch.stack([train_ds[j][0] for j in range(i, end_idx)]).to(device)
    with torch.no_grad():
        batch_preds = model(batch).squeeze().cpu().numpy()
        if batch_preds.ndim == 0:
            batch_preds = np.array([batch_preds])
        predictions.extend(batch_preds)

predictions = np.array(predictions)
rmse = np.sqrt(np.mean((predictions - y_train)**2))

plt.figure(figsize=(6,6))
plt.errorbar(y_train, predictions, xerr=y_train_err, ms=2, capsize=2, alpha=0.5, fmt='o', ecolor='gray', color=colors["steel_blue"])
plt.plot([0,1],[0,1], color=colors['firebrick_red'], linewidth=2)
plt.xlabel("Target Committor")
plt.ylabel("Predicted Committor")
plt.title(f"β = {beta:.3f}, h = {h:.3f}")
plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.tight_layout()
plt.savefig(f"{config.paths.plot_dir}/predicted_cnn.pdf")
plt.close()
print(f"Saved CNN prediction plot to {config.paths.plot_dir}/predicted_cnn.pdf")

# =======================
# --- Sigmoid Predictions with best/worst highlights ---
# =======================
predictions = sigmoid(attrs[train_idx][:,1], *popt)
rmse = np.sqrt(np.mean((predictions - y_train)**2))
errors = np.abs(predictions - y_train)
worst_idx = np.argmax(errors)
mask = (y_train >= 0.2) & (y_train <= 0.8)
best_idx = np.argmin(errors[mask]) if np.any(mask) else np.argmin(errors)
best_idx = np.arange(len(y_train))[mask][best_idx] if np.any(mask) else best_idx

plt.figure(figsize=(6,6))
plt.errorbar(y_train, predictions, xerr=y_train_err, ms=2, capsize=2, alpha=0.5, fmt='o', ecolor='gray', color=colors["steel_blue"])
plt.plot([0,1],[0,1], color=colors['firebrick_red'], linewidth=2)
plt.scatter(y_train[worst_idx], predictions[worst_idx], s=80, facecolors='none', edgecolors=colors['orange'], linewidths=2, label='Worst', zorder=10)
plt.scatter(y_train[best_idx], predictions[best_idx], s=80, facecolors='none', edgecolors=colors['light_green'], linewidths=2, label='Best', zorder=10)
plt.xlabel("Target Committor")
plt.ylabel("Predicted Committor")
plt.title(f"β = {beta:.3f}, h = {h:.3f}")
plt.text(0.05, 0.95, f"RMSE = {rmse:.4f}", transform=plt.gca().transAxes,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
plt.legend()
plt.tight_layout()
plt.savefig(f"{config.paths.plot_dir}/predicted_cluster.pdf")
plt.close()
print(f"Saved sigmoid prediction plot to {config.paths.plot_dir}/predicted_cluster.pdf")

# =======================
# --- Saliency Plot: Best & Worst Predictions ---
# =======================

def compute_saliency(idx):
    x, _ = train_ds[idx]
    x_grid = x.squeeze(0).cpu().numpy()

    # Center largest cluster
    labeled_array, num_features = label(x_grid > 0)
    if num_features > 0:
        cluster_sizes = [(labeled_array == i+1).sum() for i in range(num_features)]
        largest_idx = np.argmax(cluster_sizes)
        cluster_mask = (labeled_array == largest_idx + 1)
        coords = np.argwhere(cluster_mask)
        center_y, center_x = coords.mean(axis=0)
        shift_y = int(x_grid.shape[0]//2 - center_y)
        shift_x = int(x_grid.shape[1]//2 - center_x)
        x_rolled = np.roll(x_grid, shift=(shift_y, shift_x), axis=(0,1))
    else:
        x_rolled = x_grid

    x_tensor = torch.tensor(x_rolled, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    x_tensor.requires_grad = True

    output = model(x_tensor)
    output.backward()
    grad = x_tensor.grad.squeeze().cpu().numpy()

    grad_abs = np.abs(grad)
    grad_norm = grad_abs / grad_abs.sum()
    uniform_min = 1 / grad_norm.size
    uniform_factor = 1.25 * uniform_min
    alpha = np.zeros_like(grad_norm)
    mask = grad_norm > uniform_factor
    alpha[mask] = 0.1 + 0.9 * (grad_norm[mask] - uniform_factor) / (grad_norm[mask].max() - uniform_factor)

    saliency_overlay = np.zeros((*grad.shape, 4))
    saliency_overlay[..., 0] = 1.0
    saliency_overlay[..., 1] = 0.0
    saliency_overlay[..., 2] = 0.0
    saliency_overlay[..., 3] = alpha

    return x_rolled, saliency_overlay

# Compute saliency for best and worst predictions
best_grid, best_saliency = compute_saliency(best_idx)
worst_grid, worst_saliency = compute_saliency(worst_idx)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12,6))
for ax, idx, grid, saliency, title in zip(
        axes, [best_idx, worst_idx], 
        [best_grid, worst_grid], [best_saliency, worst_saliency], 
        ['Best Prediction', 'Worst Prediction']):
    
    # Plot saliency overlay
    up_coords = np.argwhere(grid > 0)
    ax.scatter(up_coords[:,1], up_coords[:,0], marker='x', color='black', s=25, linewidth=1, zorder=2)
    ax.imshow(saliency, interpolation='bicubic', zorder=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    
    # Compute predictions
    actual = y_train[idx]
    actual_err = attrs[train_idx][idx,3]
    sigmoid_pred = sigmoid(attrs[train_idx][idx,1], *popt)
    cnn_input = torch.stack([train_ds[idx][0]]).to(device)
    with torch.no_grad():
        cnn_pred = model(cnn_input).squeeze().item()
    
    # Prepare table data
    table_data = [
        ["Actual",   f"{actual:6.3f} ±{actual_err:6.3f}"],
        ["Sigmoid",  f"{sigmoid_pred:6.3f}"],
        ["CNN",      f"{cnn_pred:6.3f}"]
    ]

    # Create table
    the_table = ax.table(
        cellText=table_data,
        colWidths=[0.6, 0.5],
        cellLoc='left',
        loc='bottom',
        bbox=[0.15, -0.25, 0.5, 0.2]   # [left, bottom, width, height] in Axes coords
    )

    # Remove all lines (make them transparent)
    for _, cell in the_table.get_celld().items():
        cell.set_edgecolor('none')

    # Optional: tweak text size
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(16)


# Reduce horizontal space between subplots
plt.subplots_adjust(wspace=0.4)  # default ~0.4, smaller brings them closer

plt.savefig(f"{config.paths.plot_dir}/{config.model.type}_saliency.pdf", bbox_inches="tight")
plt.close()
print(f"Saved saliency plot (best & worst) to {config.paths.plot_dir}/{config.model.type}_saliency.pdf")

# Plot only the worst prediction
fig, ax = plt.subplots(1, 1, figsize=(6,6))

# Plot saliency overlay
up_coords = np.argwhere(worst_grid > 0)
ax.scatter(up_coords[:,1], up_coords[:,0], marker='x', color='black', s=25, linewidth=1, zorder=2)
ax.imshow(worst_saliency, interpolation='bicubic', zorder=1)
ax.set_xticks([])
ax.set_yticks([])

# Compute predictions
actual = y_train[worst_idx]
actual_err = attrs[train_idx][worst_idx,3]
sigmoid_pred = sigmoid(attrs[train_idx][worst_idx,1], *popt)
cnn_input = torch.stack([train_ds[worst_idx][0]]).to(device)
with torch.no_grad():
    cnn_pred = model(cnn_input).squeeze().item()

# Remove all lines
for _, cell in the_table.get_celld().items():
    cell.set_edgecolor('none')

plt.savefig(f"{config.paths.plot_dir}/{config.model.type}_worst_saliency.pdf", bbox_inches="tight")
plt.close()
print(f"Saved saliency plot (worst) to {config.paths.plot_dir}/{config.model.type}_worst_saliency.pdf")


# =======================
# --- Spatial Invariance Test ---
# =======================
with torch.no_grad():
    diffs_trans, diffs_rot, diffs_mirror = [], [], []

    for i in range(len(valid_ds)):
        img, _ = valid_ds[i]
        img = img.unsqueeze(0).to(device)
        out_orig = model(img).squeeze().item()

        # Translational invariance: shift
        shifted_img = torch.roll(img, shifts=(0, 16), dims=(2, 3))
        out_shift = model(shifted_img).squeeze().item()
        diffs_trans.append(abs(out_shift - out_orig))

        # Rotational invariance: 90° rotations
        rot_diffs = []
        for k in range(1, 4):
            rotated_img = torch.rot90(img, k=k, dims=(2, 3))
            out_rot = model(rotated_img).squeeze().item()
            rot_diffs.append(abs(out_rot - out_orig))
        diffs_rot.append(np.mean(rot_diffs))

        # Mirror invariance
        mirror_diffs = []
        mirror_diffs.append(abs(model(torch.flip(img, dims=[2])).squeeze().item() - out_orig))  # Horizontal
        mirror_diffs.append(abs(model(torch.flip(img, dims=[3])).squeeze().item() - out_orig))  # Vertical
        mirror_diffs.append(abs(model(img.transpose(2, 3)).squeeze().item() - out_orig))         # Main diagonal
        mirror_diffs.append(abs(model(torch.flip(img, dims=[2,3]).transpose(2,3)).squeeze().item() - out_orig))  # Anti-diagonal
        diffs_mirror.append(np.mean(mirror_diffs))

    TI = float(np.mean(diffs_trans))
    RI = float(np.mean(diffs_rot))
    MI = float(np.mean(diffs_mirror))
    mean_committor_error = float(np.mean(attrs[valid_idx, 3]))

    print(f"Mean Committor Error: {mean_committor_error:.4f}")
    print(f"Translational Invariance (TI): {TI:.4f}")
    print(f"Rotational Invariance (RI): {RI:.4f}")
    print(f"Mirror Invariance (MI): {MI:.4f}")

    # --- Parameters ---
    target_cluster_size = 240
    tolerance = 10
    selected_idx = None

    # --- Find first valid example ---
    for i in range(len(valid_ds)):
        img, _ = valid_ds[i]
        x_grid = img.squeeze(0).detach().cpu().numpy()
        labeled_array, num_features = label(x_grid > 0)
        
        if num_features > 0:
            cluster_sizes = [(labeled_array == j+1).sum() for j in range(num_features)]
            largest_size = max(cluster_sizes)
        else:
            largest_size = 0
        
        if abs(largest_size - target_cluster_size) <= tolerance:
            selected_idx = i
            break

    if selected_idx is None:
        raise ValueError("No grid found within the target cluster size ± tolerance.")

    base_img, _ = valid_ds[selected_idx]

    # Center the largest cluster
    x_grid = base_img.squeeze(0).detach().cpu().numpy()
    labeled_array, num_features = label(x_grid > 0)
    if num_features > 0:
        cluster_sizes = [(labeled_array == i + 1).sum() for i in range(num_features)]
        largest_idx = np.argmax(cluster_sizes)
        cluster_mask = (labeled_array == largest_idx + 1)
        coords = np.argwhere(cluster_mask)
        center_y, center_x = coords.mean(axis=0)
        shift_y = int(x_grid.shape[0] // 2 - center_y)
        shift_x = int(x_grid.shape[1] // 2 - center_x)
        x_centered = np.roll(x_grid, shift=(shift_y, shift_x), axis=(0, 1))
    else:
        x_centered = x_grid

    base_img = torch.tensor(x_centered, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Example transformations
    out_orig = model(base_img).squeeze().item()
    shifted_img = torch.roll(base_img, shifts=(0, 16), dims=(2, 3))
    rotated_img = torch.rot90(base_img, k=1, dims=(2, 3))
    mirrored_img = torch.flip(base_img, dims=[2])
    out_shift = model(shifted_img).squeeze().item()
    out_rot = model(rotated_img).squeeze().item()
    out_mirror = model(mirrored_img).squeeze().item()

    # --- Plot ---
    images = [base_img, shifted_img, rotated_img, mirrored_img]
    outputs = [out_orig, out_shift, out_rot, out_mirror]
    titles = ["Original", "Translated", "Rotated", "Mirrored"]
    #colors_plot = ["steel_blue", "orange", "forest_green", "lavender"]

    n_subplots = len(images)
    subplot_size = 6
    spacing = 0.5
    fig_width = n_subplots * subplot_size + (n_subplots-1) * spacing
    fig_height = subplot_size
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, n_subplots, wspace=spacing/subplot_size)

    for i in range(n_subplots):
        ax = fig.add_subplot(gs[0, i])
        img = images[i].squeeze(0).squeeze(0).cpu()
        up_coords = np.argwhere(img.numpy() > 0)
        ax.scatter(up_coords[:,1], up_coords[:,0], marker='x', color="black", s=30, linewidth=1)
        diff = outputs[i] - out_orig
        ax.set_title(titles[i])
        ax.set_xlabel(f"Pred: {outputs[i]:.3f} | Diff: {diff:+.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    fig.supxlabel(
        f"TI = {TI:.3f}    |    RI = {RI:.3f}    |    MI = {MI:.3f}    |    Mean Committor Error = {mean_committor_error:.3f}",
        x=0.5, ha="center", y=-0.1, fontsize=25
    )

    plt.savefig(f"{config.paths.plot_dir}/valid_invariance_test.pdf", bbox_inches="tight")
    plt.close()
    print("Saved invariance test plot to valid_invariance_test.pdf")

# 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, img, title in zip(axes.flat, images, titles):
    img_np = img.squeeze(0).squeeze(0).cpu().numpy()
    up_coords = np.argwhere(img_np > 0)
    ax.scatter(up_coords[:,1], up_coords[:,0], marker='x', color='black', s=30, linewidth=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f"{config.paths.plot_dir}/invariance_2x2.pdf", bbox_inches="tight")
plt.close()
print("Saved 2x2 invariance test plot to invariance_2x2.pdf")