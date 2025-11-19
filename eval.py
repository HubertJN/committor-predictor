import torch
import numpy as np
import matplotlib.pyplot as plt
from modules.architecture import CNN
from modules.dataset import prepare_subset, prepare_datasets
from modules.config import load_config
from scipy.ndimage import label
import matplotlib.gridspec as gridspec

# --- Load Config ---
config = load_config("config.yaml")

# --- Model type ---
model_type = config.model.type.lower()  # get from config

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Paths ---
h5path = config.paths.data
save_path = f"{config.paths.save_dir}/{model_type}.pth"

# --- Load dataset ---
grids, attrs, train_idx, valid_idx = prepare_subset(
    h5path,
    test_size=config.dataset.test_size
)
train_dl, valid_dl, train_ds, valid_ds = prepare_datasets(
    grids, attrs, train_idx, valid_idx,
    model_type, device,
    config.dataset.batch_size,
    augment=config.dataset.augment
)

# --- Load trained CNN model ---
checkpoint = torch.load("models/cnn_ch64_cn5_fc3.pth", map_location=device)

# Extract hyperparameters from checkpoint
channels = checkpoint['channels']
num_cnn_layers = checkpoint['num_cnn_layers']
num_fc_layers = checkpoint['num_fc_layers']

# Recreate the model with the same architecture
model = CNN(
    input_size=config.model.input_size,
    channels=channels,
    num_cnn_layers=num_cnn_layers,
    num_fc_layers=num_fc_layers,
    dropout=config.model.dropout
).to(device)

# Load saved state_dict
model.load_state_dict(checkpoint['model_state_dict'])
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

print("============================================")
print(f"Accuracy for validation set : {accuracy:.2f} %")
print("Number of parameters :", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("RMSE: ", rmse)
print("============================================")

# --- Scatter plot ---
plt.figure(figsize=(5, 5))
plt.scatter(y_valid, predictions, s=0.5)
plt.plot([0, 1], [0, 1], c="red")
plt.xlim(-0.1, 1.1)
plt.xlabel("Target")
plt.ylabel("Prediction")
plt.text(
    0.05, 0.95, f"RMSE = {rmse:.4f}",
    transform=plt.gca().transAxes,
    verticalalignment='top',
    horizontalalignment='left',
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.savefig(f"{config.paths.plot_dir}/valid_target_prediction.pdf")
print(f"Saved plot to {config.paths.plot_dir}/valid_target_prediction.pdf")

# --- Spatial invariance test with Mirror Invariance ---
with torch.no_grad():
    # Compute TI (translational invariance), RI (rotational), and MI (mirror) across validation set
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

    # --- Single random example visualization ---
    iimg = np.random.randint(len(valid_ds))
    base_img, _ = valid_ds[iimg]

    # Center the largest cluster before testing
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

    # Example transformations for plot
    out_orig = model(base_img).squeeze().item()
    shifted_img = torch.roll(base_img, shifts=(0, 16), dims=(2, 3))
    rotated_img = torch.rot90(base_img, k=1, dims=(2, 3))
    mirrored_img = torch.flip(base_img, dims=[2])  # example mirror for plot
    out_shift = model(shifted_img).squeeze().item()
    out_rot = model(rotated_img).squeeze().item()
    out_mirror = model(mirrored_img).squeeze().item()

    # --- Plot ---
    images = [base_img, shifted_img, rotated_img, mirrored_img]
    outputs = [out_orig, out_shift, out_rot, out_mirror]
    titles = ["Original (Centered)", "Translated", "Rotated", "Mirrored"]

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
        ax.scatter(up_coords[:,1], up_coords[:,0], marker='x', color='black', s=30, linewidth=1)
        diff = outputs[i] - out_orig
        ax.set_title(titles[i])
        ax.set_xlabel(f"Pred: {outputs[i]:.3f}   |   Diff: {diff:+.3f}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    fig.supxlabel(
        f"TI = {TI:.4f}   |   RI = {RI:.4f}   |   MI = {MI:.4f}   |   Mean Committor Error = {mean_committor_error:.4f}",
        fontsize=12, x=0.5, ha="center"
    )

    plt.savefig(f"{config.paths.plot_dir}/valid_invariance_test.pdf", bbox_inches="tight")
    plt.close()
    print("Saved invariance test plot to valid_invariance_test.pdf")

