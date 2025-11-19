import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Load results
df = pd.read_csv("models/parameter_search.csv")

# Normalize color scale across all heatmaps
vmin, vmax = df["rmse"].min(), df["rmse"].max()

# Unique values
channels = sorted(df["channels"].unique())
cnn_layers = sorted(df["cnn_layers"].unique())
fc_layers = sorted(df["fc_layers"].unique())

ncols = 4
nrows = int(np.ceil(len(channels) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

for ax, ch in zip(axes.flatten(), channels):
    sub = df[df["channels"] == ch]

    # Build pivot matrix for imshow (cnn_layers × fc_layers)
    pivot = np.full((len(cnn_layers), len(fc_layers)), np.nan)
    for _, row in sub.iterrows():
        i = cnn_layers.index(row["cnn_layers"])
        j = fc_layers.index(row["fc_layers"])
        pivot[i, j] = row["rmse"]

    im = ax.imshow(pivot, cmap="viridis_r", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")

    # Find minimum RMSE
    min_idx = np.unravel_index(np.nanargmin(pivot), pivot.shape)
    min_val = pivot[min_idx]

    # Draw black rectangle around the cell of minimum RMSE with smaller linewidth
    rect = Rectangle(
        (min_idx[1] - 0.5, min_idx[0] - 0.5),
        1,  # width
        1,  # height
        linewidth=1.5,  # slightly smaller
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(rect)

    # Add text inside the cell, smaller font and firebrick red
    ax.text(
        min_idx[1], min_idx[0],
        f"{min_val:.3f}",
        color="firebrick",
        ha="center",
        va="center",
        fontsize=7,  # slightly smaller
        fontweight="bold"
    )

    ax.set_title(f"Channels={ch}")
    ax.set_xlabel("FC Layers")
    ax.set_ylabel("CNN Layers")
    ax.set_xticks(range(len(fc_layers)))
    ax.set_xticklabels(fc_layers)
    ax.set_yticks(range(len(cnn_layers)))
    ax.set_yticklabels(cnn_layers)

# Remove unused subplots
for ax in axes.flatten()[len(channels):]:
    ax.axis("off")

# Shared colorbar
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label="RMSE")

plt.suptitle("Heatgrid of RMSE Across Hyperparameters", fontsize=14)
plt.tight_layout(rect=[0, 0, 0.9, 0.95])

plt.savefig("figures/parameter_search.pdf")
plt.close()
