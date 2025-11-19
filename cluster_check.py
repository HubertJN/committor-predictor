import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
from modules.config import load_config
from modules.dataset import load_hdf5_raw

# --- parse optional inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Override beta value")
parser.add_argument("--h", type=float, help="Override h value")
args = parser.parse_args()

# --- load config ---
config = load_config("config.yaml")

# --- apply overrides ---
beta = args.beta if args.beta is not None else config.parameters.beta
h = args.h if args.h is not None else config.parameters.h

# --- load dataset ---
training_path = f"../data/gridstates_training_{beta:.3f}_{h:.3f}.hdf5"
grids, attrs, headers = load_hdf5_raw(training_path)

print(f"Loaded {training_path}")

# --- Extract committor values ---
committor = attrs[:, 2]
committor_error = attrs[:,3]
print("Mean committor value:", np.mean(committor))

cluster = attrs[:,1]
idx = ~np.isnan(committor)
cluster = cluster[idx]
committor = committor[idx]
committor_error = committor_error[idx]

# --- select committor ~ 1 ---
tol = 1e-6
mask = np.abs(committor - 1.0) < tol

clusters_1 = cluster[mask]
grids_1 = grids[mask]

# indices of 25 smallest clusters
idx_25 = np.argsort(clusters_1)[:25]
sel_grids = grids_1[idx_25]

# --- plot 5x5 grid ---
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.flatten()

for ax, g in zip(axes, sel_grids):
    up_coords = np.argwhere(g > 0)
    ax.scatter(up_coords[:,1], up_coords[:,0], marker='x', color='black', s=5, linewidth=0.5)
    ax.axis('off')

plt.tight_layout()
plt.savefig("figures/cluster_check.pdf")

