"""nucleation_error_plot.py

For each snapshot saved by nucleation_error_test.py:
  1. Copy it to the model path that markov.py expects
  2. Run markov.py  (builds C-matrix via CNN RC)
  3. Run markov_analyse.py  (computes nucleation rate from C-matrix)
  4. Collect (val_loss, J_MSM)

Then load the brute-force nucleation rate and plot:
    residual = |J_MSM - J_brute| / J_brute
vs
    val_loss  (x-axis runs HIGH → LOW, i.e. left = overfit/early, right = converged)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.config import load_config

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--beta",    type=float, help="Override beta (default: from config)")
parser.add_argument("--h",       type=float, help="Override h    (default: from config)")
parser.add_argument("--out",     type=str,   default="figures/nucleation_error.pdf",
                    help="Output figure path")
parser.add_argument("--no-markov", action="store_true",
                    help="Skip running markov.py / markov_analyse.py and use already-saved results")
args = parser.parse_args()

config = load_config("config.yaml")
beta = args.beta if args.beta is not None else float(config.parameters.beta)
h    = args.h    if args.h    is not None else float(config.parameters.h)

# ---------------------------------------------------------------------------
# Locate snapshots
# ---------------------------------------------------------------------------
snap_dir = Path("models") / f"{beta:.3f}_{h:.3f}"

if not snap_dir.exists():
    sys.exit(f"Model directory not found: {snap_dir}")

# Find all model_XXXX.pth files (excluding _best_ files)
snapshots = sorted([p for p in snap_dir.glob("model_*.pth") if "_best_" not in p.name],
                   key=lambda p: int(p.stem.split("_")[1]))

if not snapshots:
    sys.exit(f"No model snapshots found in {snap_dir}")

print(f"Found {len(snapshots)} model snapshots for beta={beta:.3f}, h={h:.3f}")

# Per-snapshot output directory (keeps all intermediate files, never touches main model)
ne_data_dir = Path("data") / "nucleation_error"
ne_data_dir.mkdir(parents=True, exist_ok=True)

def c_matrix_path(snap_idx: int) -> Path:
    return ne_data_dir / f"C_matrices_{beta:.3f}_{h:.3f}_cnn_{snap_idx}.npz"

def msm_out_path(snap_idx: int) -> Path:
    return ne_data_dir / f"msm_analysis_{beta:.3f}_{h:.3f}_cnn_{snap_idx}.npz"

# ---------------------------------------------------------------------------
# Brute-force reference rate
# ---------------------------------------------------------------------------
brute_path = Path("data") / f"nucleation_{beta:.3f}_{h:.3f}.npz"
if not brute_path.exists():
    sys.exit(f"Brute-force file not found: {brute_path}")

brute_data = np.load(str(brute_path))
J_brute = float(np.atleast_1d(brute_data["rate_per_site"])[0])
print(f"Brute-force rate: {J_brute:.4e}")

# ---------------------------------------------------------------------------
# Loop over snapshots
# ---------------------------------------------------------------------------
results: list[dict] = []  # {val_loss, snapshot_idx, J_msm}

for snap_path in snapshots:
    import torch
    meta = torch.load(str(snap_path), map_location="cpu", weights_only=False)
    val_loss    = float(meta["val_loss"])
    snap_idx    = int(snap_path.stem.split("_")[1])  # Extract epoch from filename (model_XXXX.pth)
    print(f"\n--- Epoch {snap_idx}  val_loss={val_loss:.6f}  ({snap_path.name}) ---")

    if not args.no_markov:
        # 1. Run markov.py – pass the snapshot path and a per-snapshot C-matrix output
        cmd_markov = [
            sys.executable, "markov.py",
            "--beta",         f"{beta}",
            "--h",            f"{h}",
            "--rc",           "cnn",
            "--no-scan",
            "--model-path",   str(snap_path),
            "--c-matrix-out", str(c_matrix_path(snap_idx)),
        ]
        print(f"  Running: {' '.join(cmd_markov)}")
        ret = subprocess.run(cmd_markov, capture_output=False)
        if ret.returncode != 0:
            print(f"  WARNING: markov.py exited with code {ret.returncode}, skipping snapshot.")
            continue

        # 2. Run markov_analyse.py – pass the per-snapshot C-matrix and output path
        cmd_analyse = [
            sys.executable, "markov_analyse.py",
            "--beta",          f"{beta}",
            "--h",             f"{h}",
            "--rc",            "cnn",
            "--c-matrix-in",   str(c_matrix_path(snap_idx)),
            "--msm-out",       str(msm_out_path(snap_idx)),
        ]
        print(f"  Running: {' '.join(cmd_analyse)}")
        ret = subprocess.run(cmd_analyse, capture_output=False)
        if ret.returncode != 0:
            print(f"  WARNING: markov_analyse.py exited with code {ret.returncode}, skipping snapshot.")
            continue

    # 3. Read MSM result
    if not msm_out_path(snap_idx).exists():
        print(f"  WARNING: {msm_out_path(snap_idx)} not found, skipping.")
        continue

    msm_data = np.load(str(msm_out_path(snap_idx)))
    J_msm = float(msm_data["J_central"][0])
    print(f"  J_MSM = {J_msm:.4e}   J_brute = {J_brute:.4e}")

    results.append({
        "snapshot_idx": snap_idx,
        "val_loss":     val_loss,
        "J_msm":        J_msm,
    })

if not results:
    sys.exit("No results collected — nothing to plot.")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
val_losses = np.array([r["val_loss"]     for r in results])
J_msm_arr  = np.array([r["J_msm"]        for r in results])
residuals  = np.abs(J_msm_arr - J_brute) / np.abs(J_brute)
snap_idxs  = [r["snapshot_idx"] for r in results]

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(val_losses, residuals, zorder=3, color="steelblue", s=60, label="CNN snapshot")

# Label each point with its snapshot index
#for vl, res, idx in zip(val_losses, residuals, snap_idxs):
#    ax.annotate(f"s{idx}", (vl, res), textcoords="offset points",
#                xytext=(5, 4), fontsize=8, color="steelblue")

# Reference line at 0 residual
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

# x-axis: high → low (worst model on left, best on right)
ax.invert_xaxis()

ax.set_xlabel("Validation Loss", fontsize=12)
ax.set_ylabel("Percentage Residual to Brute-Force Rate", fontsize=12)
#ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.yscale("log")
plt.xscale("log")

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out_path), bbox_inches="tight")
print(f"\nFigure saved to {out_path}")
#plt.show()
