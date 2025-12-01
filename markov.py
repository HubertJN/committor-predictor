import time
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from modules.config import load_config
from modules.architecture import CNN
from modules.dataset import load_hdf5_raw
from multiprocessing import Pool
import gasp
np.set_printoptions(linewidth=np.inf)

# ----------------- TIMING: total -----------------
t_total_start = time.perf_counter()

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
h5path = f"../data/markov_{beta:.3f}_{h:.3f}.hdf5"

_, attrs, headers = load_hdf5_raw(h5path, load_grids=False)

L = headers["L"]
gpu_nsms = gasp.gpu_nsms
ngrids = 4 * gpu_nsms * 32

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

@torch.no_grad()
def compute_committors_for_trajectory(grids, model, device, batch_size=1024):
    """
    Fast streaming inference for large trajectories that cannot fit in GPU memory at once.
    Avoids costly torch.tensor construction inside the loop.
    """
    Tlen = len(grids)
    q = np.empty(Tlen, dtype=np.float64)

    # Preallocate CPU buffer (pinned memory for fast transfer)
    # shape = (batch_size, 1, L, L)
    L = grids.shape[1]
    cpu_batch = torch.empty((batch_size, 1, L, L),
                            dtype=torch.float32,
                            pin_memory=True)

    t0 = time.perf_counter()

    for start in range(0, Tlen, batch_size):
        end = min(start + batch_size, Tlen)
        n = end - start

        # Copy numpy → CPU pinned tensor (fast)
        # NOTE: slicing avoids reallocation
        cpu_batch[:n, 0].copy_(torch.from_numpy(grids[start:end]))

        # Async transfer CPU→GPU (fast with pin_memory)
        gpu_batch = cpu_batch[:n].to(device, non_blocking=True)

        # Model forward
        out = model(gpu_batch)

        # Move output back to CPU
        q[start:end] = out.squeeze().cpu().numpy()

    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    print(f"  Fast batched inference took {t1 - t0:.2f} s")

    return q

def bin_state(q):
    if q < q_A:
        return 0          # A state
    elif q > q_B:
        return N + 1      # B state
    else:
        k = np.digitize(q, q_bins)  # 1..N
        return k

def build_T_matrix_for_lag(m, traj_dict):
    C_matrix = np.zeros((S, S), dtype=np.int32)

    for b, frames in traj_dict.items():

        # ----------------------------------------------------
        # 1. Compute committor for the initial frame
        # ----------------------------------------------------
        s_i = b

        # ----------------------------------------------------
        # 2. Compute committors for ALL generated frames
        # ----------------------------------------------------
        # frames shape: (ngrids, L, L)
        q_dest = compute_committors_for_trajectory(frames, model, device)   # shape (ngrids,)

        # ----------------------------------------------------
        # 3. Count transitions for each independent m-step
        # ----------------------------------------------------
        for q_j in q_dest:
            s_j = bin_state(q_j)
            C_matrix[s_i, s_j] += 1

    # --------------------------------------------------------
    # 4. Normalize → transition matrix
    # --------------------------------------------------------
    print(C_matrix)
    row_sums = C_matrix.sum(axis=1, keepdims=True)
    T = np.zeros_like(C_matrix, dtype=float)

    mask = row_sums[:, 0] > 0
    T[mask] = C_matrix[mask] / row_sums[mask]

    return T

# Extract needed arrays
grid_ids_all = attrs[:, 4].astype(int)
cluster_sizes_all = attrs[:, 1]

# Sort all frames by cluster size (largest first)
sorted_idx = np.argsort(cluster_sizes_all)[::-1]

topX_gids = []
seen = set()

for idx in sorted_idx:
    gid = grid_ids_all[idx]
    if gid not in seen:
        topX_gids.append(gid)
        seen.add(gid)
    if len(topX_gids) == 2:
        break

print("Top X GIDs (by largest cluster sizes):", topX_gids)

# Load the 4 trajectories and concatenate
traj_list = []

for gid in topX_gids:
    idx = np.where(grid_ids_all == gid)[0]
    idx.sort()
    grids_gid, _, _ = load_hdf5_raw(h5path, indices=idx)
    print(f"Loaded trajectory GID={gid}, length={len(grids_gid)}")
    traj_list.append(grids_gid)

# Concatenate into one long trajectory
grids_main = np.concatenate(traj_list, axis=0)
print(f"Total concatenated length = {len(grids_main)}")

# Compute committor values for *all* frames
q_main = compute_committors_for_trajectory(grids_main, model, device)

print("Committor range:", q_main.min(), q_main.max())

# --- Committor boundaries ---
q_A = 0.02
q_B = 0.98

num_steps = 12
q_bins = np.linspace(q_A, q_B, num_steps)
N = len(q_bins) - 1
bin_ids = np.array([bin_state(q) for q in q_main])
S = N + 2

sampled_frames = {}

for b in range(0, S):   # interior bins only
    indices = np.where(bin_ids == b)[0]
    if len(indices) == 0:
        print(f"Bin {b}: no frames available.")
        continue
    sampled_frames[b] = np.random.choice(indices)
    print(f"Bin {b}: sampled frame index {sampled_frames[b]}")

m_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
n_repeats = 8
T_dict = {}

for m in m_list:
    print(f"\n=== Processing lag m={m} ===")

    generated_trajs = {}

    for b in range(S):

        # ----------------------------------------------
        # Frames inside bin b
        # ----------------------------------------------
        frames_in_bin = np.where(bin_ids == b)[0]
        if len(frames_in_bin) == 0:
            print(f"Bin {b}: no frames available, skipping.")
            continue

        # ----------------------------------------------
        # Accumulate repeated independent short trajectories
        # ----------------------------------------------
        traj_list = []     # will store n_repeats × (ngrids, L, L)

        for r in range(n_repeats):
            print(f"  Bin {b}: repeat {r+1}/{n_repeats}")

            # ---- Draw starting grids for this repeat ----
            if len(frames_in_bin) >= gpu_nsms:
                # Enough frames → sample without replacement
                chosen = np.random.choice(frames_in_bin, size=gpu_nsms, replace=False)
            else:
                # Not enough frames → sample with replacement
                chosen = np.random.choice(frames_in_bin, size=gpu_nsms, replace=True)

            gridlist = [grids_main[i].copy() for i in chosen]

            # ---- Run GASP m steps ----
            gasp.run_committor_calc(
                L, ngrids, m+1, beta, h,
                grid_output_int=m,
                mag_output_int=m,
                grid_input="NumPy",
                grid_array=gridlist,
                keep_grids=True,
                up_threshold=1.01,
                dn_threshold=-1.01,
                nsms=gpu_nsms,
                gpu_method=2,
                outname="None",
                max_keep_grids= 2*ngrids
            )

            # ---- Collect ngrids destination frames ----
            traj = np.zeros((ngrids, L, L), dtype=np.int8)
            for i in range(ngrids):
                traj[i] = gasp.grids[-1][i].grid
            print("isweep of saved trajectory", gasp.grids[-1][0].isweep)   # Sanity check

            traj_list.append(traj)

        # ----------------------------------------------
        # Concatenate N repeats into one array
        # ----------------------------------------------
        generated_trajs[b] = np.concatenate(traj_list, axis=0)

        total_len = generated_trajs[b].shape[0]
        print(f"  Bin {b}: total trajectories = {total_len}")

    # ----------------------------------------------
    # Build transition matrix for lag m
    # ----------------------------------------------
    T_m = build_T_matrix_for_lag(m, generated_trajs)
    T_dict[m] = T_m

# ----- Save all T matrices -----
np.savez(f"T_matrices_{beta:.3f}_{h:.3f}.npz", **{f"T_m{m}": T_dict[m] for m in m_list})
print(f"\nSaved all T(m) matrices to T_matrices_{beta:.3f}_{h:.3f}.npz")

# ----------------- TIMING: total -----------------
if device == "cuda":
    torch.cuda.synchronize()
t_total_end = time.perf_counter()
print(f"\nTotal runtime: {t_total_end - t_total_start:.2f} s")
