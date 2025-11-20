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

# --- Load trained model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    print(f"  Fast batched inference (len={Tlen}) took {t1 - t0:.2f} s")

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

    print(f"\nBuilding T-matrix for m={m}")
    t_start = time.perf_counter()

    for gid, grids in traj_dict.items():
        print(f"Processing GID={gid}, length={len(grids)}")

        # 1) Compute committors in batches
        q_traj = compute_committors_for_trajectory(grids, model, device)

        # 2) Build transitions
        Tlen = len(q_traj)
        for i in range(0, Tlen - m, m):
            j = i + m
            s_i = bin_state(q_traj[i])
            s_j = bin_state(q_traj[j])
            C_matrix[s_i, s_j] += 1

    # 3) Normalize (avoid zero rows)
    row_sums = C_matrix.sum(axis=1, keepdims=True)
    T_matrix = np.zeros_like(C_matrix, dtype=np.float32)
    mask = row_sums[:, 0] > 0
    T_matrix[mask] = C_matrix[mask] / row_sums[mask]

    if device == "cuda":
        torch.cuda.synchronize()
    print(f"T matrix for m={m} built in {time.perf_counter()-t_start:.2f} s")

    return T_matrix

def implied_timescales(T, tau):
    eigvals, _ = np.linalg.eig(T)
    eigvals = np.real(eigvals)
    idx = np.argsort(-eigvals)        # descending
    eigvals = eigvals[idx]
    nontrivial = eigvals[1:]          # skip λ=1
    its = -tau / np.log(nontrivial)
    return its

def grids_to_process(num_g):
    grid_ids = attrs[:, 4].astype(int)
    cluster_sizes = attrs[:, 1]

    # 1. Find the index of the absolute maximum cluster size
    imax = np.argmax(cluster_sizes)

    # 2. Extract the grid ID that has the largest cluster
    gid_global_max = grid_ids[imax]

    # 3. Randomly choose the remaining grid IDs (excluding this one)
    unique_gids = np.unique(grid_ids)
    remaining = unique_gids[unique_gids != gid_global_max]

    if num_g > 1:
        random_gids = np.random.choice(remaining, size=num_g-1, replace=False)
        selected_gids = np.concatenate(([gid_global_max], random_gids))
    else:
        selected_gids = np.array([gid_global_max])

    print(f"Selected {num_g} grid IDs (random + guaranteed max cluster):")
    print(f"  grid_id={gid_global_max}, max_cluster={cluster_sizes[imax]}")

    for gid in selected_gids[1:]:
        print(f"  grid_id={gid}")

    selected_gids = np.sort(selected_gids)

    return grid_ids, selected_gids

def _load_single_gid(args):
    """
    Worker function executed in its own process.
    Uses the user's load_hdf5_raw function.
    """
    h5path, gid, grid_ids = args
    import numpy as np
    from modules.dataset import load_hdf5_raw  # must be imported inside worker

    # Find indices for this GID
    idx = np.where(grid_ids == gid)[0]
    idx.sort()

    # Use your existing loader
    grids, attrs, _ = load_hdf5_raw(h5path, indices=idx)

    return gid, grids, attrs

def preload_all_trajectories(h5path, grid_ids, top_gids, n_workers=8):
    """
    Parallelized preload using up to n_workers processes.
    Each GID loaded independently → ideal for interleaved datasets.
    """
    args_list = [(h5path, int(gid), grid_ids) for gid in top_gids]

    traj_dict = {}

    with Pool(processes=n_workers) as pool:
        for gid, grids, attrs in pool.map(_load_single_gid, args_list):
            traj_dict[gid] = grids
            #print(f"Preloaded GID={gid}, length={len(grids)}")

    return traj_dict

# --- Committor boundaries ---
q_A = 0.02
q_B = 0.98

num_steps = 12
q_bins = np.linspace(q_A, q_B, num_steps)
N = len(q_bins) - 1
S = N + 2

m_list = [2]
its_list = []
T_dict = {}
num_g = 1

grid_ids, top_gids = grids_to_process(num_g)
traj_dict = preload_all_trajectories(h5path, grid_ids, top_gids, n_workers=8)

for m in m_list:
    tau = m * 1.0
    T_m = build_T_matrix_for_lag(m, traj_dict)
    T_dict[m] = T_m

np.savez("T_matrices.npz", **{f"T_m{m}": T_dict[m] for m in m_list})

# ----------------- TIMING: total -----------------
if device == "cuda":
    torch.cuda.synchronize()
t_total_end = time.perf_counter()
print(f"\nTotal runtime: {t_total_end - t_total_start:.2f} s")
