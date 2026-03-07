import time
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import math
from pathlib import Path
from utils.config import load_config
from utils.architecture import CNN
from utils.dataset import load_hdf5_raw, uniform_filter
from multiprocessing import Pool
import gasp
from scipy.ndimage import label
np.set_printoptions(linewidth=np.inf)


# =======================
# --- MANUALLY FILL RUNS FOR SWEEP ---
# =======================
# Enable with: python markov.py --sweep
# Example:
# RUNS = [
#     (0.511, 0.040),
#     (0.526, 0.050),
# ]

RUNS: list[tuple[float, float]] = [
    (0.474, 0.01),
    (0.486, 0.02),
    (0.498, 0.03),
    (0.511, 0.04),
]

RUNS: list[tuple[float, float]] = [
    (0.474, 0.01),
    (0.486, 0.02),
    (0.498, 0.03),
    (0.511, 0.040),
    (0.526, 0.050),
    (0.538, 0.060),
    (0.550, 0.070),
    (0.564, 0.080),
    (0.576, 0.090),
    (0.588, 0.100),
]
# =======================
# --- MANUALLY FILL SIGMOID PARAMS FOR CLUSTER RC ---
# =======================
# Enable with: python markov.py --rc cluster
# Fill this with sigmoid parameters (k, x0) used to map
# largest cluster size -> committor estimate, in the SAME ORDER as RUNS.
#
# The sigmoid is:
#   base = 1 / (1 + exp(-k*(x-x0)))
#   base0 = 1 / (1 + exp(k*x0))
#   q = (base-base0) / (1-base0+1e-10)
#
# Example template for the RUNS defined above:
# CLUSTER_SIGMOID_PARAMS = [
#     (0.123, 200.0),  # for RUNS[0]
#     (0.120, 205.0),  # for RUNS[1]
#     ...
# ]
CLUSTER_SIGMOID_PARAMS: list[tuple[float, float]] = [
    (0.0187607845, 215.370086),
    (0.0221898362, 180.460011),
    (0.0257277638, 151.121275),
    (0.0296685290, 129.229575),
    (0.0337087394, 115.113093),
    (0.0382914602, 102.084037),
    (0.0431094616, 91.743354),
]


# =======================
# --- MANUALLY FILL q_A/q_B FOR RAW CLUSTER RC ---
# =======================
# Enable with: python markov.py --rc cluster_raw
# Fill this with (q_A, q_B) bounds in *cluster-size units*, in the SAME ORDER as RUNS.
#
# Interpretation:
#   - state A if cluster_size < q_A
#   - state B if cluster_size > q_B
#   - intermediate bins are built uniformly between [q_A, q_B]
#
# Example template:
# CLUSTER_RAW_QAB = [
#     (10.0, 250.0),  # for RUNS[0]
#     (12.0, 240.0),  # for RUNS[1]
#     ...
# ]
#CLUSTER_RAW_QAB: list[tuple[float, float]] = [
#    (12, 600),
#    (10, 500),
#    (8, 450),
#    (7, 400),
#    (6, 350),
#    (6, 300),
#    (5, 250),
#]

CLUSTER_RAW_QAB: list[tuple[float, float]] = [
    (12, 600),
    (10, 500),
    (8, 450),
    (7, 400),
    (6, 350),
    (6, 300),
    (5, 250),
]

CLUSTER_RAW_QAB: list[tuple[float, float]] = [
    (22, 2500),
    (16, 1500),
    (13, 800),
    (12, 600),
]

# These globals are set inside run_one() so existing helper functions
# can remain mostly unchanged.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
q_bins = None
full_bins = None
N = None
S = None
compute_q_for_frames = None


def sigmoid(x, k, x0):
    base = 1 / (1 + np.exp(-k * (x - x0)))
    base0 = 1 / (1 + np.exp(k * x0))
    return (base - base0) / (1 - base0 + 1e-10)


def _run_index(beta: float, h: float) -> int:
    for i, (b, hh) in enumerate(RUNS):
        if np.isclose(beta, float(b), atol=5e-4) and np.isclose(h, float(hh), atol=5e-4):
            return i
    raise ValueError(
        f"(beta={beta:.3f}, h={h:.3f}) not found in RUNS. "
        "Either add it to RUNS or run with --rc cnn."
    )


def get_cluster_sigmoid_params(beta: float, h: float) -> tuple[float, float]:
    if not CLUSTER_SIGMOID_PARAMS:
        raise ValueError(
            "CLUSTER_SIGMOID_PARAMS is empty. Fill it with (k, x0) in the same order as RUNS."
        )
    if len(CLUSTER_SIGMOID_PARAMS) != len(RUNS):
        raise ValueError(
            f"CLUSTER_SIGMOID_PARAMS length ({len(CLUSTER_SIGMOID_PARAMS)}) must match RUNS length ({len(RUNS)})."
        )
    idx = _run_index(beta, h)
    k, x0 = CLUSTER_SIGMOID_PARAMS[idx]
    return float(k), float(x0)


def get_cluster_raw_qab(beta: float, h: float) -> tuple[float, float]:
    if not CLUSTER_RAW_QAB:
        raise ValueError(
            "CLUSTER_RAW_QAB is empty. Fill it with (q_A, q_B) in the same order as RUNS."
        )
    if len(CLUSTER_RAW_QAB) != len(RUNS):
        raise ValueError(
            f"CLUSTER_RAW_QAB length ({len(CLUSTER_RAW_QAB)}) must match RUNS length ({len(RUNS)})."
        )
    idx = _run_index(beta, h)
    qA, qB = CLUSTER_RAW_QAB[idx]
    qA = float(qA)
    qB = float(qB)
    if not (np.isfinite(qA) and np.isfinite(qB) and qA < qB):
        raise ValueError(f"Invalid (q_A, q_B)=({qA}, {qB}) for beta={beta:.3f}, h={h:.3f}")
    return qA, qB


def _uniform_filter_any(labels: np.ndarray, num_bins: int = 10, seed: int = 42, total_samples: int = 10000) -> np.ndarray:
    """Approx-uniform sampling for arbitrary-valued labels (not assumed in [0,1]).

    Uses quantile-based bin edges to avoid empty bins for skewed distributions.
    """

    rng = np.random.default_rng(seed)
    x = np.asarray(labels, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite labels provided")

    q = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(x, q)
    # ensure edges are non-decreasing; handle duplicates by nudging
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12

    # Map from original labels to indices
    labels_full = np.asarray(labels, dtype=float).ravel()
    idx_all = np.arange(labels_full.size)

    # Build bins
    bin_indices: list[np.ndarray] = []
    for i in range(num_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == num_bins - 1:
            mask = (labels_full >= lo) & (labels_full <= hi)
        else:
            mask = (labels_full >= lo) & (labels_full < hi)
        bin_indices.append(idx_all[mask])

    target_per_bin = max(1, int(total_samples // num_bins))
    chosen: list[int] = []
    used = set()
    for bidx in bin_indices:
        if bidx.size == 0:
            continue
        k = min(target_per_bin, int(bidx.size))
        pick = rng.choice(bidx, size=k, replace=False)
        for p in pick.tolist():
            used.add(int(p))
        chosen.extend(pick.tolist())

    # Top up to total_samples from remaining indices
    if len(chosen) < total_samples:
        remaining = np.array([i for i in idx_all.tolist() if i not in used], dtype=int)
        if remaining.size > 0:
            k2 = min(int(total_samples - len(chosen)), int(remaining.size))
            chosen.extend(rng.choice(remaining, size=k2, replace=False).tolist())

    chosen = np.array(sorted(set(int(i) for i in chosen)), dtype=int)
    print(f"Selected {chosen.size} samples approximately uniformly across {num_bins} quantile bins")
    return chosen


def largest_cluster_size_up(grid_2d: np.ndarray) -> int:
    labeled_array, num_features = label(grid_2d > 0)
    if num_features == 0:
        return 0
    sizes = np.bincount(labeled_array.ravel())
    if sizes.size <= 1:
        return 0
    return int(sizes[1:].max())


def largest_cluster_sizes_up(frames: np.ndarray) -> np.ndarray:
    # frames: (T, L, L)
    out = np.empty(frames.shape[0], dtype=np.float64)
    for i in range(frames.shape[0]):
        out[i] = largest_cluster_size_up(frames[i])
    return out

def evaluate_increasing_ones_grid(
    model: torch.nn.Module,
    L: int,
    device: str,
    scan_bins: int = 32,
):
    """Run the model on grids that start all -1 and gradually flip more sites to 1."""

    scan_bins = max(1, scan_bins)
    total_spins = L * L
    ones_values = np.unique(np.linspace(0, total_spins, scan_bins, dtype=int))
    if ones_values[-1] != total_spins:
        ones_values = np.append(ones_values, total_spins)

    base_flat = -np.ones(total_spins, dtype=np.float32)
    print("\nCommittor predictions for progressively more +1 spins:")
    print("ones | q-value | ascii")
    with torch.no_grad():
        for ones in ones_values:
            grid_flat = base_flat.copy()
            grid_flat[:ones] = 1.0
            grid = grid_flat.reshape(L, L)
            tensor = torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).to(device)
            out = model(tensor)
            value = float(out.squeeze().item())
            bars = min(10, max(0, int(round(value * 10))))
            bar = "#" * bars + "-" * (10 - bars)
            print(f"{ones:4d} | {value:.6f} | {bar}")

@torch.no_grad()
def compute_committors_for_trajectory(grids, model, device, batch_size=1024):
    """
    Fast streaming inference for large trajectories that cannot fit in GPU memory at once.
    Avoids costly torch.tensor construction inside the loop.
    """
    Tlen = len(grids)
    q = np.ones(Tlen, dtype=np.float64)

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

def norm_cdf(x, mu=0.0, sigma=1.0):
    z = (x - mu) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def build_C_matrix_for_lag(m, traj_dict, model_error):
    C_matrix = np.zeros((S, S), dtype=np.float64)

    for b, frames in traj_dict.items():

        # ----------------------------------------------------
        # 1. Compute committor for the initial frame
        # ----------------------------------------------------
        s_i = b

        # ----------------------------------------------------
        # 2. Compute committors for ALL generated frames
        # ----------------------------------------------------
        # frames shape: (ngrids, L, L)
        q_dest = compute_q_for_frames(frames)   # shape (ngrids,)

        # ----------------------------------------------------
        # 3. Count transitions for each independent m-step
        # ----------------------------------------------------
        for q_j in q_dest:
            s_j = bin_state(q_j)
            C_matrix[s_i, s_j] += 1

        for q_j in q_dest:
            for idx in range(len(full_bins) - 1):
                if idx == len(full_bins) - 2:
                    #print(full_bins[idx], q_j, norm_cdf(full_bins[idx], mu=q_j, sigma=model_error))
                    area = 1.0 - norm_cdf(full_bins[idx], mu=q_j, sigma=model_error)
                elif idx == 0:
                    area = norm_cdf(full_bins[idx+1], mu=q_j, sigma=model_error)
                else:
                    area = norm_cdf(full_bins[idx+1], mu=q_j, sigma=model_error) - norm_cdf(full_bins[idx], mu=q_j, sigma=model_error)

                #C_matrix[s_i, idx] += area

    # --------------------------------------------------------
    # 4. Return count matrix
    # --------------------------------------------------------
    print(C_matrix)

    return C_matrix


def run_one(beta: float, h: float, config, scan_bins: int, run_scan: bool, rc: str) -> None:
    global device, model, q_A, q_B, q_bins, full_bins, N, S, compute_q_for_frames

    # ----------------- TIMING: total -----------------
    t_total_start = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    h5path = f"../data/gridstates_training_{beta:.3f}_{h:.3f}.hdf5"

    print("\n" + "=" * 80)
    print(f"Running Markov analysis for beta={beta:.3f}, h={h:.3f}")
    print("=" * 80)

    _, _, headers = load_hdf5_raw(h5path, load_grids=False)
    L = headers["L"]

    gpu_nsms = gasp.gpu_nsms
    ngrids = 4 * gpu_nsms * 32

    if rc == "cnn":
        # --- Load trained model ---
        checkpoint_path = (
            f"{config.paths.save_dir}/{config.model.type}_ch{config.model.channels}_"
            f"cn{config.model.num_cnn_layers}_fc{config.model.num_fc_layers}_{beta:.3f}_{h:.3f}.pth"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        channels = checkpoint["channels"]
        num_cnn_layers = checkpoint["num_cnn_layers"]
        num_fc_layers = checkpoint["num_fc_layers"]

        model = CNN(
            channels=channels,
            num_cnn_layers=num_cnn_layers,
            num_fc_layers=num_fc_layers,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"{config.model.type.upper()} model loaded.")

        if run_scan:
            evaluate_increasing_ones_grid(model, L, device, scan_bins=scan_bins)

        compute_q_for_frames = lambda frames: compute_committors_for_trajectory(frames, model, device)
    elif rc == "cluster":
        k, x0 = get_cluster_sigmoid_params(beta, h)
        print(f"Using cluster RC with sigmoid params: k={k:.6g}, x0={x0:.6g}")

        def _q_from_frames(frames: np.ndarray) -> np.ndarray:
            cs = largest_cluster_sizes_up(frames)
            q = sigmoid(cs, k, x0)
            return np.clip(q, 0.0, 1.0)

        compute_q_for_frames = _q_from_frames
    elif rc == "cluster_raw":
        qA, qB = get_cluster_raw_qab(beta, h)
        print(f"Using raw cluster RC with (q_A, q_B)=({qA:.6g}, {qB:.6g})")

        def _q_from_frames(frames: np.ndarray) -> np.ndarray:
            return largest_cluster_sizes_up(frames)

        compute_q_for_frames = _q_from_frames
    else:
        raise ValueError("rc must be 'cnn', 'cluster', or 'cluster_raw'")

    # Load attributes
    _, attrs_all, _ = load_hdf5_raw(h5path, load_grids=False)

    # Apply uniform filter to select samples.
    # For CNN MSMs, we balance across committor labels in [0,1].
    # For cluster-based MSMs, we balance across cluster sizes (not restricted to [0,1]).
    if rc == "cnn":
        subset_indices = uniform_filter(attrs_all[:, 2], total_samples=10000)
    else:
        subset_indices = _uniform_filter_any(attrs_all[:, 1], total_samples=10000)

    # Load selected grids
    grids_main, _, _ = load_hdf5_raw(h5path, indices=subset_indices)
    print(f"Selected {len(grids_main)} grids uniformly across committor bins")

    # Compute reaction coordinate values for *all* frames
    if rc == "cnn":
        q_main = compute_committors_for_trajectory(grids_main, model, device)
    elif rc == "cluster":
        # attrs_all[:,1] is largest cluster size; subset_indices indexes into attrs_all
        cluster_main = np.asarray(attrs_all[subset_indices, 1], dtype=float)
        k, x0 = get_cluster_sigmoid_params(beta, h)
        q_main = sigmoid(cluster_main, k, x0)
        q_main = np.clip(q_main, 0.0, 1.0)
    else:  # cluster_raw
        q_main = np.asarray(attrs_all[subset_indices, 1], dtype=float)

    print("RC range:", float(np.min(q_main)), float(np.max(q_main)))

    # --- RC boundaries ---
    if rc == "cluster_raw":
        q_A, q_B = get_cluster_raw_qab(beta, h)
        model_error = 0.0
    else:
        q_A = 0.03
        q_B = 0.97
        model_error = 0.015

    num_steps = 12
    q_bins = np.linspace(q_A, q_B, num_steps)
    # Use unbounded outer edges so this works for both committor (0..1) and cluster sizes.
    full_bins = np.concatenate(([-np.inf], q_bins, [np.inf]))
    N = len(q_bins) - 1
    bin_ids = np.array([bin_state(q) for q in q_main])
    S = N + 2

    for b in range(0, S):
        indices = np.where(bin_ids == b)[0]
        if len(indices) == 0:
            print(f"Bin {b}: no frames available.")
            continue
        sampled = np.random.choice(indices)
        print(f"Bin {b}: sampled frame index {sampled}")

    m_list = [1024]
    n_repeats = 4
    C_dict = {}

    for m in m_list:
        print(f"\n=== Processing lag m={m} ===")

        generated_trajs = {}

        for b in range(S):
            frames_in_bin = np.where(bin_ids == b)[0]
            if len(frames_in_bin) == 0:
                print(f"Bin {b}: no frames available, skipping.")
                continue

            traj_list = []

            for r in range(n_repeats):
                print(f"  Bin {b}: repeat {r+1}/{n_repeats}")

                if len(frames_in_bin) >= gpu_nsms:
                    chosen = np.random.choice(frames_in_bin, size=gpu_nsms, replace=False)
                else:
                    chosen = np.random.choice(frames_in_bin, size=gpu_nsms, replace=True)

                gridlist = [grids_main[i].copy() for i in chosen]

                gasp.run_committor_calc(
                    L,
                    ngrids,
                    m + 1,
                    beta,
                    h,
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
                    max_keep_grids=2 * ngrids,
                )

                traj = np.zeros((ngrids, L, L), dtype=np.int8)
                for i in range(ngrids):
                    traj[i] = gasp.grids[-1][i].grid
                print("isweep of saved trajectory", gasp.grids[-1][0].isweep)

                traj_list.append(traj)

            generated_trajs[b] = np.concatenate(traj_list, axis=0)
            total_len = generated_trajs[b].shape[0]
            print(f"  Bin {b}: total trajectories = {total_len}")

        C_m = build_C_matrix_for_lag(m, generated_trajs, model_error)
        C_dict[m] = C_m

    Path("data").mkdir(parents=True, exist_ok=True)
    out_path = Path("data") / f"C_matrices_{beta:.3f}_{h:.3f}_{rc}.npz"
    np.savez(str(out_path), **{f"C_m{m}": C_dict[m] for m in m_list})
    print(f"\nSaved all C(m) matrices to {out_path}")

    if device == "cuda":
        torch.cuda.synchronize()
    t_total_end = time.perf_counter()
    print(f"\nTotal runtime (beta={beta:.3f}, h={h:.3f}): {t_total_end - t_total_start:.2f} s")


def main() -> None:
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, help="Override beta value")
    parser.add_argument("--h", type=float, help="Override h value")
    parser.add_argument(
        "--scan-bins",
        type=int,
        default=32,
        help="Number of grid variants with increasing +1 spins",
    )
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Skip the increasing-ones scan printout",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run over all (beta, h) pairs in RUNS (manually filled in markov.py)",
    )
    parser.add_argument(
        "--rc",
        choices=["cnn", "cluster", "cluster_raw"],
        default="cnn",
        help="Reaction coordinate used to bin the MSM: CNN committor, sigmoid(cluster_size), or raw cluster_size",
    )
    args = parser.parse_args()

    if args.sweep:
        if not RUNS:
            raise ValueError("RUNS is empty. Fill RUNS in markov.py or run without --sweep.")
        for beta, h in RUNS:
            run_one(
                beta=float(beta),
                h=float(h),
                config=config,
                scan_bins=int(args.scan_bins),
                run_scan=(not args.no_scan),
                rc=str(args.rc),
            )
    else:
        beta = args.beta if args.beta is not None else config.parameters.beta
        h = args.h if args.h is not None else config.parameters.h
        run_one(
            beta=float(beta),
            h=float(h),
            config=config,
            scan_bins=int(args.scan_bins),
            run_scan=(not args.no_scan),
            rc=str(args.rc),
        )


if __name__ == "__main__":
    main()
