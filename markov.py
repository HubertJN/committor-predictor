import time
import torch
import numpy as np
import argparse
from pathlib import Path
from utils.config import load_config
from utils.architecture import CNN
from utils.dataset import load_hdf5_raw, uniform_filter
import gasp
from scipy.ndimage import label
np.set_printoptions(linewidth=np.inf)
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
q_A = None
q_B = None
q_bins = None
full_bins = None
N = None
S = None
compute_q_for_frames = None


def _uniform_filter_any(labels: np.ndarray, num_bins: int = 10, seed: int = 42, total_samples: int = 10000) -> np.ndarray:
    """Uniform sampling across quantile bins for arbitrary-valued labels."""
    rng = np.random.default_rng(seed)
    x = np.asarray(labels, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite labels provided")

    q = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(x, q)
    for i in range(1, edges.size):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12

    labels_full = np.asarray(labels, dtype=float).ravel()
    idx_all = np.arange(labels_full.size)

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
    """Fast streaming inference for large trajectories."""
    Tlen = len(grids)
    q = np.ones(Tlen, dtype=np.float64)
    L = grids.shape[1]
    cpu_batch = torch.empty((batch_size, 1, L, L),
                            dtype=torch.float32,
                            pin_memory=True)

    t0 = time.perf_counter()
    for start in range(0, Tlen, batch_size):
        end = min(start + batch_size, Tlen)
        n = end - start
        cpu_batch[:n, 0].copy_(torch.from_numpy(grids[start:end]))
        gpu_batch = cpu_batch[:n].to(device, non_blocking=True)
        out = model(gpu_batch)
        q[start:end] = out.squeeze().cpu().numpy()

    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    print(f"  Fast batched inference took {t1 - t0:.2f} s")
    return q

def bin_state(q):
    if q < q_A:
        return 0
    elif q > q_B:
        return N + 1
    else:
        return np.digitize(q, q_bins)


def build_C_matrix_for_lag(m, traj_dict):
    C_matrix = np.zeros((S, S), dtype=np.float64)
    for b, frames in traj_dict.items():
        s_i = b
        q_dest = compute_q_for_frames(frames)
        for q_j in q_dest:
            s_j = bin_state(q_j)
            C_matrix[s_i, s_j] += 1
    print(C_matrix)
    return C_matrix


def run_one(
    beta: float,
    h: float,
    config,
    scan_bins: int,
    run_scan: bool,
    rc: str,
    lcs_qab: tuple[float, float] | None = None,
    model_path: str | None = None,
    c_matrix_out: str | None = None,
) -> None:
    global device, model, q_A, q_B, q_bins, full_bins, N, S, compute_q_for_frames

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
        if model_path is not None:
            checkpoint_path = model_path
        else:
            checkpoint_path = (
                f"{config.paths.save_dir}/{config.model.type}_ch{config.model.channels}_"
                f"cn{config.model.num_cnn_layers}_fc{config.model.num_fc_layers}_{beta:.3f}_{h:.3f}.pth"
            )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = CNN(
            channels=checkpoint["channels"],
            num_cnn_layers=checkpoint["num_cnn_layers"],
            num_fc_layers=checkpoint["num_fc_layers"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"{config.model.type.upper()} model loaded.")

        if run_scan:
            evaluate_increasing_ones_grid(model, L, device, scan_bins=scan_bins)

        compute_q_for_frames = lambda frames: compute_committors_for_trajectory(frames, model, device)
    elif rc == "lcs":
        if lcs_qab is not None:
            qA, qB = lcs_qab
        else:
            raise ValueError("lcs requires explicit (q_A, q_B) bounds via --qA and --qB")
        print(f"Using largest cluster size RC with (q_A, q_B)=({qA:.6g}, {qB:.6g})")

        def _q_from_frames(frames: np.ndarray) -> np.ndarray:
            return largest_cluster_sizes_up(frames)

        compute_q_for_frames = _q_from_frames
    else:
        raise ValueError("rc must be 'cnn' or 'lcs'")

    _, attrs_all, _ = load_hdf5_raw(h5path, load_grids=False)
    if rc == "cnn":
        subset_indices = uniform_filter(attrs_all[:, 2], total_samples=10000)
    else:
        subset_indices = _uniform_filter_any(attrs_all[:, 1], total_samples=10000)

    grids_main, _, _ = load_hdf5_raw(h5path, indices=subset_indices)
    print(f"Selected {len(grids_main)} grids uniformly across committor bins")

    if rc == "cnn": 
        # Short test run: N sweeps to refine q_A
        sweep = 8
        test_gridlist = [-np.ones_like(grids_main[0]) for _ in range(gpu_nsms)]
        
        gasp.run_committor_calc(
            L, ngrids, sweep + 1, beta, h,
            grid_output_int=sweep, mag_output_int=sweep,
            grid_input="NumPy", grid_array=test_gridlist,
            keep_grids=True, up_threshold=1.01, dn_threshold=-1.01,
            nsms=gpu_nsms, gpu_method=2, outname="None",
            max_keep_grids=2 * ngrids,
        )
        test_traj = np.zeros((ngrids, L, L), dtype=np.int8)
        for i in range(ngrids):
            test_traj[i] = gasp.grids[-1][i].grid
        
        q_test = compute_committors_for_trajectory(test_traj, model, device)
        q_A_refined = float(np.quantile(q_test, 0.85))
        print(f"Test run: q_A refined to {q_A_refined:.6f} (95th percentile of {sweep}-sweep endpoints)")
        q_A = q_A_refined
        q_B = 0.995

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(q_test, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(q_A_refined, color="red", linestyle="--", linewidth=2, label=f"q_A = {q_A_refined:.1f}")
        ax.set_xlabel("Largest Cluster Size", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Test Run LCS (2 sweeps, β={beta:.3f}, h={h:.3f})", fontsize=13)
        ax.legend()
        plt.tight_layout()

        plt.savefig("fig.pdf")
        plt.close()
        
        q_main = compute_committors_for_trajectory(grids_main, model, device)
    else:
        
        q_A,q_B = lcs_qab
        # Short test run: N sweeps to refine q_A for LCS
        sweep = 8
        test_gridlist = [-np.ones_like(grids_main[0]) for _ in range(gpu_nsms)]
        
        gasp.run_committor_calc(
            L, ngrids, sweep + 1, beta, h,
            grid_output_int=sweep, mag_output_int=sweep,
            grid_input="NumPy", grid_array=test_gridlist,
            keep_grids=True, up_threshold=1.01, dn_threshold=-1.01,
            nsms=gpu_nsms, gpu_method=2, outname="None",
            max_keep_grids=2 * ngrids,
        )
        test_traj = np.zeros((ngrids, L, L), dtype=np.int8)
        for i in range(ngrids):
            test_traj[i] = gasp.grids[-1][i].grid
                
        lcs_test = largest_cluster_sizes_up(test_traj)
        q_A_refined = float(np.quantile(lcs_test, 0.85))
        print(f"Test run: q_A refined to {q_A_refined:.1f} (75th percentile of {sweep}-sweep endpoints LCS)")
        q_A = q_A_refined
        
        # Plot histogram of LCS test
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(lcs_test, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        ax.axvline(q_A_refined, color="red", linestyle="--", linewidth=2, label=f"q_A = {q_A_refined:.1f}")
        ax.set_xlabel("Largest Cluster Size", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Test Run LCS (2 sweeps, β={beta:.3f}, h={h:.3f})", fontsize=13)
        ax.legend()
        plt.tight_layout()

        plt.savefig("fig.pdf")
        plt.close()

        q_main = np.asarray(attrs_all[subset_indices, 1], dtype=float)

    print("RC range:", float(np.min(q_main)), float(np.max(q_main)))
    num_steps = 12
    q_bins = np.linspace(q_A, q_B, num_steps)
    full_bins = np.concatenate(([-np.inf], q_bins, [np.inf]))
    N = len(q_bins) - 1
    bin_ids = np.array([bin_state(q) for q in q_main])
    S = N + 2

    for b in range(S):
        indices = np.where(bin_ids == b)[0]
        if len(indices) == 0:
            print(f"Bin {b}: no frames available.")
            continue
        sampled = np.random.choice(indices)
        print(f"Bin {b}: sampled frame index {sampled}")

    m_list = [16, 32, 64, 128, 256, 512, 1024, 2048]
    n_repeats = 1 
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
                    L, ngrids, m + 1, beta, h,
                    grid_output_int=m, mag_output_int=m,
                    grid_input="NumPy", grid_array=gridlist,
                    keep_grids=True, up_threshold=1.01, dn_threshold=-1.01,
                    nsms=gpu_nsms, gpu_method=2, outname="None",
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

        C_m = build_C_matrix_for_lag(m, generated_trajs)
        C_dict[m] = C_m

    if c_matrix_out is not None:
        out_path = Path(c_matrix_out)
    else:
        out_path = Path("data") / f"C_matrices_{beta:.3f}_{h:.3f}_{rc}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **{f"C_m{m}": C_dict[m] for m in m_list})
    print(f"\nSaved all C(m) matrices to {out_path}")

    if device == "cuda":
        torch.cuda.synchronize()
    t_total_end = time.perf_counter()
    print(f"Total runtime (beta={beta:.3f}, h={h:.3f}): {t_total_end - t_total_start:.2f} s")


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
        "--rc",
        choices=["cnn", "lcs"],
        default="cnn",
        help="Reaction coordinate used to bin the MSM: CNN committor or largest cluster size",
    )
    parser.add_argument(
        "--qA",
        type=float,
        default=None,
        help="q_A bound for --rc lcs (cluster-size units)",
    )
    parser.add_argument(
        "--qB",
        type=float,
        default=None,
        help="q_B bound for --rc lcs (cluster-size units)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Explicit path to CNN checkpoint .pth (overrides the default save_dir/{stem}.pth)",
    )
    parser.add_argument(
        "--c-matrix-out",
        type=str,
        default=None,
        help="Explicit output path for the C-matrix .npz (overrides data/C_matrices_...npz)",
    )
    args = parser.parse_args()

    lcs_qab: tuple[float, float] | None = None
    if args.qA is not None or args.qB is not None:
        if args.qA is None or args.qB is None:
            raise ValueError("Provide both --qA and --qB, or neither.")
        if not (np.isfinite(args.qA) and np.isfinite(args.qB) and float(args.qA) < float(args.qB)):
            raise ValueError(f"Invalid --qA/--qB: ({args.qA}, {args.qB}). Require finite and qA < qB.")
        lcs_qab = (float(args.qA), float(args.qB))

    beta = args.beta if args.beta is not None else config.parameters.beta
    h = args.h if args.h is not None else config.parameters.h
    run_one(
        beta=float(beta),
        h=float(h),
        config=config,
        scan_bins=int(args.scan_bins),
        run_scan=(not args.no_scan),
        rc=str(args.rc),
        lcs_qab=lcs_qab,
        model_path=args.model_path,
        c_matrix_out=args.c_matrix_out,
    )


if __name__ == "__main__":
    main()
