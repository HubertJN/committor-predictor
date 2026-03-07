import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.config import load_config
import math
from pathlib import Path

np.set_printoptions(linewidth=np.inf)


# =======================
# --- MANUALLY FILL RUNS FOR SWEEP ---
# =======================
# Enable with: python markov_analyse.py --sweep
# Example:
# RUNS = [
#     (0.511, 0.040),
#     (0.526, 0.050),
# ]
RUNS: list[tuple[float, float]] = [
    (0.474, 0.010),
    (0.486, 0.020),
    (0.498, 0.030),
    (0.511, 0.040),
    (0.526, 0.050),
    (0.538, 0.060),
    (0.550, 0.070),
    (0.564, 0.080),
    (0.576, 0.090),
    (0.588, 0.100),
]

def load_count_matrices(beta: float, h: float, rc: str) -> tuple[list[int], dict[int, np.ndarray]]:
    data_path = Path("data") / f"C_matrices_{beta:.3f}_{h:.3f}_{rc}.npz"
    # Backward-compatible fallback for older CNN-only runs
    if not data_path.exists() and rc == "cnn":
        legacy = Path("data") / f"C_matrices_{beta:.3f}_{h:.3f}.npz"
        if legacy.exists():
            data_path = legacy

    if not data_path.exists():
        raise FileNotFoundError(f"Missing count-matrix file: {data_path}")

    data = np.load(str(data_path))
    m_list = sorted(int(k.split("m")[-1]) for k in data.keys())
    C_dict: dict[int, np.ndarray] = {m: data[f"C_m{m}"] for m in m_list}
    return m_list, C_dict


def counts_to_transition_matrices(m_list: list[int], C_dict: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    T_dict: dict[int, np.ndarray] = {}
    for m in m_list:
        C = np.asarray(C_dict[m], dtype=float)
        row_sums = C.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.where(row_sums > 0, C / row_sums, 0.0)
        T_dict[m] = T
    return T_dict

# ============================
# Helper Functions
# ============================

def stochasticity_checks(T):
    row_sums = T.sum(axis=1)
    neg = np.any(T < -1e-12)
    return row_sums.min(), row_sums.max(), neg

def implied_timescales(T, tau):
    # eigenvalues
    eigvals, _ = np.linalg.eig(T)
    eigvals = np.real(eigvals)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]

    # skip λ = 1
    nontrivial = eigvals[1:6]

    # timescales
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        its = -tau / np.log(nontrivial)
    return eigvals, its

def locality_metric(T, band_radius=1):
    S = T.shape[0]
    total = T.sum()
    band_total = 0.0

    for i in range(S):
        j_min = max(0, i - band_radius)
        j_max = min(S, i + band_radius + 1)
        band_total += T[i, j_min:j_max].sum()

    return band_total / total

def ck_test(T_tau, T_2tau):
    T_pred = T_tau @ T_tau
    diff = T_pred - T_2tau
    return np.max(np.abs(diff)), np.linalg.norm(diff)

def stationary_distribution(T):
    eigvals, eigvecs = np.linalg.eig(T.T)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = eigvecs[:, idx]

    # enforce non-negativity and normalization
    j = np.argmax(np.abs(pi))
    if pi[j] < 0:
        pi = -pi
    pi = np.maximum(pi, 0)
    pi /= pi.sum()
    return pi

def mfpt_A_to_B(T, tau, A_states, B_states):
    S = T.shape[0]
    all_states = np.arange(S)

    # R = non-B states
    mask_R = np.ones(S, dtype=bool)
    mask_R[B_states] = False
    R = all_states[mask_R]

    T_RR = T[np.ix_(R, R)]
    I = np.eye(len(R))
    one = np.ones(len(R))

    t_R = np.linalg.solve(I - T_RR, tau * one)

    # MFPT from A: average over A states within R
    mfpts_A = []
    for a in A_states:
        idx_a = np.where(R == a)[0][0]
        mfpts_A.append(t_R[idx_a])

    MFPT_AB = np.mean(mfpts_A)
    return MFPT_AB

def bootstrap_J_AB_from_counts(C,
                               tau,
                               A_states,
                               B_states,
                               n_boot=500,
                               rng_seed=None):
    """
    Parametric bootstrap for nucleation rate J_AB from a count matrix C.

    - C : (S, S) int array, counts at lag tau
    - tau : lag time
    - A_states, B_states : arrays of state indices
    - n_boot : number of bootstrap samples
    """
    rng = np.random.default_rng(rng_seed)

    C = np.asarray(C, dtype=float)
    S = C.shape[0]

    # Central transition matrix from C (same as in your code)
    row_sums = C.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        T_central = np.where(row_sums > 0, C / row_sums, 0.0)

    # Central MFPT and nucleation rate
    MFPT_AB = mfpt_A_to_B(T_central, tau, A_states, B_states)
    k_AB = 1.0 / MFPT_AB
    J_AB = k_AB / (64 * 64)

    # Precompute row-wise probabilities for multinomial resampling
    p = np.zeros_like(T_central)
    row_sums_int = C.sum(axis=1)
    for i in range(S):
        Ni = row_sums_int[i]
        if Ni > 0:
            p[i] = C[i] / Ni
        else:
            # If no outgoing counts from i, keep it as a row of zeros
            # (same behavior as your T-construction)
            p[i, :] = 0.0

    # Bootstrap samples of J_AB
    J_samples = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        C_b = np.zeros_like(C)
        for i in range(S):
            Ni = row_sums_int[i]
            if Ni > 0:
                C_b[i] = rng.multinomial(Ni, p[i])
            else:
                C_b[i, :] = 0

        # Build T_b in the same way as central T
        C_b_float = C_b.astype(float)
        row_sums_b = C_b_float.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            T_b = np.where(row_sums_b > 0, C_b_float / row_sums_b, 0.0)

        MFPT_b = mfpt_A_to_B(T_b, tau, A_states, B_states)
        k_b = 1.0 / MFPT_b
        J_samples[b] = k_b / (64 * 64)

    # Summary statistics
    J_mean = np.mean(J_samples)
    J_std = np.std(J_samples, ddof=1)
    J_low, J_high = np.percentile(J_samples, [2.5, 97.5])

    stats = {
        "J_central": J_AB,
        "J_mean": J_mean,
        "J_std": J_std,
        "J_ci_95": (J_low, J_high),
    }
    return stats


def analyse_one(beta: float, h: float, rc: str, n_boot: int, rng_seed: int | None, verbose: bool) -> list[dict]:
    m_list, C_dict = load_count_matrices(beta, h, rc=rc)
    T_dict = counts_to_transition_matrices(m_list, C_dict)

    if verbose:
        print("\n=== MSM Analysis ===")

    records: list[dict] = []

    for m in m_list:
        T = T_dict[m]
        tau = m * 1.0  # Δt = 1

        if verbose:
            print(f"\n--- Lag m={m}, tau={tau} ---")

        rmin, rmax, neg = stochasticity_checks(T)
        eigvals, its = implied_timescales(T, tau)
        loc = locality_metric(T, band_radius=1)

        S_local = T.shape[0]
        A_states = np.array([0])
        B_states = np.array([S_local - 1])

        C = C_dict[m]
        stats = bootstrap_J_AB_from_counts(
            C,
            tau,
            A_states,
            B_states,
            n_boot=int(n_boot),
            rng_seed=rng_seed,
        )

        J_central = float(stats["J_central"])
        J_low, J_high = stats["J_ci_95"]
        J_std = float(stats["J_std"])

        if verbose:
            print(f"Row sum min/max: {rmin:.6f}, {rmax:.6f}")
            print(f"Any negative entries: {neg}")
            print("Eigenvalues:", eigvals[:5])
            print(f"Locality (±1 bin): {loc:.4f}")

            exp = math.floor(math.log10(abs(J_central))) if J_central != 0 else 0
            coeff_central = J_central / (10**exp) if J_central != 0 else 0
            coeff_std = J_std / (10**exp) if J_central != 0 else 0
            coeff_low = float(J_low) / (10**exp) if J_central != 0 else 0
            coeff_high = float(J_high) / (10**exp) if J_central != 0 else 0
            print(f"Nucleation rate: ({coeff_central:.3f} +/- {coeff_std:.3f})e{exp}")
            print(f"95% CI: [{coeff_low:.3f}e{exp}, {coeff_high:.3f}e{exp}]")

        record = {
            "beta": float(beta),
            "h": float(h),
            "rc": str(rc),
            "m": int(m),
            "tau": float(tau),
            "row_sum_min": float(rmin),
            "row_sum_max": float(rmax),
            "any_negative": bool(neg),
            "eigvals_first5": np.asarray(eigvals[:5], dtype=float),
            "its_first5": np.asarray(its[:5], dtype=float),
            "locality_pm1": float(loc),
            "J_central": float(J_central),
            "J_mean": float(stats["J_mean"]),
            "J_std": float(J_std),
            "J_ci_low": float(J_low),
            "J_ci_high": float(J_high),
            "n_boot": int(n_boot),
            "rng_seed": (-1 if rng_seed is None else int(rng_seed)),
        }
        records.append(record)

    return records


def save_records_npz(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("No records to save")

    betas = np.array([r["beta"] for r in records], dtype=float)
    hs = np.array([r["h"] for r in records], dtype=float)
    rc = np.array([str(r.get("rc", "cnn")) for r in records], dtype=str)
    ms = np.array([r["m"] for r in records], dtype=int)
    taus = np.array([r["tau"] for r in records], dtype=float)

    row_sum_min = np.array([r["row_sum_min"] for r in records], dtype=float)
    row_sum_max = np.array([r["row_sum_max"] for r in records], dtype=float)
    any_negative = np.array([r["any_negative"] for r in records], dtype=bool)
    locality_pm1 = np.array([r["locality_pm1"] for r in records], dtype=float)

    eigvals_first5 = np.stack([r["eigvals_first5"] for r in records], axis=0)
    its_first5 = np.stack([r["its_first5"] for r in records], axis=0)

    J_central = np.array([r["J_central"] for r in records], dtype=float)
    J_mean = np.array([r["J_mean"] for r in records], dtype=float)
    J_std = np.array([r["J_std"] for r in records], dtype=float)
    J_ci_low = np.array([r["J_ci_low"] for r in records], dtype=float)
    J_ci_high = np.array([r["J_ci_high"] for r in records], dtype=float)

    n_boot = np.array([r["n_boot"] for r in records], dtype=int)
    rng_seed = np.array([r["rng_seed"] for r in records], dtype=int)

    np.savez(
        str(out_path),
        beta=betas,
        h=hs,
        rc=rc,
        m=ms,
        tau=taus,
        row_sum_min=row_sum_min,
        row_sum_max=row_sum_max,
        any_negative=any_negative,
        locality_pm1=locality_pm1,
        eigvals_first5=eigvals_first5,
        its_first5=its_first5,
        J_central=J_central,
        J_mean=J_mean,
        J_std=J_std,
        J_ci_low=J_ci_low,
        J_ci_high=J_ci_high,
        n_boot=n_boot,
        rng_seed=rng_seed,
    )


def per_run_out_path(out_dir: Path, beta: float, h: float, rc: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"msm_analysis_{beta:.3f}_{h:.3f}_{rc}.npz"


def main() -> None:
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, help="Override beta value")
    parser.add_argument("--h", type=float, help="Override h value")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run over all (beta, h) pairs in RUNS (manually filled in markov_analyse.py)",
    )
    parser.add_argument(
        "--everything",
        action="store_true",
        help="Convenience flag: equivalent to --sweep --rc all",
    )
    parser.add_argument(
        "--rc",
        choices=["cnn", "cluster", "cluster_raw", "both", "all"],
        default="cnn",
        help="Which reaction-coordinate version(s) to analyse",
    )
    parser.add_argument("--n-boot", type=int, default=500, help="Bootstrap samples per (beta,h,m)")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed for bootstrap")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Directory to write per-run MSM analysis files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-lag diagnostics (noisy for sweeps)",
    )
    args = parser.parse_args()

    if args.everything:
        args.sweep = True
        args.rc = "all"

    out_dir = Path(args.out_dir)
    if args.rc == "both":
        rc_list = ["cnn", "cluster"]
    elif args.rc == "all":
        rc_list = ["cnn", "cluster", "cluster_raw"]
    else:
        rc_list = [str(args.rc)]

    if args.sweep:
        if not RUNS:
            raise ValueError("RUNS is empty. Fill RUNS in markov_analyse.py or run without --sweep.")
        for beta, h in RUNS:
            for rc in rc_list:
                try:
                    print("\n" + "=" * 80)
                    print(f"Analysing beta={beta:.3f}, h={h:.3f}, rc={rc}")
                    print("=" * 80)
                    recs = analyse_one(
                        beta=float(beta),
                        h=float(h),
                        rc=str(rc),
                        n_boot=int(args.n_boot),
                        rng_seed=int(args.seed),
                        verbose=bool(args.verbose),
                    )
                    out_path = per_run_out_path(out_dir, float(beta), float(h), str(rc))
                    save_records_npz(recs, out_path)
                    print(f"Saved per-run results to {out_path}")
                except FileNotFoundError as e:
                    print(str(e))
                    continue
    else:
        beta = args.beta if args.beta is not None else config.parameters.beta
        h = args.h if args.h is not None else config.parameters.h
        for rc in rc_list:
            recs = analyse_one(
                beta=float(beta),
                h=float(h),
                rc=str(rc),
                n_boot=int(args.n_boot),
                rng_seed=int(args.seed),
                verbose=True,
            )
            out_path = per_run_out_path(out_dir, float(beta), float(h), str(rc))
            save_records_npz(recs, out_path)
            print(f"Saved per-run results to {out_path}")


if __name__ == "__main__":
    main()

