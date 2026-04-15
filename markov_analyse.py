import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.config import load_config
from pathlib import Path

np.set_printoptions(linewidth=np.inf)


def load_count_matrices(beta: float, h: float, rc: str, c_matrix_in: str | None = None) -> tuple[list[int], dict[int, np.ndarray]]:
    if c_matrix_in is not None:
        data_path = Path(c_matrix_in)
    else:
        data_path = Path("data") / f"C_matrices_{beta:.3f}_{h:.3f}_{rc}.npz"
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


def stochasticity_checks(T):
    row_sums = T.sum(axis=1)
    neg = np.any(T < -1e-12)
    return row_sums.min(), row_sums.max(), neg


def implied_timescales(T, tau):
    eigvals, _ = np.linalg.eig(T)
    eigvals = np.real(eigvals)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    nontrivial = eigvals[1:6]
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

    j = np.argmax(np.abs(pi))
    if pi[j] < 0:
        pi = -pi
    pi = np.maximum(pi, 0)
    pi /= pi.sum()
    return pi


def mfpt_A_to_B(T, tau, A_states, B_states):
    S = T.shape[0]
    all_states = np.arange(S)

    mask_R = np.ones(S, dtype=bool)
    mask_R[B_states] = False
    R = all_states[mask_R]

    T_RR = T[np.ix_(R, R)]
    I = np.eye(len(R))
    one = np.ones(len(R))

    t_R = np.linalg.solve(I - T_RR, tau * one)

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
    rng = np.random.default_rng(rng_seed)

    C = np.asarray(C, dtype=float)
    S = C.shape[0]

    row_sums = C.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        T_central = np.where(row_sums > 0, C / row_sums, 0.0)

    MFPT_AB = mfpt_A_to_B(T_central, tau, A_states, B_states)
    k_AB = 1.0 / MFPT_AB
    J_AB = k_AB / (64 * 64)

    p = np.zeros_like(T_central)
    row_sums_int = C.sum(axis=1)
    for i in range(S):
        Ni = row_sums_int[i]
        if Ni > 0:
            p[i] = C[i] / Ni
        else:
            p[i, :] = 0.0

    J_samples = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        C_b = np.zeros_like(C)
        for i in range(S):
            Ni = row_sums_int[i]
            if Ni > 0:
                C_b[i] = rng.multinomial(Ni, p[i])
            else:
                C_b[i, :] = 0

        C_b_float = C_b.astype(float)
        row_sums_b = C_b_float.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            T_b = np.where(row_sums_b > 0, C_b_float / row_sums_b, 0.0)

        MFPT_b = mfpt_A_to_B(T_b, tau, A_states, B_states)
        k_b = 1.0 / MFPT_b
        J_samples[b] = k_b / (64 * 64)

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


def lumped_basin_metrics(T, tau, A_states, B_states):
    pi = stationary_distribution(T)

    # Conditional stationary weights inside each lumped basin
    wA = pi[A_states].astype(float)
    if wA.sum() > 0:
        wA /= wA.sum()
    else:
        wA = np.ones(len(A_states), dtype=float) / len(A_states)

    wB = pi[B_states].astype(float)
    if wB.sum() > 0:
        wB /= wB.sum()
    else:
        wB = np.ones(len(B_states), dtype=float) / len(B_states)

    P_AA = float(np.sum(wA[:, None] * T[np.ix_(A_states, A_states)]))
    P_AB = float(np.sum(wA[:, None] * T[np.ix_(A_states, B_states)]))
    P_BB = float(np.sum(wB[:, None] * T[np.ix_(B_states, B_states)]))
    P_BA = float(np.sum(wB[:, None] * T[np.ix_(B_states, A_states)]))

    pi_A = float(pi[A_states].sum())
    pi_B = float(pi[B_states].sum())

    MFPT_AB = float(mfpt_A_to_B(T, tau, A_states, B_states))
    k_AB = float(1.0 / MFPT_AB)
    J_AB = float(k_AB / (64 * 64))

    return {
        "P_AA": P_AA,
        "P_AB": P_AB,
        "P_BB": P_BB,
        "P_BA": P_BA,
        "pi_A": pi_A,
        "pi_B": pi_B,
        "MFPT_AB": MFPT_AB,
        "k_AB": k_AB,
        "J_AB": J_AB,
    }


def analyse_one(beta: float, h: float, rc: str, n_boot: int, rng_seed: int | None, verbose: bool,
                c_matrix_in: str | None = None) -> list[dict]:
    m_list, C_dict = load_count_matrices(beta, h, rc=rc, c_matrix_in=c_matrix_in)
    T_dict = counts_to_transition_matrices(m_list, C_dict)

    if verbose:
        print("\n=== MSM Analysis ===")

    records: list[dict] = []

    for m in m_list:
        T = T_dict[m]
        tau = m * 1.0

        if verbose:
            print(f"\n--- Lag m={m}, tau={tau} ---")

        rmin, rmax, neg = stochasticity_checks(T)
        eigvals, its = implied_timescales(T, tau)
        loc = locality_metric(T, band_radius=1)

        S_local = T.shape[0]
        A_states = np.array([0])
        B_states = np.array([S_local - 1])

        metrics = lumped_basin_metrics(T, tau, A_states, B_states)
        P_AA = metrics["P_AA"]
        P_AB = metrics["P_AB"]
        P_BB = metrics["P_BB"]
        P_BA = metrics["P_BA"]
        pi_A = metrics["pi_A"]
        pi_B = metrics["pi_B"]
        MFPT_AB = metrics["MFPT_AB"]
        k_AB = metrics["k_AB"]

        # Candidate lumped A basins
        A_candidates = [
            np.array([0]),
            np.array([0, 1]) if S_local >= 2 else np.array([0]),
            np.array([0, 1, 2]) if S_local >= 3 else np.array([0]),
        ]
        candidate_metrics = []
        for A_cand in A_candidates:
            cand = lumped_basin_metrics(T, tau, A_cand, B_states)
            candidate_metrics.append((A_cand, cand))

        ck_max = np.nan
        ck_fro = np.nan
        if 2 * m in T_dict:
            ck_max, ck_fro = ck_test(T, T_dict[2 * m])

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
            print(f"P(A->A): {P_AA:.6f}")
            print(f"P(A->B): {P_AB:.6f}")
            print(f"P(B->B): {P_BB:.6f}")
            print(f"P(B->A): {P_BA:.6f}")
            print(f"pi(A): {pi_A:.6f}")
            print(f"pi(B): {pi_B:.6f}")
            print(f"MFPT A->B: {MFPT_AB:.6f}")
            print(f"k_AB: {k_AB:.6e}")

            print("Candidate lumped A basins:")
            for A_cand, cand in candidate_metrics:
                label = "{" + ",".join(str(int(x)) for x in A_cand) + "}"
                print(
                    f"  A={label}: "
                    f"P(A->A)={cand['P_AA']:.6f}, "
                    f"P(A->B)={cand['P_AB']:.6f}, "
                    f"pi(A)={cand['pi_A']:.6f}, "
                    f"MFPT={cand['MFPT_AB']:.6f}, "
                    f"k_AB={cand['k_AB']:.6e}, "
                    f"J_AB={cand['J_AB']:.6e}"
                )

            if np.isfinite(ck_max):
                print(f"CK max abs error (2tau vs T(tau)^2): {ck_max:.6e}")
                print(f"CK Frobenius error: {ck_fro:.6e}")
            else:
                print("CK test: skipped (2m matrix not available)")

            exp = int(np.floor(np.log10(abs(J_central)))) if J_central != 0 else 0
            coeff_central = J_central / (10**exp) if J_central != 0 else 0
            coeff_std = J_std / (10**exp) if J_central != 0 else 0
            coeff_low = float(J_low) / (10**exp) if J_central != 0 else 0
            coeff_high = float(J_high) / (10**exp) if J_central != 0 else 0
            exp_str = f"e{exp}"
            print(f"Nucleation rate: ({coeff_central:.3f} +/- {coeff_std:.3f}){exp_str}")
            print(f"95% CI: [{coeff_low:.3f}{exp_str}, {coeff_high:.3f}{exp_str}]")

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
            "P_AA": P_AA,
            "P_AB": P_AB,
            "P_BB": P_BB,
            "P_BA": P_BA,
            "pi_A": pi_A,
            "pi_B": pi_B,
            "MFPT_AB": MFPT_AB,
            "k_AB": k_AB,
            "P_AA_01": float(candidate_metrics[1][1]["P_AA"]),
            "P_AA_012": float(candidate_metrics[2][1]["P_AA"]),
            "P_AB_01": float(candidate_metrics[1][1]["P_AB"]),
            "P_AB_012": float(candidate_metrics[2][1]["P_AB"]),
            "pi_A_01": float(candidate_metrics[1][1]["pi_A"]),
            "pi_A_012": float(candidate_metrics[2][1]["pi_A"]),
            "MFPT_AB_01": float(candidate_metrics[1][1]["MFPT_AB"]),
            "MFPT_AB_012": float(candidate_metrics[2][1]["MFPT_AB"]),
            "k_AB_01": float(candidate_metrics[1][1]["k_AB"]),
            "k_AB_012": float(candidate_metrics[2][1]["k_AB"]),
            "ck_max_abs": float(ck_max) if np.isfinite(ck_max) else np.nan,
            "ck_fro": float(ck_fro) if np.isfinite(ck_fro) else np.nan,
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

    P_AA = np.array([r["P_AA"] for r in records], dtype=float)
    P_AB = np.array([r["P_AB"] for r in records], dtype=float)
    P_BB = np.array([r["P_BB"] for r in records], dtype=float)
    P_BA = np.array([r["P_BA"] for r in records], dtype=float)
    pi_A = np.array([r["pi_A"] for r in records], dtype=float)
    pi_B = np.array([r["pi_B"] for r in records], dtype=float)
    MFPT_AB = np.array([r["MFPT_AB"] for r in records], dtype=float)
    k_AB = np.array([r["k_AB"] for r in records], dtype=float)

    P_AA_01 = np.array([r["P_AA_01"] for r in records], dtype=float)
    P_AA_012 = np.array([r["P_AA_012"] for r in records], dtype=float)
    P_AB_01 = np.array([r["P_AB_01"] for r in records], dtype=float)
    P_AB_012 = np.array([r["P_AB_012"] for r in records], dtype=float)
    pi_A_01 = np.array([r["pi_A_01"] for r in records], dtype=float)
    pi_A_012 = np.array([r["pi_A_012"] for r in records], dtype=float)
    MFPT_AB_01 = np.array([r["MFPT_AB_01"] for r in records], dtype=float)
    MFPT_AB_012 = np.array([r["MFPT_AB_012"] for r in records], dtype=float)
    k_AB_01 = np.array([r["k_AB_01"] for r in records], dtype=float)
    k_AB_012 = np.array([r["k_AB_012"] for r in records], dtype=float)

    ck_max_abs = np.array([r["ck_max_abs"] for r in records], dtype=float)
    ck_fro = np.array([r["ck_fro"] for r in records], dtype=float)

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
        P_AA=P_AA,
        P_AB=P_AB,
        P_BB=P_BB,
        P_BA=P_BA,
        pi_A=pi_A,
        pi_B=pi_B,
        MFPT_AB=MFPT_AB,
        k_AB=k_AB,
        P_AA_01=P_AA_01,
        P_AA_012=P_AA_012,
        P_AB_01=P_AB_01,
        P_AB_012=P_AB_012,
        pi_A_01=pi_A_01,
        pi_A_012=pi_A_012,
        MFPT_AB_01=MFPT_AB_01,
        MFPT_AB_012=MFPT_AB_012,
        k_AB_01=k_AB_01,
        k_AB_012=k_AB_012,
        ck_max_abs=ck_max_abs,
        ck_fro=ck_fro,
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
        "--rc",
        choices=["cnn", "lcs"],
        default="cnn",
        help="Reaction coordinate: CNN committor or largest cluster size",
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
        "--c-matrix-in",
        type=str,
        default=None,
        help="Explicit path to C-matrix .npz input (overrides data/C_matrices_...npz)",
    )
    parser.add_argument(
        "--msm-out",
        type=str,
        default=None,
        help="Explicit output path for the MSM analysis .npz (overrides --out-dir default)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-lag diagnostics",
    )
    args = parser.parse_args()

    beta = args.beta if args.beta is not None else config.parameters.beta
    h = args.h if args.h is not None else config.parameters.h
    out_dir = Path(args.out_dir)

    recs = analyse_one(
        beta=float(beta),
        h=float(h),
        rc=str(args.rc),
        n_boot=int(args.n_boot),
        rng_seed=int(args.seed),
        verbose=True,
        c_matrix_in=args.c_matrix_in,
    )
    if args.msm_out is not None:
        out_path = Path(args.msm_out)
    else:
        out_path = per_run_out_path(out_dir, float(beta), float(h), str(args.rc))
    save_records_npz(recs, out_path)
    print(f"Saved per-run results to {out_path}")


if __name__ == "__main__":
    main()