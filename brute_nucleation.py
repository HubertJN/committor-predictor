from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np  # Grids are stored as NumPy Arrays


def lambda_from_increments(n, t_points, K_cum, max_iter=1000, tol=1e-10):
    """
    n: total systems
    t_points: increasing array of times [t1, ..., tm]
    K_cum: cumulative events at those times [K(t1), ..., K(tm)]
    Returns MLE via Newton-Raphson on the binomial-increments likelihood and SE from observed information.
    """
    t = np.asarray(t_points, dtype=float)
    K = np.asarray(K_cum, dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("t_points must be strictly increasing")
    if np.any(np.diff(K) < 0) or K[0] < 0 or K[-1] > n:
        raise ValueError("K_cum must be nondecreasing in [0, n]")

    dt = np.diff(np.concatenate(([0.0], t)))
    dK = np.diff(np.concatenate(([0.0], K)))
    R_prev = n - np.concatenate(([0.0], K[:-1]))

    # Initialize lam using overall end-point (single-time estimate)
    p_end = K[-1] / n
    lam = 0.0 if p_end <= 0 else (np.inf if p_end >= 1 else -np.log(1.0 - p_end) / t[-1])

    # Bound initial lam
    if not np.isfinite(lam) or lam <= 0:
        lam = (K[-1] + 1e-6) / (n * t[-1] + 1e-6)

    lconv = False
    for _ in range(max_iter):
        exp_term = np.exp(-lam * dt)
        p = 1.0 - exp_term                 # Binomial increment probs
        p_prime = dt * exp_term            # dp/dlambda = dt * e^{-lambda dt}

        # Score U(lambda) = sum p' * (dK - R_prev * p) / (p*(1-p))
        denom = p * (1.0 - p)
        # Avoid zero denominators
        safe = denom > 0
        U = np.sum(p_prime[safe] * (dK[safe] - R_prev[safe] * p[safe]) / denom[safe])

        # Observed information I(lambda) ~ sum R_prev * (p')^2 / (p*(1-p))
        I = np.sum(R_prev[safe] * (p_prime[safe]**2) / denom[safe])

        step = U / I if I > 0 else 0.0
        lam_new = lam + step
        if lam_new <= 0:
            lam_new = lam / 2.0
        if abs(lam_new - lam) < tol *  lam:
            lam = lam_new
            print(f"Converged estimate of lambda to : {lam}")
            lconv = True
            break
        lam = lam_new


    if not lconv:
        print(f"Reached max iterations before converging to specified tolerance!")

    se = (1.0 / np.sqrt(I)) if I > 0 else np.nan
    ci = (lam - 1.96*se, lam + 1.96*se) if np.isfinite(se) else (np.nan, np.nan)
    return {"lambda_hat": lam, "se": se, "ci95_normal": ci}


def estimate_rate_from_frac(
    frac: np.ndarray,
    ngrids: int,
    L: int,
    mag_output_int: int,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> dict:
    """Estimate nucleation rate (per site) from fraction nucleated time series.

    Uses the existing binomial-increments MLE on cumulative events K(t) = frac * ngrids.
    An induction time is estimated as the last time before the first nonzero event.
    """

    frac = np.asarray(frac, dtype=float)
    if frac.ndim != 1:
        raise ValueError("frac must be a 1D array")
    if frac.size < 3:
        raise ValueError("frac is too short to estimate a rate")

    # Time axis in sweeps (matches sampling interval)
    xdata = np.arange(frac.size, dtype=float) * float(mag_output_int)

    # First observed nucleation (if any)
    nz = np.argwhere(frac > 0)
    if nz.size == 0:
        return {
            "lambda_hat": 0.0,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "t_ind": float(xdata[-1]),
            "rate_per_site": 0.0,
            "rate_per_site_se": np.nan,
            "rate_per_site_ci_low": np.nan,
            "rate_per_site_ci_high": np.nan,
        }

    first_idx = int(nz[0][0])
    si = max(0, first_idx - 1)

    # Induction time: take the last time before first nonzero event
    t_ind = float(xdata[si])

    K = frac * float(ngrids)
    shifted_x = xdata - t_ind

    # Use tail starting at si so shifted_x starts at 0 and then increases
    rate_est = lambda_from_increments(
        ngrids,
        shifted_x[si:],
        K[si:],
        max_iter=max_iter,
        tol=tol,
    )

    lam = float(rate_est["lambda_hat"])
    se = float(rate_est["se"])
    ci_low, ci_high = rate_est["ci95_normal"]
    ci_low = float(ci_low)
    ci_high = float(ci_high)

    area = float(L * L)
    return {
        "lambda_hat": lam,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_ind": t_ind,
        "rate_per_site": lam / area,
        "rate_per_site_se": se / area if np.isfinite(se) else np.nan,
        "rate_per_site_ci_low": ci_low / area if np.isfinite(ci_low) else np.nan,
        "rate_per_site_ci_high": ci_high / area if np.isfinite(ci_high) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, required=True, help="Beta value")
    parser.add_argument("--h", type=float, required=True, help="h value")
    parser.add_argument("--L", type=int, default=64)
    parser.add_argument(
        "--ngrids",
        type=int,
        default=3328,
        help="Override number of grids (default: 4*gasp.gpu_nsms*32)",
    )
    parser.add_argument("--nsweeps", type=int, default=100000)
    parser.add_argument("--grid-output-int", type=int, default=100)
    parser.add_argument("--mag-output-int", type=int, default=100)
    parser.add_argument("--cv", type=str, default="largest_cluster")
    parser.add_argument("--up-threshold", type=int, default=2048)
    parser.add_argument("--gpu-method", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path for this run (default: data/nucleation_{beta:.3f}_{h:.3f}.npz)",
    )
    args = parser.parse_args()

    L = int(args.L)
    ngrids = int(args.ngrids)
    nsweeps = int(args.nsweeps)
    grid_output_int = int(args.grid_output_int)
    mag_output_int = int(args.mag_output_int)

    beta = float(args.beta)
    h = float(args.h)

    print("\n" + "=" * 80)
    print(f"Brute nucleation: beta={beta:.3f}, h={h:.3f}")
    print("=" * 80)

    import gasp

    frac = gasp.run_nucleation_swarm(
        int(L),
        int(ngrids),
        int(nsweeps),
        float(beta),
        float(h),
        grid_output_int=int(grid_output_int),
        mag_output_int=int(mag_output_int),
        cv=str(args.cv),
        up_threshold=int(args.up_threshold),
        keep_grids=False,
        gpu_method=int(args.gpu_method),
        outname="None",
    )
    frac = np.asarray(frac, dtype=float)

    est = estimate_rate_from_frac(
        frac=frac,
        ngrids=ngrids,
        L=L,
        mag_output_int=mag_output_int,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
    )

    print(f"Induction time (sweeps): {est['t_ind']:.1f}")
    print(f"Nucleation rate per site: {est['rate_per_site']:.6e}")
    if np.isfinite(est["rate_per_site_se"]):
        print(f"SE (per site):           {est['rate_per_site_se']:.6e}")
        print(
            f"95% CI (per site):       [{est['rate_per_site_ci_low']:.6e}, {est['rate_per_site_ci_high']:.6e}]"
        )

    if args.out is None:
        out_dir = Path("data")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Intentionally omit suffix; np.savez will append .npz
        out_path = out_dir / f"nucleation_{beta:.3f}_{h:.3f}"
        saved_path_for_print = out_path.with_suffix(".npz")
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        saved_path_for_print = out_path

    np.savez(
        str(out_path),
        beta=np.asarray([beta], dtype=float),
        h=np.asarray([h], dtype=float),
        L=np.asarray([L], dtype=int),
        ngrids=np.asarray([ngrids], dtype=int),
        nsweeps=np.asarray([nsweeps], dtype=int),
        grid_output_int=np.asarray([grid_output_int], dtype=int),
        mag_output_int=np.asarray([mag_output_int], dtype=int),
        cv=np.asarray([str(args.cv)], dtype=object),
        up_threshold=np.asarray([int(args.up_threshold)], dtype=int),
        gpu_method=np.asarray([int(args.gpu_method)], dtype=int),
        frac=np.asarray(frac, dtype=float),
        rate_per_site=np.asarray([float(est["rate_per_site"])], dtype=float),
        rate_per_site_se=np.asarray([float(est["rate_per_site_se"])], dtype=float),
        rate_per_site_ci_low=np.asarray([float(est["rate_per_site_ci_low"])], dtype=float),
        rate_per_site_ci_high=np.asarray([float(est["rate_per_site_ci_high"])], dtype=float),
        t_ind=np.asarray([float(est["t_ind"])], dtype=float),
        lambda_hat=np.asarray([float(est["lambda_hat"])], dtype=float),
        lambda_se=np.asarray([float(est["se"])], dtype=float),
        lambda_ci_low=np.asarray([float(est["ci_low"])], dtype=float),
        lambda_ci_high=np.asarray([float(est["ci_high"])], dtype=float),
        success=np.asarray([True], dtype=bool),
        error_msg=np.asarray([""], dtype=object),
    )

    print(f"\nSaved brute-force nucleation result to {saved_path_for_print}")


if __name__ == "__main__":
    main()
