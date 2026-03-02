"""Plot nucleation rates vs beta (log y-axis).

This script is intended to consume the consolidated sweep output produced by
markov_analyse.py (default: data/msm_analysis_sweep.npz).

It plots J(beta) on a log-scale y-axis.

Typical usage:
  python plot_nucleation_rates.py --m 1024

If your sweep includes multiple h values, it will plot one curve per h.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.config import load_config


# =======================
# --- Color Palette ---
# =======================
colors = {
    "steel_blue": "#1F77B4",
    "light_steel_blue": "#AEC7E8",
    "orange": "#FF7F0E",
    "light_orange": "#FFBB78",
    "forest_green": "#2CA02C",
    "light_green": "#98DF8A",
    "firebrick_red": "#D62728",
    "soft_red": "#FF9896",
    "lavender": "#9467BD",
    "light_lavender": "#C5B0D5",
    "gray": "#7F7F7F",
    "light_gray": "#C7C7C7",
    "turquoise": "#17BECF",
    "light_turquoise": "#9EDAE5",
}


FONT_SIZE = 16
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE * 0.9,
        "ytick.labelsize": FONT_SIZE * 0.9,
        "legend.fontsize": FONT_SIZE * 0.9,
        "figure.titlesize": FONT_SIZE * 1.1,
    }
)


def _unique_sorted(values: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(values))


def _asym_yerr(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    lo = np.maximum(0.0, y - low)
    hi = np.maximum(0.0, high - y)
    return np.vstack([lo, hi])


def _load_brute_force_rates(brute_dir: Path) -> dict[str, np.ndarray]:
    """Load brute-force nucleation rate files from a directory.

    Expected per-run file format:
      data/nucleation_{beta:.3f}_{h:.3f}.npz
    containing at least: beta, h, rate_per_site.
    """

    if not brute_dir.exists():
        raise FileNotFoundError(f"Brute-force directory not found: {brute_dir}")

    files = sorted(brute_dir.glob("nucleation_*.npz"))
    if not files:
        raise FileNotFoundError(f"No brute-force files matching nucleation_*.npz in {brute_dir}")

    betas: list[float] = []
    hs: list[float] = []
    rates: list[float] = []
    rate_se: list[float] = []
    rate_ci_low: list[float] = []
    rate_ci_high: list[float] = []

    for fp in files:
        d = np.load(str(fp))
        if "beta" not in d or "h" not in d or "rate_per_site" not in d:
            continue
        b = float(np.asarray(d["beta"]).ravel()[0])
        hh = float(np.asarray(d["h"]).ravel()[0])
        r = float(np.asarray(d["rate_per_site"]).ravel()[0])
        se = float(np.asarray(d["rate_per_site_se"]).ravel()[0]) if "rate_per_site_se" in d else np.nan
        cil = (
            float(np.asarray(d["rate_per_site_ci_low"]).ravel()[0]) if "rate_per_site_ci_low" in d else np.nan
        )
        cih = (
            float(np.asarray(d["rate_per_site_ci_high"]).ravel()[0]) if "rate_per_site_ci_high" in d else np.nan
        )
        if np.isfinite(b) and np.isfinite(hh) and np.isfinite(r):
            betas.append(b)
            hs.append(hh)
            rates.append(r)
            rate_se.append(se)
            rate_ci_low.append(cil)
            rate_ci_high.append(cih)

    if not betas:
        raise ValueError(
            f"Found {len(files)} files in {brute_dir}, but none contained beta/h/rate_per_site."
        )

    return {
        "beta": np.asarray(betas, dtype=float),
        "h": np.asarray(hs, dtype=float),
        "rate_per_site": np.asarray(rates, dtype=float),
        "rate_per_site_se": np.asarray(rate_se, dtype=float),
        "rate_per_site_ci_low": np.asarray(rate_ci_low, dtype=float),
        "rate_per_site_ci_high": np.asarray(rate_ci_high, dtype=float),
    }


def _load_msm_per_run_files(msm_dir: Path) -> dict[str, np.ndarray]:
    """Load per-run MSM analysis outputs from a directory.

    Expected files:
      msm_analysis_{beta:.3f}_{h:.3f}_{rc}.npz
        where rc in {cnn, cluster, cluster_raw}.
    """

    if not msm_dir.exists():
        raise FileNotFoundError(f"MSM directory not found: {msm_dir}")

    files = sorted(msm_dir.glob("msm_analysis_*.npz"))
    if not files:
        raise FileNotFoundError(f"No MSM analysis files matching msm_analysis_*.npz in {msm_dir}")

    beta: list[float] = []
    h: list[float] = []
    rc: list[str] = []
    m: list[int] = []
    J_central: list[float] = []
    J_mean: list[float] = []
    J_std: list[float] = []
    J_ci_low: list[float] = []
    J_ci_high: list[float] = []

    for fp in files:
        d = np.load(str(fp))
        # Each file may contain multiple m values (records per lag)
        beta_arr = np.asarray(d["beta"], dtype=float)
        h_arr = np.asarray(d["h"], dtype=float)
        rc_arr = np.asarray(d["rc"], dtype=str) if "rc" in d else np.full(beta_arr.shape, "cnn", dtype=str)
        m_arr = np.asarray(d["m"], dtype=int)

        Jc = np.asarray(d["J_central"], dtype=float) if "J_central" in d else None
        Jm = np.asarray(d["J_mean"], dtype=float) if "J_mean" in d else None
        Js = np.asarray(d["J_std"], dtype=float) if "J_std" in d else None
        Jl = np.asarray(d["J_ci_low"], dtype=float) if "J_ci_low" in d else None
        Jh = np.asarray(d["J_ci_high"], dtype=float) if "J_ci_high" in d else None

        nrec = beta_arr.size
        if Jc is None or Jm is None:
            continue

        for i in range(nrec):
            beta.append(float(beta_arr[i]))
            h.append(float(h_arr[i]))
            rc.append(str(rc_arr[i]))
            m.append(int(m_arr[i]))
            J_central.append(float(Jc[i]))
            J_mean.append(float(Jm[i]))
            J_std.append(float(Js[i]) if Js is not None else np.nan)
            J_ci_low.append(float(Jl[i]) if Jl is not None else np.nan)
            J_ci_high.append(float(Jh[i]) if Jh is not None else np.nan)

    if not beta:
        raise ValueError(f"Found MSM files in {msm_dir}, but none contained expected arrays.")

    return {
        "beta": np.asarray(beta, dtype=float),
        "h": np.asarray(h, dtype=float),
        "rc": np.asarray(rc, dtype=str),
        "m": np.asarray(m, dtype=int),
        "J_central": np.asarray(J_central, dtype=float),
        "J_mean": np.asarray(J_mean, dtype=float),
        "J_std": np.asarray(J_std, dtype=float),
        "J_ci_low": np.asarray(J_ci_low, dtype=float),
        "J_ci_high": np.asarray(J_ci_high, dtype=float),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        type=str,
        default=None,
        help="Legacy consolidated sweep output from markov_analyse.py (optional)",
    )
    parser.add_argument(
        "--msm-dir",
        type=str,
        default="data",
        help="Directory containing per-run MSM analysis files msm_analysis_*.npz",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=None,
        help="Lag m to plot. Required if infile contains multiple m values.",
    )
    parser.add_argument(
        "--use",
        choices=["central", "mean"],
        default="central",
        help="Which nucleation rate estimate to plot",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="nucleation_rates_vs_beta.pdf",
        help="Output filename",
    )
    parser.add_argument(
        "--brute-dir",
        type=str,
        default=None,
        help="Optional directory containing brute-force files named data/nucleation_{beta:.3f}_{h:.3f}.npz",
    )
    parser.add_argument(
        "--rc",
        action="append",
        choices=["cnn", "cluster", "cluster_raw"],
        default=None,
        help="MSM reaction-coordinate curve(s) to plot. Repeatable. Default: cnn and cluster.",
    )
    parser.add_argument(
        "--all-rc",
        action="store_true",
        help="Plot all RCs found (cnn, cluster, cluster_raw)",
    )
    args = parser.parse_args()

    config = load_config("config.yaml")

    if args.infile is not None:
        # Legacy path: single consolidated file.
        infile = Path(args.infile)
        if not infile.exists():
            raise FileNotFoundError(f"Input file not found: {infile}")
        data = np.load(str(infile))
        beta = np.asarray(data["beta"], dtype=float)
        h = np.asarray(data["h"], dtype=float)
        m = np.asarray(data["m"], dtype=int)
        rc = np.asarray(data["rc"], dtype=str) if "rc" in data else np.full(beta.shape, "cnn", dtype=str)

        J_central_all = np.asarray(data["J_central"], dtype=float)
        J_mean_all = np.asarray(data["J_mean"], dtype=float)
        J_std_all = np.asarray(data["J_std"], dtype=float) if "J_std" in data else None
        J_ci_low_all = np.asarray(data["J_ci_low"], dtype=float) if "J_ci_low" in data else None
        J_ci_high_all = np.asarray(data["J_ci_high"], dtype=float) if "J_ci_high" in data else None
    else:
        msm = _load_msm_per_run_files(Path(args.msm_dir))
        beta = msm["beta"]
        h = msm["h"]
        m = msm["m"]
        rc = msm["rc"]

        J_central_all = msm["J_central"]
        J_mean_all = msm["J_mean"]
        J_std_all = msm["J_std"]
        J_ci_low_all = msm["J_ci_low"]
        J_ci_high_all = msm["J_ci_high"]

    if args.use == "central":
        J = np.asarray(J_central_all, dtype=float)
        J_label = r"I MCSS$^{-1}$"
    else:
        J = np.asarray(J_mean_all, dtype=float)
        J_label = r"I MCSS$^{-1}$"

    J_std = np.asarray(J_std_all, dtype=float) if J_std_all is not None else None
    J_ci_low = np.asarray(J_ci_low_all, dtype=float) if J_ci_low_all is not None else None
    J_ci_high = np.asarray(J_ci_high_all, dtype=float) if J_ci_high_all is not None else None

    unique_m = _unique_sorted(m)
    if args.m is None:
        if unique_m.size == 1:
            chosen_m = int(unique_m[0])
        else:
            raise ValueError(f"Multiple m values found in infile: {unique_m.tolist()}. Provide --m.")
    else:
        chosen_m = int(args.m)
        if chosen_m not in set(unique_m.tolist()):
            raise ValueError(f"Requested m={chosen_m} not found in infile. Available m: {unique_m.tolist()}")

    mask_m = m == chosen_m
    beta = beta[mask_m]
    h = h[mask_m]
    J = J[mask_m]
    rc = rc[mask_m]
    if J_std is not None:
        J_std = J_std[mask_m]
    if J_ci_low is not None:
        J_ci_low = J_ci_low[mask_m]
    if J_ci_high is not None:
        J_ci_high = J_ci_high[mask_m]

    unique_h = _unique_sorted(h)
    unique_rc = _unique_sorted(rc)

    if args.all_rc:
        rc_to_plot = ["cnn", "cluster", "cluster_raw"]
    elif args.rc is None:
        rc_to_plot = ["cnn", "cluster"]
    else:
        # Preserve user-provided order; de-duplicate.
        rc_to_plot = []
        for v in args.rc:
            if v not in rc_to_plot:
                rc_to_plot.append(v)

    brute = None
    if args.brute_dir is not None:
        brute = _load_brute_force_rates(Path(args.brute_dir))

    # Output directory
    out_dir = Path(config.paths.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out

    # Plot
    plt.figure(figsize=(9, 5))

    palette = [
        (colors["steel_blue"], colors["light_steel_blue"]),
        (colors["orange"], colors["light_orange"]),
        (colors["forest_green"], colors["light_green"]),
        (colors["lavender"], colors["light_lavender"]),
        (colors["firebrick_red"], colors["soft_red"]),
        (colors["turquoise"], colors["light_turquoise"]),
        (colors["gray"], colors["light_gray"]),
    ]

    rc_style = {
        "cnn": {"fmt": "o", "linestyle": "-", "markersize": 7},
        "cluster": {"fmt": "s", "linestyle": "--", "markersize": 6.5},
        "cluster_raw": {"fmt": "^", "linestyle": "-.", "markersize": 6.5},
    }

    for idx, h_val in enumerate(unique_h):
        c_main, c_line = palette[idx % len(palette)]

        labeled_h = False

        for rc_val in rc_to_plot:
            if rc_val not in set(unique_rc.tolist()):
                continue

            mask = (h == h_val) & (rc == rc_val)
            if not np.any(mask):
                continue

            beta_h = beta[mask]
            J_h = J[mask]
            J_std_h = J_std[mask] if J_std is not None else None
            J_ci_low_h = J_ci_low[mask] if J_ci_low is not None else None
            J_ci_high_h = J_ci_high[mask] if J_ci_high is not None else None

            order = np.argsort(beta_h)
            beta_h = beta_h[order]
            J_h = J_h[order]
            if J_std_h is not None:
                J_std_h = J_std_h[order]
            if J_ci_low_h is not None:
                J_ci_low_h = J_ci_low_h[order]
            if J_ci_high_h is not None:
                J_ci_high_h = J_ci_high_h[order]

            # MSM points + error bars
            yerr_msm = None
            if (
                args.use == "central"
                and J_ci_low_h is not None
                and J_ci_high_h is not None
                and np.all(np.isfinite(J_ci_low_h))
                and np.all(np.isfinite(J_ci_high_h))
            ):
                yerr_msm = _asym_yerr(J_h, J_ci_low_h, J_ci_high_h)
            elif J_std_h is not None and np.any(np.isfinite(J_std_h)):
                yerr_msm = np.where(np.isfinite(J_std_h), J_std_h, 0.0)

            style = rc_style.get(rc_val, {"fmt": "o", "linestyle": "-", "markersize": 7})
            label = None
            if unique_h.size > 1 and not labeled_h:
                label = f"h={float(h_val):.3f}"
                labeled_h = True
            plt.errorbar(
                beta_h,
                J_h,
                yerr=yerr_msm,
                fmt=style["fmt"],
                markersize=style["markersize"],
                color=c_main,
                ecolor=c_main,
                elinewidth=1.2,
                capsize=2,
                capthick=1.0,
                alpha=1.0,
                label=label,
                zorder=3,
            )
            plt.plot(
                beta_h,
                J_h,
                color=c_line,
                linewidth=2,
                alpha=1.0,
                linestyle=style["linestyle"],
            )

        # Brute-force overlay for this h
        if brute is not None:
            mask_b = np.isclose(brute["h"], float(h_val), atol=5e-4)
            if np.any(mask_b):
                beta_b = brute["beta"][mask_b]
                J_b = brute["rate_per_site"][mask_b]
                se_b = brute["rate_per_site_se"][mask_b]
                cil_b = brute["rate_per_site_ci_low"][mask_b]
                cih_b = brute["rate_per_site_ci_high"][mask_b]
                order_b = np.argsort(beta_b)
                beta_b = beta_b[order_b]
                J_b = J_b[order_b]
                se_b = se_b[order_b]
                cil_b = cil_b[order_b]
                cih_b = cih_b[order_b]

                yerr_brute = None
                if np.all(np.isfinite(cil_b)) and np.all(np.isfinite(cih_b)):
                    yerr_brute = _asym_yerr(J_b, cil_b, cih_b)
                elif np.any(np.isfinite(se_b)):
                    yerr_brute = np.where(np.isfinite(se_b), se_b, 0.0)

                plt.errorbar(
                    beta_b,
                    J_b,
                    yerr=yerr_brute,
                    fmt="x",
                    markersize=8,
                    color=c_main,
                    ecolor=c_line,
                    elinewidth=0.9,
                    capsize=2,
                    capthick=0.9,
                    alpha=0.55,
                    zorder=2,
                )
                plt.plot(
                    beta_b,
                    J_b,
                    color=c_line,
                    linewidth=1.5,
                    alpha=0.6,
                    linestyle=":",
                )

    plt.yscale("log")
    plt.xlabel(r"$\beta$")
    plt.ylabel(J_label)
    #plt.title(f"Nucleation rate vs beta")
    show_style_legend = (brute is not None) or (len([r for r in rc_to_plot if r in set(unique_rc.tolist())]) > 1)

    if show_style_legend:
        # Add marker-style legend entries (black) without duplicating the h-color legend.
        style_handles: list[Line2D] = []

        if "cnn" in set(unique_rc.tolist()) and "cnn" in rc_to_plot:
            style_handles.append(Line2D([0], [0], color="black", marker="o", linestyle="-", linewidth=2, label="MSM (CNN)"))
        if "cluster" in set(unique_rc.tolist()) and "cluster" in rc_to_plot:
            style_handles.append(
                Line2D([0], [0], color="black", marker="s", linestyle="--", linewidth=2, label="MSM (Cluster)")
            )
        if "cluster_raw" in set(unique_rc.tolist()) and "cluster_raw" in rc_to_plot:
            style_handles.append(
                Line2D([0], [0], color="black", marker="^", linestyle="-.", linewidth=2, label="MSM (Cluster raw)")
            )
        if brute is not None:
            style_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="x",
                    linestyle=":",
                    linewidth=1.5,
                    label="Brute force",
                )
            )

        ax = plt.gca()
        if unique_h.size > 1:
            hh, ll = ax.get_legend_handles_labels()
            ax.legend(
                handles=[*hh, *style_handles],
                labels=[*ll, *[h.get_label() for h in style_handles]],
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
            )
        else:
            ax.legend(
                handles=style_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
            )

        plt.tight_layout()
    elif unique_h.size > 1:
        plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
