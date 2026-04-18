"""Plot nucleation rates vs beta (log y-axis).

Reads per-run MSM outputs (msm_analysis_*.npz) and overlays brute-force
nucleation rates (nucleation_*.npz).

You always get the brute-force curve(s). The only option is which MSM reaction
coordinate(s) to show: cnn, lcs, or both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.config import load_config


FONT_SIZE = 16
MAX_H_VALUES_PER_BETA = 3

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


def _beta_window_mask(beta: np.ndarray, beta_min: float | None, beta_max: float | None) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    mask = np.ones(beta.shape, dtype=bool)
    if beta_min is not None:
        mask &= beta >= float(beta_min)
    if beta_max is not None:
        mask &= beta <= float(beta_max)
    return mask


def _top_h_pair_set(
    beta: np.ndarray,
    h: np.ndarray,
    max_h_values_per_beta: int,
) -> set[tuple[float, float]]:
    beta = np.asarray(beta, dtype=float)
    h = np.asarray(h, dtype=float)

    if beta.size == 0 or max_h_values_per_beta <= 0:
        return set()

    allowed_pairs: set[tuple[float, float]] = set()
    unique_beta = np.unique(beta)
    for beta_val in unique_beta:
        mask_b = np.isclose(beta, float(beta_val), atol=5e-4)
        h_for_beta = np.unique(h[mask_b])
        if h_for_beta.size == 0:
            continue
        highest_h = np.sort(h_for_beta)[-max_h_values_per_beta:]
        for h_val in highest_h:
            allowed_pairs.add((float(beta_val), float(h_val)))

    return allowed_pairs


def _pair_mask(beta: np.ndarray, h: np.ndarray, allowed_pairs: set[tuple[float, float]]) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    h = np.asarray(h, dtype=float)

    if beta.size == 0:
        return np.zeros(beta.shape, dtype=bool)
    if not allowed_pairs:
        return np.ones(beta.shape, dtype=bool)

    mask = np.zeros(beta.shape, dtype=bool)
    for beta_val, h_val in allowed_pairs:
        mask |= np.isclose(beta, beta_val, atol=5e-4) & np.isclose(h, h_val, atol=5e-4)
    return mask


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
        where rc in {cnn, cluster, lcs}.
    
    Filters to only include records with lag_is_selected=True if available.
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
        
        # Load lag_is_selected if available, otherwise include all
        lag_selected = np.asarray(d["lag_is_selected"], dtype=bool) if "lag_is_selected" in d else np.ones(beta_arr.shape, dtype=bool)

        nrec = beta_arr.size
        if Jc is None or Jm is None:
            continue

        for i in range(nrec):
            # Skip records that weren't selected by automatic lag selection
            if not lag_selected[i]:
                continue
            
            beta.append(float(beta_arr[i]))
            h.append(float(h_arr[i]))
            # Map cluster_raw to lcs for backward compatibility
            rc_val = str(rc_arr[i])
            if rc_val == "cluster_raw":
                rc_val = "lcs"
            rc.append(rc_val)
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
        "--msm-dir",
        type=str,
        default="data",
        help="Directory containing per-run MSM analysis files msm_analysis_*.npz",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="nucleation_rates_vs_beta.pdf",
        help="Output filename",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(12.0, 5.0),
        metavar=("W", "H"),
        help="Figure size in inches: width height",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.5,
        help="Minimum beta to plot (inclusive)",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=0.6,
        help="Maximum beta to plot (inclusive)",
    )
    parser.add_argument(
        "--brute-dir",
        type=str,
        default=None,
        help="Directory containing brute-force files named nucleation_{beta:.3f}_{h:.3f}.npz. Defaults to --msm-dir.",
    )
    parser.add_argument(
        "--rc",
        choices=["cnn", "lcs", "both"],
        default="both",
        help="Which MSM reaction-coordinate curve(s) to plot",
    )
    args = parser.parse_args()

    config = load_config("config.yaml")

    msm = _load_msm_per_run_files(Path(args.msm_dir))
    beta = msm["beta"]
    h = msm["h"]
    m = msm["m"]
    rc = msm["rc"]

    # Prefer central estimate if present.
    J = np.asarray(msm["J_central"], dtype=float)
    J_label = r"I MCSS$^{-1}$"
    J_std = np.asarray(msm["J_std"], dtype=float)
    J_ci_low = np.asarray(msm["J_ci_low"], dtype=float)
    J_ci_high = np.asarray(msm["J_ci_high"], dtype=float)

    # Optional beta window
    mask_beta = _beta_window_mask(beta, args.beta_min, args.beta_max)
    beta = beta[mask_beta]
    h = h[mask_beta]
    J = J[mask_beta]
    rc = rc[mask_beta]
    if J_std is not None:
        J_std = J_std[mask_beta]
    if J_ci_low is not None:
        J_ci_low = J_ci_low[mask_beta]
    if J_ci_high is not None:
        J_ci_high = J_ci_high[mask_beta]

    unique_rc = _unique_sorted(rc)

    if args.rc == "both":
        rc_to_plot = ["cnn", "lcs"]
    else:
        rc_to_plot = [args.rc]

    brute_dir = Path(args.brute_dir) if args.brute_dir is not None else Path(args.msm_dir)
    brute = _load_brute_force_rates(brute_dir)

    brute_mask_beta = _beta_window_mask(brute["beta"], args.beta_min, args.beta_max)
    for k in list(brute.keys()):
        brute[k] = np.asarray(brute[k])[brute_mask_beta]
    if brute["beta"].size == 0:
        raise ValueError("No brute-force points remain after applying --beta-min/--beta-max.")

    # Hardcoded filter: only keep the highest h values for each beta.
    beta_all = np.concatenate([beta, brute["beta"]]) if beta.size > 0 else np.asarray(brute["beta"], dtype=float)
    h_all = np.concatenate([h, brute["h"]]) if h.size > 0 else np.asarray(brute["h"], dtype=float)
    allowed_pairs = _top_h_pair_set(beta_all, h_all, MAX_H_VALUES_PER_BETA)

    mask_top_h_msm = _pair_mask(beta, h, allowed_pairs)
    beta = beta[mask_top_h_msm]
    h = h[mask_top_h_msm]
    J = J[mask_top_h_msm]
    rc = rc[mask_top_h_msm]
    if J_std is not None:
        J_std = J_std[mask_top_h_msm]
    if J_ci_low is not None:
        J_ci_low = J_ci_low[mask_top_h_msm]
    if J_ci_high is not None:
        J_ci_high = J_ci_high[mask_top_h_msm]

    mask_top_h_brute = _pair_mask(brute["beta"], brute["h"], allowed_pairs)
    for k in list(brute.keys()):
        brute[k] = np.asarray(brute[k])[mask_top_h_brute]

    # Plot h values present in either MSM or brute data.
    if h.size > 0:
        unique_h = _unique_sorted(np.concatenate([h, brute["h"]]))
    else:
        unique_h = _unique_sorted(brute["h"])

    # Output directory
    out_dir = Path(config.paths.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out

    # Plot
    fig_w, fig_h = float(args.figsize[0]), float(args.figsize[1])
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    cmap = plt.get_cmap("tab10")
    rc_style = {
        "cnn": {"fmt": "o", "linestyle": "-", "markersize": 7},
        "lcs": {"fmt": "^", "linestyle": "--", "markersize": 6.5},
    }
    brute_annotations: list[tuple[float, float, str, tuple[float, float, float, float], int]] = []

    # Create color mapping based on beta values
    unique_beta_all = _unique_sorted(np.concatenate([beta, brute["beta"]]))
    beta_to_color = {}
    for idx, b in enumerate(unique_beta_all):
        beta_to_color[b] = cmap(idx % 10)

    for idx, h_val in enumerate(unique_h):

        for rc_val in rc_to_plot:
            if rc_val not in set(unique_rc.tolist()):
                continue

            mask = (h == h_val) & (rc == rc_val)
            if not np.any(mask):
                continue

            # Get unique betas for this (h, rc) combination
            betas_for_hrc = np.unique(beta[mask])
            
            for b_val in betas_for_hrc:
                # Plot only points with this beta
                mask_b = mask & np.isclose(beta, b_val, atol=5e-4)
                
                beta_h = beta[mask_b]
                J_h = J[mask_b]
                J_std_h = J_std[mask_b] if J_std is not None else None
                J_ci_low_h = J_ci_low[mask_b] if J_ci_low is not None else None
                J_ci_high_h = J_ci_high[mask_b] if J_ci_high is not None else None

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
                    J_ci_low_h is not None
                    and J_ci_high_h is not None
                    and np.all(np.isfinite(J_ci_low_h))
                    and np.all(np.isfinite(J_ci_high_h))
                ):
                    yerr_msm = _asym_yerr(J_h, J_ci_low_h, J_ci_high_h)
                elif J_std_h is not None and np.any(np.isfinite(J_std_h)):
                    yerr_msm = np.where(np.isfinite(J_std_h), J_std_h, 0.0)

                c = beta_to_color[b_val]
                style = rc_style.get(rc_val, {"fmt": "o", "linestyle": "-", "markersize": 7})
                plt.errorbar(
                    beta_h,
                    J_h,
                    yerr=yerr_msm,
                    fmt=style["fmt"],
                    markersize=style["markersize"],
                    color=c,
                    ecolor=c,
                    elinewidth=1.2,
                    capsize=2,
                    capthick=1.0,
                    alpha=1.0,
                    zorder=3,
                )

        # Brute-force overlay for this h (always)
        mask_b = np.isclose(brute["h"], float(h_val), atol=5e-4)
        if np.any(mask_b):
            # Get unique betas for brute-force at this h
            brute_betas = np.unique(brute["beta"][mask_b])
            
            for b_val in brute_betas:
                # Plot only brute-force points with this beta
                mask_b_beta = mask_b & np.isclose(brute["beta"], b_val, atol=5e-4)
                
                beta_b = brute["beta"][mask_b_beta]
                J_b = brute["rate_per_site"][mask_b_beta]
                se_b = brute["rate_per_site_se"][mask_b_beta]
                cil_b = brute["rate_per_site_ci_low"][mask_b_beta]
                cih_b = brute["rate_per_site_ci_high"][mask_b_beta]
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

                c = beta_to_color[b_val]
                plt.errorbar(
                    beta_b,
                    J_b,
                    yerr=yerr_brute,
                    fmt="x",
                    markersize=8,
                    color=c,
                    ecolor=c,
                    elinewidth=0.9,
                    capsize=2,
                    capthick=0.9,
                    alpha=0.45,
                    zorder=2,
                )

                # Annotate each brute-force point with its field value.
                h_text = f"h={float(h_val):.3f}"
                for j, (xb, yb) in enumerate(zip(beta_b.tolist(), J_b.tolist())):
                    brute_annotations.append((float(xb), float(yb), h_text, c, j))

    plt.yscale("log")
    plt.ylabel(r"I MCSS$^{-1}$")
    plt.xlabel(r"$\beta$")
    ax.margins(x=0.08, y=0.18)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    log_y_min = np.log10(y_min)
    log_y_max = np.log10(y_max)
    log_y_span = max(log_y_max - log_y_min, 1e-12)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer=renderer)
    placed_label_bboxes = []

    for xb, yb, h_text, c, j in brute_annotations:
        x_frac = (xb - x_min) / max(x_max - x_min, 1e-12)
        y_frac = (np.log10(yb) - log_y_min) / log_y_span

        place_left = x_frac > 0.82
        place_below = y_frac > 0.82
        if y_frac < 0.18:
            place_below = False

        preferred_dx = -6 if place_left else 6
        preferred_dy = -10 if place_below else 6
        if 0.18 <= y_frac <= 0.82 and j % 2 == 1:
            preferred_dy = -10

        candidate_offsets = [
            (preferred_dx, preferred_dy),
            (preferred_dx, -preferred_dy),
            (-preferred_dx, preferred_dy),
            (-preferred_dx, -preferred_dy),
            (1.7 * preferred_dx, preferred_dy),
            (1.7 * preferred_dx, -preferred_dy),
            (-1.7 * preferred_dx, preferred_dy),
            (-1.7 * preferred_dx, -preferred_dy),
            (preferred_dx, 1.7 * preferred_dy),
            (preferred_dx, -1.7 * preferred_dy),
        ]

        t = None
        selected_bbox = None
        for dx, dy in candidate_offsets:
            candidate = ax.annotate(
                h_text,
                xy=(xb, yb),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=("right" if dx < 0 else "left"),
                va=("top" if dy < 0 else "bottom"),
                color=c,
                fontsize=FONT_SIZE * 0.85,
                alpha=0.9,
                annotation_clip=True,
                clip_on=True,
            )
            fig.canvas.draw()
            bbox = candidate.get_window_extent(renderer=renderer).expanded(1.08, 1.15)
            inside_axes = (
                bbox.x0 >= axes_bbox.x0
                and bbox.x1 <= axes_bbox.x1
                and bbox.y0 >= axes_bbox.y0
                and bbox.y1 <= axes_bbox.y1
            )
            overlaps_existing = any(bbox.overlaps(prev_bbox) for prev_bbox in placed_label_bboxes)
            if inside_axes and not overlaps_existing:
                t = candidate
                selected_bbox = bbox
                break
            candidate.remove()

        if t is None:
            dx, dy = candidate_offsets[0]
            t = ax.annotate(
                h_text,
                xy=(xb, yb),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=("right" if dx < 0 else "left"),
                va=("top" if dy < 0 else "bottom"),
                color=c,
                fontsize=FONT_SIZE * 0.85,
                alpha=0.9,
                annotation_clip=True,
                clip_on=True,
            )
            fig.canvas.draw()
            selected_bbox = t.get_window_extent(renderer=renderer).expanded(1.08, 1.15)

        try:
            t.set_underline(True)
        except Exception:
            pass
        if selected_bbox is not None:
            placed_label_bboxes.append(selected_bbox)

    # Style legend (marker/linestyle meanings)
    style_handles: list[Line2D] = []
    if "cnn" in set(unique_rc.tolist()) and "cnn" in rc_to_plot:
        style_handles.append(
            Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=7, label="MSM (CNN)")
        )
    if "lcs" in set(unique_rc.tolist()) and "lcs" in rc_to_plot:
        style_handles.append(
            Line2D(
                [0], [0], color="black", marker="^", linestyle="None", markersize=6.5, label="MSM (LCS)"
            )
        )
    style_handles.append(
        Line2D([0], [0], color="black", marker="x", linestyle="None", markersize=8, label="Brute force")
    )

    ax.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(style_handles),
    )
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
