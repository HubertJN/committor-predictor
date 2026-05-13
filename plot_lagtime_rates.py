from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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
mpl.rcParams["svg.fonttype"] = "none"


COLORS = {
    "cnn": "#1F77B4",
    "lcs": "#2CA02C",
    "brute": "#D62728",
}


def _as_scalar(value, dtype=float):
    return np.asarray(value, dtype=dtype).ravel()[0]


def _asym_yerr(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    lo = np.maximum(0.0, y - low)
    hi = np.maximum(0.0, high - y)
    return np.vstack([lo, hi])


def load_msm_lag_rates(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MSM file not found: {path}")

    data = np.load(str(path), allow_pickle=True)
    required = {"m", "J_central", "J_ci_low", "J_ci_high"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")

    m = np.asarray(data["m"], dtype=float)
    rate = np.asarray(data["J_central"], dtype=float)
    ci_low = np.asarray(data["J_ci_low"], dtype=float)
    ci_high = np.asarray(data["J_ci_high"], dtype=float)

    selected = np.zeros(m.shape, dtype=bool)
    if "lag_is_selected" in data:
        selected = np.asarray(data["lag_is_selected"], dtype=bool)

    order = np.argsort(m)
    return {
        "m": m[order],
        "rate": rate[order],
        "ci_low": ci_low[order],
        "ci_high": ci_high[order],
        "selected": selected[order],
    }


def load_brute_rate(path: str | Path) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Brute-force file not found: {path}")

    data = np.load(str(path), allow_pickle=True)
    if "rate_per_site" not in data:
        raise ValueError(f"{path} is missing required key: rate_per_site")

    rate = float(_as_scalar(data["rate_per_site"]))
    ci_low = float(_as_scalar(data["rate_per_site_ci_low"])) if "rate_per_site_ci_low" in data else np.nan
    ci_high = float(_as_scalar(data["rate_per_site_ci_high"])) if "rate_per_site_ci_high" in data else np.nan
    return {"rate": rate, "ci_low": ci_low, "ci_high": ci_high}


def plot_series(ax, data: dict[str, np.ndarray], label: str, color: str, marker: str) -> None:
    yerr = None
    if np.all(np.isfinite(data["ci_low"])) and np.all(np.isfinite(data["ci_high"])):
        yerr = _asym_yerr(data["rate"], data["ci_low"], data["ci_high"])

    ax.errorbar(
        data["m"],
        data["rate"],
        yerr=yerr,
        marker=marker,
        linestyle="-",
        linewidth=1.8,
        markersize=6,
        capsize=2,
        color=color,
        ecolor=color,
        label=label,
        zorder=3,
    )

    selected = np.asarray(data["selected"], dtype=bool)
    if np.any(selected):
        ax.scatter(
            data["m"][selected],
            data["rate"][selected],
            s=130,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=2.0,
            zorder=5,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot q-NN and LCS MSM nucleation rates as a function of lag time."
    )
    parser.add_argument("--cnn-msm", required=True, help="Path to msm_analysis_..._cnn.npz")
    parser.add_argument("--lcs-msm", required=True, help="Path to msm_analysis_..._lcs.npz")
    parser.add_argument("--brute", required=True, help="Path to brute-force nucleation_...npz")
    parser.add_argument("--out", default="figures/lagtime_rates.svg", help="Output figure path")
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.5, 4.6), metavar=("W", "H"))
    parser.add_argument("--yscale", choices=["log", "linear", "auto"], default="log")
    args = parser.parse_args()

    cnn = load_msm_lag_rates(args.cnn_msm)
    lcs = load_msm_lag_rates(args.lcs_msm)
    brute = load_brute_rate(args.brute)

    fig, ax = plt.subplots(figsize=(float(args.figsize[0]), float(args.figsize[1])))

    plot_series(ax, cnn, "q-NN MSM", COLORS["cnn"], "o")
    plot_series(ax, lcs, "LCS MSM", COLORS["lcs"], "s")

    brute_rate = brute["rate"]
    if args.yscale == "auto":
        positive_rates = np.concatenate([cnn["rate"], lcs["rate"], np.array([brute_rate])])
        positive_rates = positive_rates[positive_rates > 0]
        use_log = positive_rates.size > 0 and positive_rates.max() / positive_rates.min() > 10.0
    else:
        use_log = args.yscale == "log"

    ax.axhline(brute_rate, color=COLORS["brute"], linestyle="--", linewidth=1.8, label="Brute force")
    if np.isfinite(brute["ci_low"]) and np.isfinite(brute["ci_high"]):
        span_low = brute["ci_low"]
        if use_log and span_low <= 0.0:
            positive_rates = np.concatenate([cnn["rate"], lcs["rate"], np.array([brute_rate, brute["ci_high"]])])
            positive_rates = positive_rates[positive_rates > 0]
            span_low = float(positive_rates.min() * 0.5) if positive_rates.size else np.nan
        if np.isfinite(span_low) and brute["ci_high"] > span_low:
            ax.axhspan(
                span_low,
                brute["ci_high"],
                color=COLORS["brute"],
                alpha=0.12,
                linewidth=0,
                zorder=1,
            )

    ax.set_xlabel("Lag time / MC sweeps")
    ax.set_ylabel(r"Nucleation rate per spin / MCSS$^{-1}$")
    ax.set_xscale("log", base=2)

    if use_log:
        ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lag-time rate plot to {out_path}")


if __name__ == "__main__":
    main()
