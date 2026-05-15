#!/usr/bin/env python
"""Plot deeptime implied-timescale diagnostics for q-NN and LCS."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from deeptime.markov.msm import MarkovStateModel

from markov_deeptime import (
    available_count_lags,
    choose_coarse_graining,
    coarse_grain_first_n_states,
    count_to_transition,
    load_count_matrix,
)


mpl.rcParams["svg.fonttype"] = "none"

BETA = 0.550
H_FIELD = 0.070
N_TIMESCALES = 3
RETENTION_THRESHOLD = 0.5
OUT_PATH = Path("figures") / "its_diagnostic_0.550_0.070.svg"


def load_selected_lag(beta: float, h: float, rc: str) -> int:
    path = Path("data") / f"msm_analysis_{beta:.3f}_{h:.3f}_{rc}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing MSM analysis file: {path}")
    with np.load(path) as data:
        return int(np.asarray(data["selected_m"]).item())


def compute_implied_timescales(beta: float, h: float, rc: str) -> tuple[np.ndarray, np.ndarray]:
    lags = np.asarray(available_count_lags(beta, h, rc), dtype=int)
    timescales = []

    for lag in lags:
        counts = load_count_matrix(beta, h, rc, int(lag))
        coarse_choice = choose_coarse_graining(counts, retention_threshold=RETENTION_THRESHOLD)
        coarse_counts = coarse_grain_first_n_states(counts, coarse_choice["n_combine"])
        transition = count_to_transition(coarse_counts, normalize_rows=True)
        msm = MarkovStateModel(transition_matrix=transition, lagtime=int(lag), reversible=False)
        values = np.asarray(msm.timescales()[:N_TIMESCALES], dtype=float)

        if values.size < N_TIMESCALES:
            padded = np.full(N_TIMESCALES, np.nan, dtype=float)
            padded[: values.size] = values
            values = padded
        timescales.append(values)

    return lags, np.asarray(timescales, dtype=float)


def plot_panel(ax: plt.Axes, rc: str, title: str) -> None:
    lags, timescales = compute_implied_timescales(BETA, H_FIELD, rc)
    selected_lag = load_selected_lag(BETA, H_FIELD, rc)
    selected_idx = int(np.where(lags == selected_lag)[0][0])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx in range(N_TIMESCALES):
        ax.plot(
            lags,
            timescales[:, idx],
            marker="o",
            ms=4,
            lw=1.4,
            color=colors[idx],
            label=fr"$t_{idx + 1}$",
        )

    selected_y = timescales[selected_idx, 0]
    ax.axvline(selected_lag, color="0.55", lw=1.0, ls="--", zorder=1)
    ax.scatter(
        [selected_lag],
        [selected_y],
        s=150,
        facecolors="none",
        edgecolors="black",
        linewidths=1.6,
        zorder=5,
    )

    ax.set_title(title)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"Lag time, $\tau$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)

    plot_panel(axes[0], "cnn", r"$p_B$-NN")
    plot_panel(axes[1], "lcs", "LGCS")
    axes[0].set_ylabel("Implied timescale")
    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ITS diagnostic plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
