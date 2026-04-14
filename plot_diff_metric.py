from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from utils.architecture import CNN
from utils.dataset import prepare_subset, prepare_datasets
from utils.config import load_config


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
    "brown": "#8C564B",
    "tan": "#C49C94",
    "orchid": "#E377C2",
    "light_orchid": "#F7B6D2",
    "gray": "#7F7F7F",
    "light_gray": "#C7C7C7",
    "yellow_green": "#BCBD22",
    "light_yellow_green": "#DBDB8D",
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


RUNS = [
    (0.511, 0.040),
    (0.526, 0.050),
    (0.538, 0.060),
    (0.550, 0.070),
    (0.564, 0.080),
    (0.576, 0.090),
    (0.588, 0.100),
]

RUNS_NEW = [
    (0.511, 0.0580),
    (0.526, 0.0700),
    (0.538, 0.0820),
    (0.550, 0.0930),
    (0.564, 0.1050),
    (0.576, 0.1160),
    (0.588, 0.1280),
]


def _load_model_from_checkpoint(checkpoint_path: str | Path, device: str) -> CNN:
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model = CNN(
        channels=checkpoint["channels"],
        num_cnn_layers=checkpoint["num_cnn_layers"],
        num_fc_layers=checkpoint["num_fc_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _predict_on_dataset(model: torch.nn.Module, dataset, device: str, chunk_size: int = 1024) -> np.ndarray:
    preds: list[np.ndarray] = []
    for start in range(0, len(dataset), chunk_size):
        end = min(start + chunk_size, len(dataset))
        batch = torch.stack([dataset[j][0] for j in range(start, end)]).to(device)
        with torch.no_grad():
            batch_preds = model(batch).squeeze().detach().cpu().numpy()
        if np.ndim(batch_preds) == 0:
            batch_preds = np.array([batch_preds])
        preds.append(np.asarray(batch_preds))
    return np.concatenate(preds, axis=0)


def sigmoid(x: np.ndarray, k: float, x0: float) -> np.ndarray:
    """Sigmoid used for cluster-size committor baseline (matches generate_plots.py)."""

    base = 1.0 / (1.0 + np.exp(-k * (x - x0)))
    base0 = 1.0 / (1.0 + np.exp(k * x0))
    return (base - base0) / (1.0 - base0 + 1e-10)


def fit_sigmoid(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        try:
            popt, _ = curve_fit(sigmoid, x, y, p0=[0.1, float(np.median(x))], maxfev=10000)
            return float(popt[0]), float(popt[1])
        except RuntimeError as e:
            print(f"Sigmoid fit failed: {e}, using default parameters")
            return 0.1, float(np.median(x))


def fraction_outside_band(y_true: np.ndarray, y_pred: np.ndarray, percent: float, mode: str, eps: float = 1e-8) -> float:
    """Return fraction of points outside a band around y=x.

    mode:
      - "relative_pred": |y_pred - y_true| > (percent/100)*max(|y_pred|, eps)
      - "absolute":      |y_pred - y_true| > (percent/100)
    """

    diff = np.abs(y_pred - y_true)

    if mode == "relative_pred":
        band = (percent / 100.0) * np.maximum(np.abs(y_pred), eps)
    elif mode == "absolute":
        band = np.full_like(diff, fill_value=(percent / 100.0), dtype=float)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return float(np.mean(diff > band))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--percent", type=float, default=2.5, help="Band size in percent")
    parser.add_argument(
        "--mode",
        type=str,
        default="absolute",
        choices=["relative_pred", "absolute"],
        help="Band definition; default matches 'percent of the prediction'",
    )
    parser.add_argument("--chunk-size", type=int, default=1024, help="Inference chunk size")
    parser.add_argument("--out", type=str, default="diff_metric_vs_beta.pdf", help="Output filename")
    args = parser.parse_args()

    if not RUNS:
        raise ValueError("RUNS is empty. Fill RUNS with the (beta, h) values you want to evaluate.")

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fractions_cnn: list[float] = []
    fractions_cluster: list[float] = []
    betas: list[float] = []
    
    fractions_cnn_new: list[float] = []
    fractions_cluster_new: list[float] = []
    betas_new: list[float] = []

    for run in RUNS:
        beta = float(run[0])
        h = float(run[1])

        # Dataset (uses the same naming convention as generate_plots.py)
        h5path = f"../data/gridstates_training_{beta:.3f}_{h:.3f}.hdf5"
        print(f"Loading dataset from {h5path}...")

        grids, attrs, train_idx, valid_idx, test_idx = prepare_subset(h5path, test_size=config.dataset.test_size)
        _, _, _, _, _, test_ds = prepare_datasets(
            grids,
            attrs,
            train_idx,
            valid_idx,
            test_idx,
            device,
            config.dataset.batch_size,
            augment=False,
        )

        # Model checkpoint (uses the same naming convention as generate_plots.py)
        checkpoint_path = (
            f"{config.paths.save_dir}/"
            f"{config.model.type}_ch{config.model.channels}_cn{config.model.num_cnn_layers}_fc{config.model.num_fc_layers}_"
            f"{beta:.3f}_{h:.3f}.pth"
        )
        print(f"Loading model from {checkpoint_path}...")
        model = _load_model_from_checkpoint(checkpoint_path, device=device)

        # Targets and predictions on TEST
        y_true = np.array([test_ds[i][1].item() for i in range(len(test_ds))], dtype=float)
        y_pred = _predict_on_dataset(model, test_ds, device=device, chunk_size=int(args.chunk_size))

        # Cluster-size baseline on TEST (fit sigmoid to (cluster_size -> committor) on the test set)
        test_cluster_size = np.asarray(attrs[test_idx, 1], dtype=float)
        k, x0 = fit_sigmoid(test_cluster_size, y_true)
        y_pred_cluster = sigmoid(test_cluster_size, k, x0)

        frac_cnn = fraction_outside_band(y_true=y_true, y_pred=y_pred, percent=float(args.percent), mode=str(args.mode))
        frac_cluster = fraction_outside_band(
            y_true=y_true,
            y_pred=y_pred_cluster,
            percent=float(args.percent),
            mode=str(args.mode),
        )
        print(
            f"beta={beta:.3f}, h={h:.3f}: outside ±{args.percent}% band ({args.mode}) "
            f"CNN={frac_cnn:.4f} | Cluster={frac_cluster:.4f}"
        )

        betas.append(beta)
        fractions_cnn.append(frac_cnn)
        fractions_cluster.append(frac_cluster)

    for run in RUNS_NEW:
        beta = float(run[0])
        h = float(run[1])

        # Dataset (uses the same naming convention as generate_plots.py)
        h5path = f"../data/gridstates_training_{beta:.3f}_{h:.3f}.hdf5"
        print(f"Loading dataset from {h5path}...")

        grids, attrs, train_idx, valid_idx, test_idx = prepare_subset(h5path, test_size=config.dataset.test_size)
        _, _, _, _, _, test_ds = prepare_datasets(
            grids,
            attrs,
            train_idx,
            valid_idx,
            test_idx,
            device,
            config.dataset.batch_size,
            augment=False,
        )

        # Model checkpoint (uses the same naming convention as generate_plots.py)
        checkpoint_path = (
            f"{config.paths.save_dir}/"
            f"{config.model.type}_ch{config.model.channels}_cn{config.model.num_cnn_layers}_fc{config.model.num_fc_layers}_"
            f"{beta:.3f}_{h:.3f}.pth"
        )
        print(f"Loading model from {checkpoint_path}...")
        model = _load_model_from_checkpoint(checkpoint_path, device=device)

        # Targets and predictions on TEST
        y_true = np.array([test_ds[i][1].item() for i in range(len(test_ds))], dtype=float)
        y_pred = _predict_on_dataset(model, test_ds, device=device, chunk_size=int(args.chunk_size))

        # Cluster-size baseline on TEST (fit sigmoid to (cluster_size -> committor) on the test set)
        test_cluster_size = np.asarray(attrs[test_idx, 1], dtype=float)
        k, x0 = fit_sigmoid(test_cluster_size, y_true)
        y_pred_cluster = sigmoid(test_cluster_size, k, x0)

        frac_cnn = fraction_outside_band(y_true=y_true, y_pred=y_pred, percent=float(args.percent), mode=str(args.mode))
        frac_cluster = fraction_outside_band(
            y_true=y_true,
            y_pred=y_pred_cluster,
            percent=float(args.percent),
            mode=str(args.mode),
        )
        print(
            f"beta={beta:.3f}, h={h:.3f}: outside ±{args.percent}% band ({args.mode}) "
            f"CNN={frac_cnn:.4f} | Cluster={frac_cluster:.4f}"
        )

        betas_new.append(beta)
        fractions_cnn_new.append(frac_cnn)
        fractions_cluster_new.append(frac_cluster)
    
    # Plot
    out_dir = Path(config.paths.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out

    # Sort by beta for nicer lines
    order = np.argsort(np.asarray(betas, dtype=float))
    betas_sorted = np.asarray(betas, dtype=float)[order]
    cnn_sorted = np.asarray(fractions_cnn, dtype=float)[order]
    cluster_sorted = np.asarray(fractions_cluster, dtype=float)[order]
    
    order_new = np.argsort(np.asarray(betas_new, dtype=float))
    betas_sorted_new = np.asarray(betas_new, dtype=float)[order_new]
    cnn_sorted_new = np.asarray(fractions_cnn_new, dtype=float)[order_new]
    cluster_sorted_new = np.asarray(fractions_cluster_new, dtype=float)[order_new]

    plt.figure(figsize=(7, 5))
    
    # Original runs
    plt.scatter(betas_sorted, cluster_sorted, s=60, color=colors["orange"], alpha=0.9, label="Cluster (Low)")
    plt.plot(betas_sorted, cluster_sorted, color=colors["light_orange"], linewidth=2, alpha=0.8)
    plt.scatter(betas_sorted, cnn_sorted, s=60, color=colors["steel_blue"], alpha=0.9, label="CNN (Low)")
    plt.plot(betas_sorted, cnn_sorted, color=colors["light_steel_blue"], linewidth=2, alpha=0.8)
    
    # New runs
    plt.scatter(betas_sorted_new, cluster_sorted_new, s=60, color=colors["orange"], marker="s", alpha=0.6, label="Cluster (High)")
    plt.plot(betas_sorted_new, cluster_sorted_new, color=colors["light_orange"], linewidth=2, alpha=0.5, linestyle="--")
    plt.scatter(betas_sorted_new, cnn_sorted_new, s=60, color=colors["steel_blue"], marker="s", alpha=0.6, label="CNN (High)")
    plt.plot(betas_sorted_new, cnn_sorted_new, color=colors["light_steel_blue"], linewidth=2, alpha=0.5, linestyle="--")

    plt.xlabel(r"$\beta$")
    plt.ylabel(f"Fraction outside ±{args.percent}% band")
    #plt.title("Out-of-band proportion vs beta (test set)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
