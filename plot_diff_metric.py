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


def load_runs_from_csv(csv_path: str | Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
    """Load (beta, h) pairs from CSV and group by lowest/middle/highest h per beta.
    
    Expected CSV format:
      h,beta,dn,up
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Parse CSV manually to avoid dependencies
    runs_lowest = []
    runs_middle = []
    runs_highest = []
    
    # Group by beta
    beta_to_h = {}
    
    with open(csv_path, 'r') as f:
        header = f.readline().strip()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                h = float(parts[0])
                beta = float(parts[1])
                if beta not in beta_to_h:
                    beta_to_h[beta] = []
                beta_to_h[beta].append(h)
            except ValueError:
                continue
    
    # For each beta, sort h values and classify
    for beta in sorted(beta_to_h.keys()):
        h_values = sorted(beta_to_h[beta])
        
        if len(h_values) >= 1:
            runs_lowest.append((beta, h_values[0]))
        if len(h_values) >= 2:
            runs_middle.append((beta, h_values[len(h_values) // 2]))
        if len(h_values) >= 3:
            runs_highest.append((beta, h_values[-1]))
    
    return runs_lowest, runs_middle, runs_highest


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
    parser.add_argument("--csv", type=str, default="h_beta_dn_up.csv", help="Path to CSV file with h,beta,dn,up columns")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Load runs from CSV
    RUNS, RUNS_NEW, RUNS_NEWER = load_runs_from_csv(args.csv)
    
    if not RUNS:
        raise ValueError("No runs loaded from CSV. Check CSV file format.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fractions_cnn: list[float] = []
    fractions_cluster: list[float] = []
    betas: list[float] = []
    
    fractions_cnn_new: list[float] = []
    fractions_cluster_new: list[float] = []
    betas_new: list[float] = []

    fractions_cnn_newer: list[float] = []
    fractions_cluster_newer: list[float] = []
    betas_newer: list[float] = []
    
    # Track all predictions for potential second plot
    all_predictions: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    worst_cnn_fraction = -1.0
    worst_cnn_beta = None
    worst_cnn_h = None
    worst_cluster_fraction = -1.0
    worst_cluster_beta = None
    worst_cluster_h = None

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
        all_predictions[(beta, h)] = (y_true.copy(), y_pred.copy(), y_pred_cluster.copy())
        
        # Track worst CNN and cluster fractions separately
        if frac_cnn > worst_cnn_fraction:
            worst_cnn_fraction = frac_cnn
            worst_cnn_beta = beta
            worst_cnn_h = h
        if frac_cluster > worst_cluster_fraction:
            worst_cluster_fraction = frac_cluster
            worst_cluster_beta = beta
            worst_cluster_h = h

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
        all_predictions[(beta, h)] = (y_true.copy(), y_pred.copy(), y_pred_cluster.copy())
        
        # Track worst CNN and cluster fractions separately
        if frac_cnn > worst_cnn_fraction:
            worst_cnn_fraction = frac_cnn
            worst_cnn_beta = beta
            worst_cnn_h = h
        if frac_cluster > worst_cluster_fraction:
            worst_cluster_fraction = frac_cluster
            worst_cluster_beta = beta
            worst_cluster_h = h

    for run in RUNS_NEWER:
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

        betas_newer.append(beta)
        fractions_cnn_newer.append(frac_cnn)
        fractions_cluster_newer.append(frac_cluster)
        all_predictions[(beta, h)] = (y_true.copy(), y_pred.copy(), y_pred_cluster.copy())
        
        # Track worst CNN and cluster fractions separately
        if frac_cnn > worst_cnn_fraction:
            worst_cnn_fraction = frac_cnn
            worst_cnn_beta = beta
            worst_cnn_h = h
        if frac_cluster > worst_cluster_fraction:
            worst_cluster_fraction = frac_cluster
            worst_cluster_beta = beta
            worst_cluster_h = h
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

    order_newer = np.argsort(np.asarray(betas_newer, dtype=float))
    betas_sorted_newer = np.asarray(betas_newer, dtype=float)[order_newer]
    cnn_sorted_newer = np.asarray(fractions_cnn_newer, dtype=float)[order_newer]
    cluster_sorted_newer = np.asarray(fractions_cluster_newer, dtype=float)[order_newer]

    plt.figure(figsize=(7.5, 5))
    ax = plt.gca()
    
    # Lowest h runs (blue)
    ax.plot(betas_sorted, cluster_sorted, color=colors["light_steel_blue"], linewidth=2, alpha=0.8, zorder=1)
    ax.scatter(betas_sorted, cluster_sorted, s=60, marker="o", color=colors["steel_blue"], alpha=0.9, zorder=2)
    ax.plot(betas_sorted, cnn_sorted, color=colors["light_steel_blue"], linewidth=2, alpha=0.8, zorder=1)
    ax.scatter(betas_sorted, cnn_sorted, s=60, marker="s", color=colors["steel_blue"], alpha=0.9, zorder=2)
    
    # Middle h runs (green)
    ax.plot(betas_sorted_new, cluster_sorted_new, color=colors["light_green"], linewidth=2, alpha=0.8, linestyle="--", zorder=1)
    ax.scatter(betas_sorted_new, cluster_sorted_new, s=60, marker="o", color=colors["forest_green"], alpha=0.9, zorder=2)
    ax.plot(betas_sorted_new, cnn_sorted_new, color=colors["light_green"], linewidth=2, alpha=0.8, linestyle="--", zorder=1)
    ax.scatter(betas_sorted_new, cnn_sorted_new, s=60, marker="s", color=colors["forest_green"], alpha=0.9, zorder=2)

    # Highest h runs (red)
    ax.plot(betas_sorted_newer, cluster_sorted_newer, color=colors["soft_red"], linewidth=2, alpha=0.8, linestyle=":", zorder=1)
    ax.scatter(betas_sorted_newer, cluster_sorted_newer, s=60, marker="o", color=colors["firebrick_red"], alpha=0.9, zorder=2)
    ax.plot(betas_sorted_newer, cnn_sorted_newer, color=colors["soft_red"], linewidth=2, alpha=0.8, linestyle=":", zorder=1)
    ax.scatter(betas_sorted_newer, cnn_sorted_newer, s=60, marker="s", color=colors["firebrick_red"], alpha=0.9, zorder=2)

    # Create legend 1: markers (Metric type)
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=10, label="LCS"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=10, label="q-NN"),
    ]
    legend1 = ax.legend(handles=marker_handles, loc="upper left", bbox_to_anchor=(1.05, 1))
    ax.add_artist(legend1)

    # Create legend 2: colors (Field strength)
    color_handles = [
        Line2D([0], [0], color=colors["firebrick_red"], linewidth=3, label="A"),
        Line2D([0], [0], color=colors["forest_green"], linewidth=3, label="B"),
        Line2D([0], [0], color=colors["steel_blue"], linewidth=3, label="C"),
    ]
    legend2 = ax.legend(handles=color_handles, loc="upper left", bbox_to_anchor=(1.05, 0.75))
    
    # Color the legend labels to match line colors
    for text, color in zip(legend2.get_texts()[:], [colors["firebrick_red"], colors["forest_green"], colors["steel_blue"]]):
        text.set_color(color)

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(f"Fraction outside ±{args.percent}% band")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {out_path}")
    
    # Helper function to create scatter plot
    def create_scatter_plot(y_true, y_pred, metric_label, beta, h, fraction_outside, args, out_dir, config):
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, color=colors["steel_blue"], zorder=1)
        
        # Diagonal line (perfect prediction, red, from 0 to 1)
        ax.plot([0, 1], [0, 1], "-", color=colors["firebrick_red"], linewidth=2, alpha=0.7, zorder=3)
        
        # Band lines (±2.5%)
        band_width = float(args.percent) / 100.0
        upper_band = np.array([y_true - band_width, y_true + band_width])
        lower_band = np.array([y_true + band_width, y_true - band_width])
        
        # Create smoother band visualization
        sorted_indices = np.argsort(y_true)
        y_true_sorted = y_true[sorted_indices]
        upper_bound = y_true_sorted + band_width
        lower_bound = y_true_sorted - band_width
        
        ax.plot(y_true_sorted, upper_bound, "--", color=colors["firebrick_red"], linewidth=2, alpha=0.7, label=f"±{args.percent}% band", zorder=3)
        ax.plot(y_true_sorted, lower_bound, "--", color=colors["firebrick_red"], linewidth=2, alpha=0.7, zorder=3)
        ax.fill_between(y_true_sorted, lower_bound, upper_bound, color=colors["firebrick_red"], alpha=0.1, zorder=3)
        
        ax.set_xlabel("Target Committor")
        ax.set_ylabel(f"Predicted Committor")
        ax.set_title(f"β={beta:.3f}, h={h:.3f}\n(Fraction outside band: {fraction_outside:.4f})")
        ax.grid(True, alpha=0.2)
        ax.legend()
        ax.set_aspect("equal")
        
        return plt.gcf()
    
    # Create scatter plot for worst CNN
    if worst_cnn_beta is not None and worst_cnn_h is not None:
        y_true_worst_cnn, y_pred_cnn_worst, _ = all_predictions[(worst_cnn_beta, worst_cnn_h)]
        create_scatter_plot(y_true_worst_cnn, y_pred_cnn_worst, "q-NN", worst_cnn_beta, worst_cnn_h, worst_cnn_fraction, args, out_dir, config)
        worst_cnn_path = out_dir / f"committor_scatter_cnn_{worst_cnn_beta:.3f}_{worst_cnn_h:.3f}.pdf"
        plt.savefig(worst_cnn_path, bbox_inches='tight')
        plt.close()
        print(f"Saved worst CNN scatter plot to {worst_cnn_path}")
    
    # Create scatter plot for worst cluster
    if worst_cluster_beta is not None and worst_cluster_h is not None:
        y_true_worst_cluster, _, y_pred_cluster_worst = all_predictions[(worst_cluster_beta, worst_cluster_h)]
        create_scatter_plot(y_true_worst_cluster, y_pred_cluster_worst, "Cluster Size", worst_cluster_beta, worst_cluster_h, worst_cluster_fraction, args, out_dir, config)
        worst_cluster_path = out_dir / f"committor_scatter_cluster_{worst_cluster_beta:.3f}_{worst_cluster_h:.3f}.pdf"
        plt.savefig(worst_cluster_path, bbox_inches='tight')
        plt.close()
        print(f"Saved worst cluster scatter plot to {worst_cluster_path}")


if __name__ == "__main__":
    main()
