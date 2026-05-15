from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

from utils.architecture import CNN
from utils.clusters import fk_largest_cluster_sizes_up
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

mpl.rcParams["svg.fonttype"] = "none"

CACHE_SCHEMA_VERSION = 3
GROUP_NAMES = ("low_h", "mid_h", "high_h")
METHODS = ("cnn", "lcs", "fk")


def lighten_color(color: str, amount: float = 0.25):
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb + (1.0 - rgb) * amount)


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


def _build_cache_payload(
    *,
    betas: list[float],
    fractions_cnn: list[float],
    fractions_cluster: list[float],
    fractions_fk: list[float],
    betas_new: list[float],
    fractions_cnn_new: list[float],
    fractions_cluster_new: list[float],
    fractions_fk_new: list[float],
    betas_newer: list[float],
    fractions_cnn_newer: list[float],
    fractions_cluster_newer: list[float],
    fractions_fk_newer: list[float],
    worst_cnn_fraction: float,
    worst_cnn_beta: float | None,
    worst_cnn_h: float | None,
    worst_cluster_fraction: float,
    worst_cluster_beta: float | None,
    worst_cluster_h: float | None,
    worst_fk_fraction: float,
    worst_fk_beta: float | None,
    worst_fk_h: float | None,
    fk_draws: int,
    fk_aggregate: str,
    all_predictions: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict:
    groups = [
        {
            "name": GROUP_NAMES[0],
            "betas": list(betas),
            "fractions": {
                "cnn": list(fractions_cnn),
                "lcs": list(fractions_cluster),
                "fk": list(fractions_fk),
            },
        },
        {
            "name": GROUP_NAMES[1],
            "betas": list(betas_new),
            "fractions": {
                "cnn": list(fractions_cnn_new),
                "lcs": list(fractions_cluster_new),
                "fk": list(fractions_fk_new),
            },
        },
        {
            "name": GROUP_NAMES[2],
            "betas": list(betas_newer),
            "fractions": {
                "cnn": list(fractions_cnn_newer),
                "lcs": list(fractions_cluster_newer),
                "fk": list(fractions_fk_newer),
            },
        },
    ]
    worst_cases = {
        "cnn": {"fraction": float(worst_cnn_fraction), "beta": None if worst_cnn_beta is None else float(worst_cnn_beta), "h": None if worst_cnn_h is None else float(worst_cnn_h)},
        "lcs": {"fraction": float(worst_cluster_fraction), "beta": None if worst_cluster_beta is None else float(worst_cluster_beta), "h": None if worst_cluster_h is None else float(worst_cluster_h)},
        "fk": {"fraction": float(worst_fk_fraction), "beta": None if worst_fk_beta is None else float(worst_fk_beta), "h": None if worst_fk_h is None else float(worst_fk_h)},
    }
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fk_draws": int(fk_draws),
        "fk_aggregate": str(fk_aggregate),
        "groups": groups,
        "worst_cases": worst_cases,
        "predictions": all_predictions,
    }


def _save_data_cache(cache_path: str | Path, data: dict) -> None:
    """Save collected data to NPZ cache file."""
    cache_path = Path(cache_path)

    groups = data["groups"]
    worst_cases = data["worst_cases"]
    all_preds = data["predictions"]
    fk_draws = int(data["fk_draws"])
    fk_aggregate = str(data["fk_aggregate"])

    # Prepare prediction records for NPZ storage.
    pred_keys = []
    pred_y_true = []
    pred_y_pred_cnn = []
    pred_y_pred_lcs = []
    pred_y_pred_fk = []

    for (beta, h), (y_true, y_pred_cnn, y_pred_lcs, y_pred_fk) in all_preds.items():
        pred_keys.append([beta, h])
        pred_y_true.append(y_true)
        pred_y_pred_cnn.append(y_pred_cnn)
        pred_y_pred_lcs.append(y_pred_lcs)
        pred_y_pred_fk.append(y_pred_fk)

    group_names = np.asarray([g["name"] for g in groups], dtype=object)
    group_betas = np.asarray([np.asarray(g["betas"], dtype=float) for g in groups], dtype=object)
    group_frac_cnn = np.asarray([np.asarray(g["fractions"]["cnn"], dtype=float) for g in groups], dtype=object)
    group_frac_lcs = np.asarray([np.asarray(g["fractions"]["lcs"], dtype=float) for g in groups], dtype=object)
    group_frac_fk = np.asarray([np.asarray(g["fractions"]["fk"], dtype=float) for g in groups], dtype=object)

    worst_methods = np.asarray(METHODS, dtype=object)
    worst_fraction = np.asarray([float(worst_cases[m]["fraction"]) for m in METHODS], dtype=float)
    worst_beta = np.asarray(
        [np.nan if worst_cases[m]["beta"] is None else float(worst_cases[m]["beta"]) for m in METHODS],
        dtype=float,
    )
    worst_h = np.asarray(
        [np.nan if worst_cases[m]["h"] is None else float(worst_cases[m]["h"]) for m in METHODS],
        dtype=float,
    )

    np.savez_compressed(
        str(cache_path),
        cache_schema_version=np.asarray([CACHE_SCHEMA_VERSION], dtype=np.int64),
        fk_draws=np.asarray([fk_draws], dtype=np.int64),
        fk_aggregate=np.asarray([fk_aggregate], dtype=object),
        group_names=group_names,
        group_betas=group_betas,
        group_frac_cnn=group_frac_cnn,
        group_frac_lcs=group_frac_lcs,
        group_frac_fk=group_frac_fk,
        worst_methods=worst_methods,
        worst_fraction=worst_fraction,
        worst_beta=worst_beta,
        worst_h=worst_h,
        pred_keys=np.array(pred_keys),
        pred_y_true=np.array(pred_y_true, dtype=object),
        pred_y_pred_cnn=np.array(pred_y_pred_cnn, dtype=object),
        pred_y_pred_lcs=np.array(pred_y_pred_lcs, dtype=object),
        pred_y_pred_fk=np.array(pred_y_pred_fk, dtype=object),
    )
    print(f"Saved data cache to {cache_path}")


def _load_data_cache(
    cache_path: str | Path,
    expected_fk_draws: int,
    expected_fk_aggregate: str = "mean",
) -> dict | None:
    """Load collected data from NPZ cache file. Returns None if cache doesn't exist."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    
    try:
        cached = np.load(str(cache_path), allow_pickle=True)
        required_keys = {
            "cache_schema_version",
            "fk_draws",
            "fk_aggregate",
            "group_names",
            "group_betas",
            "group_frac_cnn",
            "group_frac_lcs",
            "group_frac_fk",
            "worst_methods",
            "worst_fraction",
            "worst_beta",
            "worst_h",
            "pred_y_pred_cnn",
            "pred_y_pred_lcs",
            "pred_y_pred_fk",
        }
        missing = sorted(required_keys.difference(cached.files))
        if missing:
            print(f"Cache schema mismatch; missing keys ({', '.join(missing)}). Regenerating.")
            return None

        schema_version = int(np.asarray(cached["cache_schema_version"], dtype=np.int64).ravel()[0])
        if schema_version != CACHE_SCHEMA_VERSION:
            print(f"Cache schema version {schema_version} != expected {CACHE_SCHEMA_VERSION}; regenerating.")
            return None
        fk_draws = int(np.asarray(cached["fk_draws"], dtype=np.int64).ravel()[0])
        fk_aggregate = str(np.asarray(cached["fk_aggregate"], dtype=object).ravel()[0])
        if fk_draws != int(expected_fk_draws):
            print(
                f"Cache FK draws {fk_draws} != expected {int(expected_fk_draws)}; regenerating."
            )
            return None
        if fk_aggregate != str(expected_fk_aggregate):
            print(
                f"Cache FK aggregate {fk_aggregate!r} != expected {str(expected_fk_aggregate)!r}; regenerating."
            )
            return None

        # Reconstruct predictions dict from NPZ data.
        all_predictions = {}
        for i, (beta, h) in enumerate(cached["pred_keys"]):
            key = (float(beta), float(h))
            all_predictions[key] = (
                cached["pred_y_true"][i].astype(np.float64),
                cached["pred_y_pred_cnn"][i].astype(np.float64),
                cached["pred_y_pred_lcs"][i].astype(np.float64),
                cached["pred_y_pred_fk"][i].astype(np.float64),
            )

        group_names = [str(x) for x in np.asarray(cached["group_names"], dtype=object).tolist()]
        group_betas = np.asarray(cached["group_betas"], dtype=object)
        group_frac_cnn = np.asarray(cached["group_frac_cnn"], dtype=object)
        group_frac_lcs = np.asarray(cached["group_frac_lcs"], dtype=object)
        group_frac_fk = np.asarray(cached["group_frac_fk"], dtype=object)

        groups = []
        for i, name in enumerate(group_names):
            groups.append(
                {
                    "name": name,
                    "betas": np.asarray(group_betas[i], dtype=float).tolist(),
                    "fractions": {
                        "cnn": np.asarray(group_frac_cnn[i], dtype=float).tolist(),
                        "lcs": np.asarray(group_frac_lcs[i], dtype=float).tolist(),
                        "fk": np.asarray(group_frac_fk[i], dtype=float).tolist(),
                    },
                }
            )

        worst_methods = [str(x) for x in np.asarray(cached["worst_methods"], dtype=object).tolist()]
        worst_fraction = np.asarray(cached["worst_fraction"], dtype=float)
        worst_beta = np.asarray(cached["worst_beta"], dtype=float)
        worst_h = np.asarray(cached["worst_h"], dtype=float)
        worst_cases = {}
        for i, method in enumerate(worst_methods):
            worst_cases[method] = {
                "fraction": float(worst_fraction[i]),
                "beta": None if np.isnan(worst_beta[i]) else float(worst_beta[i]),
                "h": None if np.isnan(worst_h[i]) else float(worst_h[i]),
            }

        data = {
            "schema_version": schema_version,
            "fk_draws": fk_draws,
            "fk_aggregate": fk_aggregate,
            "groups": groups,
            "worst_cases": worst_cases,
            "predictions": all_predictions,
        }
        print(f"Loaded data cache from {cache_path}")
        return data
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None


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
    parser.add_argument("--out", type=str, default="diff_metric_vs_beta.svg", help="Output filename")
    parser.add_argument("--csv", type=str, default="h_beta_dn_up.csv", help="Path to CSV file with h,beta,dn,up columns")
    parser.add_argument("--data-cache", type=str, default="diff_metric_data.npz", help="Path to data cache file")
    parser.add_argument("--fk-seed", type=int, default=12345, help="Base RNG seed for FK clusters")
    parser.add_argument("--fk-draws", type=int, default=32, help="Number of FK bond realizations per frame")
    parser.add_argument("--regenerate", action="store_true", default=False, help="Regenerate data cache instead of loading from disk")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Load runs from CSV
    RUNS, RUNS_NEW, RUNS_NEWER = load_runs_from_csv(args.csv)
    
    if not RUNS:
        raise ValueError("No runs loaded from CSV. Check CSV file format.")

    # Try to load cached data
    use_cache = not args.regenerate
    cache_path = Path(config.paths.plot_dir) / args.data_cache
    
    if use_cache:
        cached_data = _load_data_cache(cache_path, expected_fk_draws=int(args.fk_draws), expected_fk_aggregate="mean")
    else:
        cached_data = None
    
    if cached_data is not None:
        # Use cached data from structured schema.
        groups_by_name = {group["name"]: group for group in cached_data["groups"]}
        low_group = groups_by_name[GROUP_NAMES[0]]
        mid_group = groups_by_name[GROUP_NAMES[1]]
        high_group = groups_by_name[GROUP_NAMES[2]]

        betas = low_group["betas"]
        fractions_cnn = low_group["fractions"]["cnn"]
        fractions_cluster = low_group["fractions"]["lcs"]
        fractions_fk = low_group["fractions"]["fk"]

        betas_new = mid_group["betas"]
        fractions_cnn_new = mid_group["fractions"]["cnn"]
        fractions_cluster_new = mid_group["fractions"]["lcs"]
        fractions_fk_new = mid_group["fractions"]["fk"]

        betas_newer = high_group["betas"]
        fractions_cnn_newer = high_group["fractions"]["cnn"]
        fractions_cluster_newer = high_group["fractions"]["lcs"]
        fractions_fk_newer = high_group["fractions"]["fk"]

        all_predictions = cached_data["predictions"]
        worst_cnn_fraction = cached_data["worst_cases"]["cnn"]["fraction"]
        worst_cnn_beta = cached_data["worst_cases"]["cnn"]["beta"]
        worst_cnn_h = cached_data["worst_cases"]["cnn"]["h"]
        worst_cluster_fraction = cached_data["worst_cases"]["lcs"]["fraction"]
        worst_cluster_beta = cached_data["worst_cases"]["lcs"]["beta"]
        worst_cluster_h = cached_data["worst_cases"]["lcs"]["h"]
        worst_fk_fraction = cached_data["worst_cases"]["fk"]["fraction"]
        worst_fk_beta = cached_data["worst_cases"]["fk"]["beta"]
        worst_fk_h = cached_data["worst_cases"]["fk"]["h"]
    else:
        # Collect data from scratch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        fractions_cnn: list[float] = []
        fractions_cluster: list[float] = []
        fractions_fk: list[float] = []
        betas: list[float] = []
        
        fractions_cnn_new: list[float] = []
        fractions_cluster_new: list[float] = []
        fractions_fk_new: list[float] = []
        betas_new: list[float] = []

        fractions_cnn_newer: list[float] = []
        fractions_cluster_newer: list[float] = []
        fractions_fk_newer: list[float] = []
        betas_newer: list[float] = []
        
        # Track all predictions for potential second plot
        all_predictions: dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        worst_cnn_fraction = -1.0
        worst_cnn_beta = None
        worst_cnn_h = None
        worst_cluster_fraction = -1.0
        worst_cluster_beta = None
        worst_cluster_h = None
        worst_fk_fraction = -1.0
        worst_fk_beta = None
        worst_fk_h = None

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

            test_fk_size = fk_largest_cluster_sizes_up(
                grids[test_idx],
                beta=beta,
                h=h,
                seed=int(args.fk_seed),
                indices=test_idx,
                n_draws=int(args.fk_draws),
            )
            k_fk, x0_fk = fit_sigmoid(test_fk_size, y_true)
            y_pred_fk = sigmoid(test_fk_size, k_fk, x0_fk)

            frac_cnn = fraction_outside_band(y_true=y_true, y_pred=y_pred, percent=float(args.percent), mode=str(args.mode))
            frac_cluster = fraction_outside_band(
                y_true=y_true,
                y_pred=y_pred_cluster,
                percent=float(args.percent),
                mode=str(args.mode),
            )
            frac_fk = fraction_outside_band(
                y_true=y_true,
                y_pred=y_pred_fk,
                percent=float(args.percent),
                mode=str(args.mode),
            )
            print(
                f"beta={beta:.3f}, h={h:.3f}: outside ±{args.percent}% band ({args.mode}) "
                f"CNN={frac_cnn:.4f} | LCS={frac_cluster:.4f} | FK={frac_fk:.4f}"
            )

            betas.append(beta)
            fractions_cnn.append(frac_cnn)
            fractions_cluster.append(frac_cluster)
            fractions_fk.append(frac_fk)
            all_predictions[(beta, h)] = (y_true.copy(), y_pred.copy(), y_pred_cluster.copy(), y_pred_fk.copy())
            
            # Track worst CNN and cluster fractions separately
            if frac_cnn > worst_cnn_fraction:
                worst_cnn_fraction = frac_cnn
                worst_cnn_beta = beta
                worst_cnn_h = h
            if frac_cluster > worst_cluster_fraction:
                worst_cluster_fraction = frac_cluster
                worst_cluster_beta = beta
                worst_cluster_h = h
            if frac_fk > worst_fk_fraction:
                worst_fk_fraction = frac_fk
                worst_fk_beta = beta
                worst_fk_h = h

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

            test_fk_size = fk_largest_cluster_sizes_up(
                grids[test_idx],
                beta=beta,
                h=h,
                seed=int(args.fk_seed),
                indices=test_idx,
                n_draws=int(args.fk_draws),
            )
            k_fk, x0_fk = fit_sigmoid(test_fk_size, y_true)
            y_pred_fk = sigmoid(test_fk_size, k_fk, x0_fk)

            frac_cnn = fraction_outside_band(y_true=y_true, y_pred=y_pred, percent=float(args.percent), mode=str(args.mode))
            frac_cluster = fraction_outside_band(
                y_true=y_true,
                y_pred=y_pred_cluster,
                percent=float(args.percent),
                mode=str(args.mode),
            )
            frac_fk = fraction_outside_band(
                y_true=y_true,
                y_pred=y_pred_fk,
                percent=float(args.percent),
                mode=str(args.mode),
            )
            print(
                f"beta={beta:.3f}, h={h:.3f}: outside ±{args.percent}% band ({args.mode}) "
                f"CNN={frac_cnn:.4f} | LCS={frac_cluster:.4f} | FK={frac_fk:.4f}"
            )

            betas_new.append(beta)
            fractions_cnn_new.append(frac_cnn)
            fractions_cluster_new.append(frac_cluster)
            fractions_fk_new.append(frac_fk)
            all_predictions[(beta, h)] = (y_true.copy(), y_pred.copy(), y_pred_cluster.copy(), y_pred_fk.copy())
            
            # Track worst CNN and cluster fractions separately
            if frac_cnn > worst_cnn_fraction:
                worst_cnn_fraction = frac_cnn
                worst_cnn_beta = beta
                worst_cnn_h = h
            if frac_cluster > worst_cluster_fraction:
                worst_cluster_fraction = frac_cluster
                worst_cluster_beta = beta
                worst_cluster_h = h
            if frac_fk > worst_fk_fraction:
                worst_fk_fraction = frac_fk
                worst_fk_beta = beta
                worst_fk_h = h

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

            test_fk_size = fk_largest_cluster_sizes_up(
                grids[test_idx],
                beta=beta,
                h=h,
                seed=int(args.fk_seed),
                indices=test_idx,
                n_draws=int(args.fk_draws),
            )
            k_fk, x0_fk = fit_sigmoid(test_fk_size, y_true)
            y_pred_fk = sigmoid(test_fk_size, k_fk, x0_fk)

            frac_cnn = fraction_outside_band(y_true=y_true, y_pred=y_pred, percent=float(args.percent), mode=str(args.mode))
            frac_cluster = fraction_outside_band(
                y_true=y_true,
                y_pred=y_pred_cluster,
                percent=float(args.percent),
                mode=str(args.mode),
            )
            frac_fk = fraction_outside_band(
                y_true=y_true,
                y_pred=y_pred_fk,
                percent=float(args.percent),
                mode=str(args.mode),
            )
            print(
                f"beta={beta:.3f}, h={h:.3f}: outside ±{args.percent}% band ({args.mode}) "
                f"CNN={frac_cnn:.4f} | LCS={frac_cluster:.4f} | FK={frac_fk:.4f}"
            )

            betas_newer.append(beta)
            fractions_cnn_newer.append(frac_cnn)
            fractions_cluster_newer.append(frac_cluster)
            fractions_fk_newer.append(frac_fk)
            all_predictions[(beta, h)] = (y_true.copy(), y_pred.copy(), y_pred_cluster.copy(), y_pred_fk.copy())
            
            # Track worst CNN and cluster fractions separately
            if frac_cnn > worst_cnn_fraction:
                worst_cnn_fraction = frac_cnn
                worst_cnn_beta = beta
                worst_cnn_h = h
            if frac_cluster > worst_cluster_fraction:
                worst_cluster_fraction = frac_cluster
                worst_cluster_beta = beta
                worst_cluster_h = h
            if frac_fk > worst_fk_fraction:
                worst_fk_fraction = frac_fk
                worst_fk_beta = beta
                worst_fk_h = h
        
        # Save data cache
        cache_payload = _build_cache_payload(
            betas=betas,
            fractions_cnn=fractions_cnn,
            fractions_cluster=fractions_cluster,
            fractions_fk=fractions_fk,
            betas_new=betas_new,
            fractions_cnn_new=fractions_cnn_new,
            fractions_cluster_new=fractions_cluster_new,
            fractions_fk_new=fractions_fk_new,
            betas_newer=betas_newer,
            fractions_cnn_newer=fractions_cnn_newer,
            fractions_cluster_newer=fractions_cluster_newer,
            fractions_fk_newer=fractions_fk_newer,
            worst_cnn_fraction=worst_cnn_fraction,
            worst_cnn_beta=worst_cnn_beta,
            worst_cnn_h=worst_cnn_h,
            worst_cluster_fraction=worst_cluster_fraction,
            worst_cluster_beta=worst_cluster_beta,
            worst_cluster_h=worst_cluster_h,
            worst_fk_fraction=worst_fk_fraction,
            worst_fk_beta=worst_fk_beta,
            worst_fk_h=worst_fk_h,
            fk_draws=int(args.fk_draws),
            fk_aggregate="mean",
            all_predictions=all_predictions,
        )
        _save_data_cache(cache_path, cache_payload)
    
    out_dir = Path(config.paths.plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out

    # Sort by beta for nicer lines
    order = np.argsort(np.asarray(betas, dtype=float))
    betas_sorted = np.asarray(betas, dtype=float)[order]
    cnn_sorted = np.asarray(fractions_cnn, dtype=float)[order]
    cluster_sorted = np.asarray(fractions_cluster, dtype=float)[order]
    fk_sorted = np.asarray(fractions_fk, dtype=float)[order]
    
    order_new = np.argsort(np.asarray(betas_new, dtype=float))
    betas_sorted_new = np.asarray(betas_new, dtype=float)[order_new]
    cnn_sorted_new = np.asarray(fractions_cnn_new, dtype=float)[order_new]
    cluster_sorted_new = np.asarray(fractions_cluster_new, dtype=float)[order_new]
    fk_sorted_new = np.asarray(fractions_fk_new, dtype=float)[order_new]

    order_newer = np.argsort(np.asarray(betas_newer, dtype=float))
    betas_sorted_newer = np.asarray(betas_newer, dtype=float)[order_newer]
    cnn_sorted_newer = np.asarray(fractions_cnn_newer, dtype=float)[order_newer]
    cluster_sorted_newer = np.asarray(fractions_cluster_newer, dtype=float)[order_newer]
    fk_sorted_newer = np.asarray(fractions_fk_newer, dtype=float)[order_newer]

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    
    # Lowest h runs (blue)
    ax.plot(betas_sorted, cluster_sorted, color=colors["light_steel_blue"], linewidth=2, alpha=0.8, zorder=1)
    ax.scatter(betas_sorted, cluster_sorted, s=60, marker="s", color=colors["steel_blue"], alpha=0.9, zorder=2)
    ax.plot(betas_sorted, cnn_sorted, color=colors["light_steel_blue"], linewidth=2, alpha=0.8, zorder=1)
    ax.scatter(betas_sorted, cnn_sorted, s=60, marker="o", color=colors["steel_blue"], alpha=0.9, zorder=2)
    ax.plot(betas_sorted, fk_sorted, color=colors["light_steel_blue"], linewidth=2, alpha=0.8, zorder=1)
    ax.scatter(betas_sorted, fk_sorted, s=70, marker="^", color=colors["steel_blue"], alpha=0.9, zorder=2)
    
    # Middle h runs (green)
    ax.plot(betas_sorted_new, cluster_sorted_new, color=colors["light_green"], linewidth=2, alpha=0.8, linestyle="--", zorder=1)
    ax.scatter(betas_sorted_new, cluster_sorted_new, s=60, marker="s", color=colors["forest_green"], alpha=0.9, zorder=2)
    ax.plot(betas_sorted_new, cnn_sorted_new, color=colors["light_green"], linewidth=2, alpha=0.8, linestyle="--", zorder=1)
    ax.scatter(betas_sorted_new, cnn_sorted_new, s=60, marker="o", color=colors["forest_green"], alpha=0.9, zorder=2)
    ax.plot(betas_sorted_new, fk_sorted_new, color=colors["light_green"], linewidth=2, alpha=0.8, linestyle="--", zorder=1)
    ax.scatter(betas_sorted_new, fk_sorted_new, s=70, marker="^", color=colors["forest_green"], alpha=0.9, zorder=2)

    # Highest h runs (red)
    ax.plot(betas_sorted_newer, cluster_sorted_newer, color=colors["soft_red"], linewidth=2, alpha=0.8, linestyle=":", zorder=1)
    ax.scatter(betas_sorted_newer, cluster_sorted_newer, s=60, marker="s", color=colors["firebrick_red"], alpha=0.9, zorder=2)
    ax.plot(betas_sorted_newer, cnn_sorted_newer, color=colors["soft_red"], linewidth=2, alpha=0.8, linestyle=":", zorder=1)
    ax.scatter(betas_sorted_newer, cnn_sorted_newer, s=60, marker="o", color=colors["firebrick_red"], alpha=0.9, zorder=2)
    ax.plot(betas_sorted_newer, fk_sorted_newer, color=colors["soft_red"], linewidth=2, alpha=0.8, linestyle=":", zorder=1)
    ax.scatter(betas_sorted_newer, fk_sorted_newer, s=70, marker="^", color=colors["firebrick_red"], alpha=0.9, zorder=2)

    # Create legend 1: markers (Metric type)
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black", markersize=10, label="LCS"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=10, label="q-NN"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="black", markersize=10, label="FK"),
    ]
    legend1 = ax.legend(
        handles=marker_handles,
        loc="upper center",
        bbox_to_anchor=(0.2, -0.18),
        ncol=len(marker_handles),
        frameon=True,
    )
    ax.add_artist(legend1)

    # Create legend 2: colors (Field strength)
    color_handles = [
        Line2D([0], [0], color=colors["firebrick_red"], linewidth=3, label="A"),
        Line2D([0], [0], color=colors["forest_green"], linewidth=3, label="B"),
        Line2D([0], [0], color=colors["steel_blue"], linewidth=3, label="C"),
    ]
    legend2 = ax.legend(
        handles=color_handles,
        loc="upper center",
        bbox_to_anchor=(0.78, -0.18),
        ncol=len(color_handles),
        frameon=True,
    )
    
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
    
    # Helper function to create scatter plot with binned and subsampled data
    def create_scatter_plot(y_true, y_pred, metric_label, beta, h, fraction_outside, args, out_dir, config, highlight_worst=False, highlight_best=False):
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        metric_name = metric_label.lower()
        if metric_name in {"cluster", "cluster size", "lcs"}:
            metric_marker = "s"
        elif metric_name in {"fk", "fortuin-kasteleyn"}:
            metric_marker = "^"
        else:
            metric_marker = "o"
        
        # Bin target committor into 50 bins and subsample predictions uniformly
        n_bins = 50
        bin_edges = np.linspace(0, 1, n_bins + 1)
        selected_indices = []
        errors = np.abs(y_pred - y_true)
        best_idx = int(np.argmin(errors))
        worst_idx = int(np.argmax(errors))
        central_mask = (y_true >= 0.2) & (y_true <= 0.8)
        if np.any(central_mask):
            central_indices = np.arange(len(y_true))[central_mask]
            best_idx = int(central_indices[np.argmin(errors[central_mask])])
        
        for i in range(n_bins):
            # Get indices of points in this bin
            mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
            if i == n_bins - 1:  # Last bin includes the right edge
                mask = (y_true >= bin_edges[i]) & (y_true <= bin_edges[i + 1])
            
            bin_indices = np.where(mask)[0]
            
            if len(bin_indices) == 0:
                continue
            
            # Get predicted committors for this bin
            bin_preds = y_pred[bin_indices]
            
            # Sort by predicted committor
            sort_idx = np.argsort(bin_preds)
            sorted_bin_indices = bin_indices[sort_idx]
            
            # Uniformly select 50 points including both endpoints
            n_points = len(sorted_bin_indices)
            to_sample = 10
            if n_points <= to_sample:
                # Include all points if we have 50 or fewer
                uniform_indices = np.arange(n_points)
            else:
                # Uniformly select 50 points from sorted predictions
                uniform_indices = np.round(np.linspace(0, n_points - 1, to_sample)).astype(int)
            
            selected_indices.extend(sorted_bin_indices[uniform_indices].tolist())
        
        selected_indices.extend([best_idx, worst_idx])
        selected_indices = np.asarray(sorted(set(selected_indices)), dtype=int)
        subsampled_y_true = y_true[selected_indices]
        subsampled_y_pred = y_pred[selected_indices]
        
        # Scatter plot
        ax.scatter(
            subsampled_y_true,
            subsampled_y_pred,
            alpha=0.5,
            s=30,
            marker=metric_marker,
            color=colors["steel_blue"],
            zorder=2,
        )
        
        # Diagonal line (perfect prediction, red, from 0 to 1)
        ax.plot([0, 1], [0, 1], "-", color=colors["firebrick_red"], linewidth=2, alpha=0.7, zorder=4)
        
        # Band lines (±percent)
        band_width = float(args.percent) / 100.0
        y_band = np.linspace(0, 1, 100)
        upper_bound = y_band + band_width
        lower_bound = y_band - band_width
        
        # Clip band to [0, 1] range
        upper_bound = np.clip(upper_bound, 0, 1)
        lower_bound = np.clip(lower_bound, 0, 1)
        
        ax.plot(y_band, upper_bound, "--", color=colors["firebrick_red"], linewidth=2, alpha=0.7, label=f"±{args.percent}% band", zorder=4)
        ax.plot(y_band, lower_bound, "--", color=colors["firebrick_red"], linewidth=2, alpha=0.7, zorder=4)
        ax.fill_between(y_band, lower_bound, upper_bound, color=colors["firebrick_red"], alpha=0.1, zorder=3)
        
        if highlight_worst:
            # Calculate error for all points (use unsubsampled data)
            worst_true = y_true[worst_idx]
            worst_pred = y_pred[worst_idx]
            
            # Draw open circle around worst point
            ax.scatter(
                [worst_true],
                [worst_pred],
                s=200,
                marker="o",
                facecolors="none",
                edgecolors=lighten_color(colors["firebrick_red"], 0.25),
                linewidths=2.5,
                label="Worst",
                zorder=10,
            )
        
        if highlight_best:
            # Find best point (minimum error)
            best_true = y_true[best_idx]
            best_pred = y_pred[best_idx]
            
            # Draw open circle around best point in green
            ax.scatter(
                [best_true],
                [best_pred],
                s=200,
                marker="o",
                facecolors="none",
                edgecolors=lighten_color(colors["forest_green"], 0.5),
                linewidths=2.5,
                label="Best",
                zorder=10,
            )
        
        ax.set_xlabel("Target Committor")
        ax.set_ylabel(f"Predicted Committor")
        ax.set_title(f"β={beta:.3f}, h={h:.3f}\n(Fraction outside band: {fraction_outside:.4f})")
        ax.grid(True, alpha=0.2, zorder=0)
        ax.legend()
        ax.set_aspect("equal")
        #ax.set_xlim(0, 1)
        #ax.set_ylim(0, 1)
        
        return plt.gcf()
    
    # Create scatter plot for worst CNN
    if worst_cnn_beta is not None and worst_cnn_h is not None:
        y_true_worst_cnn, y_pred_cnn_worst, _, _ = all_predictions[(worst_cnn_beta, worst_cnn_h)]

        create_scatter_plot(y_true_worst_cnn, y_pred_cnn_worst, "q-NN", worst_cnn_beta, worst_cnn_h, worst_cnn_fraction, args, out_dir, config)
        worst_cnn_path = out_dir / f"committor_scatter_cnn_{worst_cnn_beta:.3f}_{worst_cnn_h:.3f}.svg"
        plt.savefig(worst_cnn_path, bbox_inches='tight')
        plt.close()
        print(f"Saved worst CNN scatter plot to {worst_cnn_path}")
    
    # Create scatter plot for worst cluster
    if worst_cluster_beta is not None and worst_cluster_h is not None:
        y_true_worst_cluster, _, y_pred_cluster_worst, _ = all_predictions[(worst_cluster_beta, worst_cluster_h)]

        create_scatter_plot(y_true_worst_cluster, y_pred_cluster_worst, "Cluster Size", worst_cluster_beta, worst_cluster_h, worst_cluster_fraction, args, out_dir, config, highlight_worst=True, highlight_best=True)
        worst_cluster_path = out_dir / f"committor_scatter_cluster_{worst_cluster_beta:.3f}_{worst_cluster_h:.3f}.svg"
        plt.savefig(worst_cluster_path, bbox_inches='tight')
        plt.close()
        print(f"Saved worst cluster scatter plot to {worst_cluster_path}")

    # Create scatter plot for worst FK cluster
    if worst_fk_beta is not None and worst_fk_h is not None:
        y_true_worst_fk, _, _, y_pred_fk_worst = all_predictions[(worst_fk_beta, worst_fk_h)]

        create_scatter_plot(y_true_worst_fk, y_pred_fk_worst, "FK", worst_fk_beta, worst_fk_h, worst_fk_fraction, args, out_dir, config, highlight_worst=True, highlight_best=True)
        worst_fk_path = out_dir / f"committor_scatter_fk_{worst_fk_beta:.3f}_{worst_fk_h:.3f}.svg"
        plt.savefig(worst_fk_path, bbox_inches='tight')
        plt.close()
        print(f"Saved worst FK scatter plot to {worst_fk_path}")


if __name__ == "__main__":
    main()
