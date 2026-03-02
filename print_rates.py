from pathlib import Path
import numpy as np

def print_nucleation_rates(directory="data"):
    """
    Reads all nucleation_*.npz files in `directory`
    and prints h, beta, and rate_per_site only.
    """
    directory = Path(directory)
    files = sorted(directory.glob("nucleation_*.npz"))

    if not files:
        print("No nucleation files found.")
        return

    rows = []

    for f in files:
        with np.load(f, allow_pickle=True) as data:
            rows.append((
                float(data["h"][0]),
                float(data["beta"][0]),
                float(data["rate_per_site"][0]),
            ))

    # Sort by h, then beta
    rows.sort(key=lambda x: (x[0], x[1]))

    # Header
    print(f"{'h':>8} {'beta':>8} {'rate_per_site':>16}")
    print("-" * 34)

    for h, beta, rate in rows:
        print(f"{h:8.3f} {beta:8.3f} {rate:16.6e}")