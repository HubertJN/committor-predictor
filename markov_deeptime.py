"""
Interactive Markov State Model builder: Load count matrices and convert to transition matrices.
"""
import numpy as np
from pathlib import Path

np.set_printoptions(linewidth=np.inf, precision=6, suppress=True)


def load_count_matrix(beta: float, h: float, rc: str, m: int, c_matrix_in: str | None = None) -> np.ndarray:
    """Load a specific count matrix from NPZ file."""
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
    
    # Get list of available lags
    m_list = sorted(int(k.split("m")[-1]) for k in data.keys() if k.startswith("C_m"))
    
    if m not in m_list:
        print(f"Available lags: {m_list}")
        raise ValueError(f"Lag m={m} not found in count matrix file")
    
    return np.asarray(data[f"C_m{m}"], dtype=float)


def count_to_transition(C: np.ndarray, normalize_rows: bool = True) -> np.ndarray:
    """
    Convert count matrix to transition matrix.
    
    Args:
        C: Count matrix (n_states, n_states)
        normalize_rows: If True, normalize by row sums. If False, normalize by total sum.
    
    Returns:
        Transition matrix P
    """
    C = np.asarray(C, dtype=float)
    
    if normalize_rows:
        # Row normalization: P[i,j] = C[i,j] / sum(C[i,:])
        row_sums = C.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(row_sums > 0, C / row_sums, 0.0)
    else:
        # Column normalization: P[i,j] = C[i,j] / sum(C[:,j])
        col_sums = C.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(col_sums > 0, C / col_sums, 0.0)
    
    return P


def inspect_matrix(C: np.ndarray, P: np.ndarray, name: str = "Matrix") -> None:
    """Print statistics about count and transition matrices."""
    print(f"\n{'='*60}")
    print(f"{name} Statistics")
    print(f"{'='*60}")
    print(f"Shape: {C.shape}")
    print(f"Count matrix total: {C.sum():.0f}")
    print(f"Number of states: {C.shape[0]}")
    print(f"\nRow sum statistics (count matrix):")
    print(f"  Min: {C.sum(axis=1).min():.0f}")
    print(f"  Max: {C.sum(axis=1).max():.0f}")
    print(f"  Mean: {C.sum(axis=1).mean():.1f}")
    print(f"\nTransition matrix row sums (should be ~1.0):")
    row_sums = P.sum(axis=1)
    print(f"  Min: {row_sums.min():.6f}")
    print(f"  Max: {row_sums.max():.6f}")
    print(f"  Mean: {row_sums.mean():.6f}")
    print(f"\nTransition matrix (first 5x5):")
    print(P[:5, :5])


# Interactive session
if __name__ == "__main__":
    print("Interactive MSM Builder - Count to Transition Matrix\n")
    
    # Load default or user-specified parameters
    try:
        beta = float(input("Enter beta [default 0.511]: ") or 0.511)
        h = float(input("Enter h [default 0.040]: ") or 0.040)
        rc = input("RC (cnn/lcs) [default cnn]: ").strip() or "cnn"
        m = int(input("Enter lag m [default 16]: ") or 16)
    except ValueError:
        print("Invalid input. Using defaults: beta=0.511, h=0.040, rc=cnn, m=16")
        beta, h, rc, m = 0.511, 0.040, "cnn", 16
    
    try:
        # Load count matrix
        print(f"\nLoading C_m{m} for (beta={beta:.3f}, h={h:.3f}, rc={rc})...")
        C = load_count_matrix(beta, h, rc, m)
        
        # Convert to transition matrix
        P = count_to_transition(C, normalize_rows=True)
        
        # Inspect
        inspect_matrix(C, P, f"Lag m={m}")
        
        # Store for further analysis
        print(f"\nVariables available:")
        print(f"  C - Count matrix ({C.shape})")
        print(f"  P - Transition matrix ({P.shape})")
        print(f"  beta={beta}, h={h}, rc='{rc}', m={m}")
        
    except Exception as e:
        print(f"Error: {e}")

