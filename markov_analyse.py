import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.config import load_config
import math

np.set_printoptions(linewidth=np.inf)

# --- Load Config ---
config = load_config("config.yaml")

# --- parse optional inputs ---
parser = argparse.ArgumentParser()
parser.add_argument("--beta", type=float, help="Override beta value")
parser.add_argument("--h", type=float, help="Override h value")
args = parser.parse_args()

# --- apply overrides ---
beta = args.beta if args.beta is not None else config.parameters.beta
h = args.h if args.h is not None else config.parameters.h

# ============================
# Load Count Matrices
# ============================

data = np.load(f"data/C_matrices_{beta:.3f}_{h:.3f}.npz")
m_list = sorted(int(k.split("m")[-1]) for k in data.keys())

# Build dict m -> C
C_dict = {}
for m in m_list:
    C_dict[m] = data[f"C_m{m}"]

# Convert counts to transition matrices
T_dict = {}
for m in m_list:
    C = C_dict[m].astype(float)
    row_sums = C.sum(axis=1, keepdims=True)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        T = np.where(row_sums > 0, C / row_sums, 0.0)
    T_dict[m] = T

# ============================
# Helper Functions
# ============================

def stochasticity_checks(T):
    row_sums = T.sum(axis=1)
    neg = np.any(T < -1e-12)
    return row_sums.min(), row_sums.max(), neg

def implied_timescales(T, tau):
    # eigenvalues
    eigvals, _ = np.linalg.eig(T)
    eigvals = np.real(eigvals)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]

    # skip λ = 1
    nontrivial = eigvals[1:6]

    # timescales
    its = -tau / np.log(nontrivial)
    return eigvals, its

def locality_metric(T, band_radius=1):
    S = T.shape[0]
    total = T.sum()
    band_total = 0.0

    for i in range(S):
        j_min = max(0, i - band_radius)
        j_max = min(S, i + band_radius + 1)
        band_total += T[i, j_min:j_max].sum()

    return band_total / total

def ck_test(T_tau, T_2tau):
    T_pred = T_tau @ T_tau
    diff = T_pred - T_2tau
    return np.max(np.abs(diff)), np.linalg.norm(diff)

def stationary_distribution(T):
    eigvals, eigvecs = np.linalg.eig(T.T)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = eigvecs[:, idx]

    # enforce non-negativity and normalization
    j = np.argmax(np.abs(pi))
    if pi[j] < 0:
        pi = -pi
    pi = np.maximum(pi, 0)
    pi /= pi.sum()
    return pi

def mfpt_A_to_B(T, tau, A_states, B_states):
    S = T.shape[0]
    all_states = np.arange(S)

    # R = non-B states
    mask_R = np.ones(S, dtype=bool)
    mask_R[B_states] = False
    R = all_states[mask_R]

    T_RR = T[np.ix_(R, R)]
    I = np.eye(len(R))
    one = np.ones(len(R))

    t_R = np.linalg.solve(I - T_RR, tau * one)

    # MFPT from A: average over A states within R
    mfpts_A = []
    for a in A_states:
        idx_a = np.where(R == a)[0][0]
        mfpts_A.append(t_R[idx_a])

    MFPT_AB = np.mean(mfpts_A)
    return MFPT_AB

def bootstrap_J_AB_from_counts(C,
                               tau,
                               A_states,
                               B_states,
                               n_boot=500,
                               rng_seed=None):
    """
    Parametric bootstrap for nucleation rate J_AB from a count matrix C.

    - C : (S, S) int array, counts at lag tau
    - tau : lag time
    - A_states, B_states : arrays of state indices
    - n_boot : number of bootstrap samples
    """
    rng = np.random.default_rng(rng_seed)

    C = np.asarray(C, dtype=float)
    print(C)
    S = C.shape[0]

    # Central transition matrix from C (same as in your code)
    row_sums = C.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        T_central = np.where(row_sums > 0, C / row_sums, 0.0)

    # Central MFPT and nucleation rate
    MFPT_AB = mfpt_A_to_B(T_central, tau, A_states, B_states)
    k_AB = 1.0 / MFPT_AB
    J_AB = k_AB / (64 * 64)

    # Precompute row-wise probabilities for multinomial resampling
    p = np.zeros_like(T_central)
    row_sums_int = C.sum(axis=1)
    for i in range(S):
        Ni = row_sums_int[i]
        if Ni > 0:
            p[i] = C[i] / Ni
        else:
            # If no outgoing counts from i, keep it as a row of zeros
            # (same behavior as your T-construction)
            p[i, :] = 0.0

    # Bootstrap samples of J_AB
    J_samples = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        C_b = np.zeros_like(C)
        for i in range(S):
            Ni = row_sums_int[i]
            if Ni > 0:
                C_b[i] = rng.multinomial(Ni, p[i])
            else:
                C_b[i, :] = 0

        # Build T_b in the same way as central T
        C_b_float = C_b.astype(float)
        row_sums_b = C_b_float.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            T_b = np.where(row_sums_b > 0, C_b_float / row_sums_b, 0.0)

        MFPT_b = mfpt_A_to_B(T_b, tau, A_states, B_states)
        k_b = 1.0 / MFPT_b
        J_samples[b] = k_b / (64 * 64)

    # Summary statistics
    J_mean = np.mean(J_samples)
    J_std = np.std(J_samples, ddof=1)
    J_low, J_high = np.percentile(J_samples, [2.5, 97.5])

    stats = {
        "J_central": J_AB,
        "J_mean": J_mean,
        "J_std": J_std,
        "J_ci_95": (J_low, J_high),
        "samples": J_samples,
    }
    return stats


# ============================
# Analysis Loop
# ============================

print("\n=== MSM Analysis ===")

its_results = {}

for m in m_list:
    T = T_dict[m]
    tau = m * 1.0  # Δt = 1

    print(f"\n--- Lag m={m}, tau={tau} ---")

    # 1. Stochasticity
    rmin, rmax, neg = stochasticity_checks(T)
    print(f"Row sum min/max: {rmin:.6f}, {rmax:.6f}")
    print(f"Any negative entries: {neg}")

    # 2. Eigenvalues + implied timescales
    eigvals, its = implied_timescales(T, tau)
    its_results[m] = its

    print("Eigenvalues:", eigvals[:5])
    #print("Implied timescales:", its[:5])

    # 3. Locality
    loc = locality_metric(T, band_radius=1)
    print(f"Locality (±1 bin): {loc:.4f}")

    S = T.shape[0]
    A_states = np.array([0])        # or multiple indices
    B_states = np.array([S - 1])

    # Use the original count matrix for this lag
    C = C_dict[m]

    stats = bootstrap_J_AB_from_counts(
        C,
        tau,
        A_states,
        B_states,
        n_boot=500,      # adjust as desired
        rng_seed=12345   # fixed seed for reproducibility
    )

    J_central = stats["J_central"]
    J_low, J_high = stats["J_ci_95"]
    J_std = stats["J_std"]

    exp = math.floor(math.log10(abs(J_central))) if J_central != 0 else 0
    coeff_central = J_central / (10 ** exp) if J_central != 0 else 0
    coeff_std = J_std / (10 ** exp) if J_central != 0 else 0
    coeff_low = J_low / (10 ** exp) if J_central != 0 else 0
    coeff_high = J_high / (10 ** exp) if J_central != 0 else 0

    print(f"Nucleation rate: ({coeff_central:.3f} +/- {coeff_std:.3f})e{exp}")
    print(f"95% CI: [{coeff_low:.3f}e{exp}, {coeff_high:.3f}e{exp}]")

