import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=np.inf)

# ============================
# Load Transition Matrices
# ============================

data = np.load("T_matrices.npz")
m_list = sorted(int(k.split("m")[-1]) for k in data.keys())

# Build dict m -> T
T_dict = {}
for m in m_list:
    T_dict[m] = data[f"T_m{m}"]

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
    nontrivial = eigvals[1:]

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
    print("Implied timescales:", its[:5])

    # 3. Locality
    loc = locality_metric(T, band_radius=1)
    print(f"Locality (±1 bin): {loc:.4f}")

    # 4. Chapman–Kolmogorov test, if possible
    if (2*m) in T_dict:
        ck_max, ck_fro = ck_test(T, T_dict[2*m])
        print(f"CK max |diff|  vs T(2τ): {ck_max:.4e}")
        print(f"CK Frobenius norm vs T(2τ): {ck_fro:.4e}")
    else:
        print(f"No CK test available for m={m} -> 2m not saved.")

    S = T.shape[0]
    A_states = np.array([0])        # or multiple indices
    B_states = np.array([S - 1])

    MFPT_AB = mfpt_A_to_B(T, tau, A_states, B_states)
    k_AB = 1.0 / MFPT_AB

    J_AB = k_AB / (64*64)

    print("Nucleation rate:", J_AB)

# ============================
# Compare implied timescales across lags
# ============================

print("\n=== Implied Timescales Comparison ===")
for m in m_list:
    print(f"m={m}, tau={m}: {its_results[m][:5]}")

exit()

q_A = 0.05
q_B = 0.95

num_steps = int((q_B-q_A)/0.05)
q_bins = np.linspace(q_A, q_B, num_steps)

q_bins = np.concatenate((np.array([0]), q_bins, np.array([1])))

q_values = (q_bins[:-1] + q_bins[1:]) / 2

plt.plot(q_values, pi)
plt.yscale('log')
plt.savefig("figures/pi.pdf")

for i in range(S):
    q_i = 0
    for j in range(S):
        #if i != j:
        q_i += q_values[j]*T[i,j]
    #print(q_bins[i], q_i, q_bins[i+1])

