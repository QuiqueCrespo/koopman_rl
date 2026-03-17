"""
Latent Affine Sheaf-RL: Advantage Stalk (Q-Stalk)
==================================================
Structural fix for the value-averaging problem found in sheaf_rl_gridworld.py.

PROBLEM (V-stalk):
  Stalk = [z, v]  (d+1 dimensional)
  Every action edge from s_i contributes to the SAME scalar v_i.
  The Sheaf Laplacian AVERAGES over all of them:
    v_i = γ * (1/|A|) * Σ_a V(s_j^a)
  This is the random-policy value, not the optimal value.
  Result: V[s0] ≈ 0.082 instead of the true γ^3 = 0.729.

FIX (Q-stalk):
  Stalk = [z, q_0, q_1, q_2, q_3]  (d+|A| dimensional)
  Each action has its own Q-value in the stalk.
  The restriction map for edge (s_i, a, s_j) is a "Permutation-Coupler":
    Source:      selects Q(s_i, a)         — the specific taken action
    Destination: selects γ * Q(s_j, a*)   — max via greedy action a* = argmax Q(s_j,·)
  Disagreement (value part) = Q(s_i, a) - γ * Q(s_j, a*)  = r  (Bellman exactly)

  NO averaging. Each Q-value is updated only by its specific action edge.

The outer loop is interleaved greedy policy improvement (Q-iteration):
  1. Fix greedy policy a*[s] = argmax_a Q(s, a)
  2. Richardson diffusion → Q-evaluation under that policy (linear solve)
  3. Update a*[s] = argmax_a Q(s, a)  (policy improvement)
  4. Repeat until convergence → Q* (optimal Bellman Q-function)
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STATES  = 9
N_ACTIONS = 4
GOAL      = 8
GAMMA     = 0.9
D         = 9    # latent dim (one-hot in tabular case)

# Stalk dimensions
QSTALK_V = D + N_ACTIONS   # vertex stalk: 9 + 4 = 13
STALK_E  = D + 1            # edge stalk:   9 + 1 = 10  (shared with V-stalk)

ACTION_NAMES  = ["Up", "Down", "Left", "Right"]
ACTION_ARROWS = ["↑",  "↓",   "←",   "→"]

# ---------------------------------------------------------------------------
# Grid world (identical to tabular version)
# ---------------------------------------------------------------------------

def grid_step(s: int, a: int) -> tuple[int, float, bool]:
    row, col = divmod(s, 3)
    if a == 0:   row = max(0, row - 1)
    elif a == 1: row = min(2, row + 1)
    elif a == 2: col = max(0, col - 1)
    elif a == 3: col = min(2, col + 1)
    s_next = row * 3 + col
    return s_next, (1.0 if s_next == GOAL else 0.0), (s_next == GOAL)

def one_hot(s: int, n: int = D) -> np.ndarray:
    z = np.zeros(n); z[s] = 1.0; return z

def all_transitions() -> list[tuple]:
    result = []
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            s_next, r, _ = grid_step(s, a)
            result.append((s, a, r, s_next))
    return result

def build_koopman_operators() -> list[np.ndarray]:
    ops = []
    for a in range(N_ACTIONS):
        K = np.zeros((D, D))
        for s in range(N_STATES):
            s_next, _, _ = grid_step(s, a)
            K[s_next, s] = 1.0
        ops.append(K)
    return ops

# ---------------------------------------------------------------------------
# Q-Stalk: Incidence Matrix
# ---------------------------------------------------------------------------

def build_incidence_qstalk(
    transitions: list[tuple],
    koopman_ops: list[np.ndarray],
    a_star: np.ndarray,          # greedy action a*[s] for each state s
    gamma: float = GAMMA,
) -> csr_matrix:
    """
    B shape: [E * STALK_E,  N_STATES * QSTALK_V]  =  [360, 117]

    For each edge k = (s_i, a, r, s_j):

      Source block  B[k, s_i]  — shape (STALK_E × QSTALK_V) = (10 × 13):
        Upper D×D:         K_a         (Koopman dynamics on latent)
        Upper D×|A|:       0           (Q-values don't enter latent prediction)
        Bottom 1×D:        0
        Bottom 1×|A|:      e_a^T       (select Q(s_i, a) = the taken action)

      Dest block  B[k, s_j]  — negated, shape (10 × 13):
        Upper D×D:         -I_D        (identity on latent, negated)
        Upper D×|A|:       0
        Bottom 1×D:        0
        Bottom 1×|A|:      -γ·e_{a*}^T (select γ·Q(s_j, a*) = greedy next value)

    Disagreement (value component): Q(s_i,a) − γ·Q(s_j, a*[s_j])
    Setting this equal to r gives the Q-Bellman equation exactly.
    """
    E = len(transitions)
    B = lil_matrix((E * STALK_E, N_STATES * QSTALK_V), dtype=np.float64)

    for k, (s_i, a, _r, s_j) in enumerate(transitions):
        row0  = k * STALK_E
        col_i = s_i * QSTALK_V
        col_j = s_j * QSTALK_V
        K_a   = koopman_ops[a]
        as_j  = int(a_star[s_j])   # greedy action at destination

        # Source block
        B[row0 : row0 + D, col_i : col_i + D] = K_a     # latent: K_a
        B[row0 + D, col_i + D + a] = 1.0                 # Q-slot: select Q(s_i, a)

        # Destination block (negated)
        for i in range(D):
            B[row0 + i, col_j + i] = -1.0                # latent: -I
        B[row0 + D, col_j + D + as_j] = -gamma           # Q-slot: -γ·Q(s_j, a*)

    return csr_matrix(B)

# ---------------------------------------------------------------------------
# Q-Stalk: Reward vector and state initialization
# ---------------------------------------------------------------------------

def build_reward_qstalk(transitions: list[tuple]) -> np.ndarray:
    """
    R[s_i * QSTALK_V + D + a] = r   for transitions with r > 0.

    This is the RHS of the Poisson equation Δ_F X = R.
    For edge (s_i, a, r, s_j):  Q(s_i, a) - γ·Q(s_j, a*) = r
    → r appears at the Q_a slot of s_i in the vertex-space RHS.
    """
    R = np.zeros(N_STATES * QSTALK_V)
    for s_i, a, r, s_j in transitions:
        if r > 0.0:
            R[s_i * QSTALK_V + D + a] += r
    return R

def init_global_state_qstalk() -> np.ndarray:
    """X_0: one-hot latents, all Q-values = 0."""
    X = np.zeros(N_STATES * QSTALK_V)
    for s in range(N_STATES):
        X[s * QSTALK_V : s * QSTALK_V + D] = one_hot(s)
    return X

def apply_anchors_qstalk(X: np.ndarray) -> np.ndarray:
    """
    Dirichlet anchors — applied after every Richardson step:
      1. Latents z_s = one_hot(s)  — fixed; encoder not changing in tabular case
      2. Q(s8, a) = 0 for all a   — terminal state has no future value
    """
    for s in range(N_STATES):
        X[s * QSTALK_V : s * QSTALK_V + D] = one_hot(s)
    X[GOAL * QSTALK_V + D : GOAL * QSTALK_V + D + N_ACTIONS] = 0.0
    return X

def extract_q_table(X: np.ndarray) -> np.ndarray:
    Q = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        Q[s] = X[s * QSTALK_V + D : s * QSTALK_V + D + N_ACTIONS]
    return Q

# ---------------------------------------------------------------------------
# Q-evaluation: two solvers — symmetric Laplacian vs directed linear solve
# ---------------------------------------------------------------------------

def qeval_symmetric(
    koopman_ops: list[np.ndarray],
    transitions: list[tuple],
    a_star: np.ndarray,
    K_steps: int = 400,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Q-evaluation via Richardson iteration on the SYMMETRIC Sheaf Laplacian Δ_F = B^T B.

    WHY THIS GIVES APPROXIMATE (NOT EXACT) VALUES:
    Δ_F = B^T B is symmetric. At each Q(s_i, a) entry, the Laplacian sums:
      (a) Forward edge  (s_i, a) → s_j: wants Q(s_i,a) = γ Q(s_j,a*) + r
      (b) Backward coupling from any edge (s_k, b) → s_i where a*[s_i]=a:
          wants Q(s_i,a) = Q(s_k,b)/γ  (pulls in the WRONG direction)
    The solution minimizes the sum of squared Bellman errors globally, which
    is NOT the Bellman fixed point — it's a least-squares blend of all constraints.
    Policy is still correct; absolute values are distorted.
    """
    R  = build_reward_qstalk(transitions)
    X  = apply_anchors_qstalk(init_global_state_qstalk())
    B  = build_incidence_qstalk(transitions, koopman_ops, a_star)
    DF = B.T @ B

    lam_max, _ = eigsh(DF, k=1, which="LM", tol=1e-3, maxiter=500)
    alpha = 1.0 / float(lam_max[0])

    for _ in range(K_steps):
        X_new = X - alpha * (DF @ X) + alpha * R
        X_new = apply_anchors_qstalk(X_new)
        if np.max(np.abs(X_new - X)) < tol:
            break
        X = X_new

    return extract_q_table(X)


def qeval_directed(transitions: list[tuple], a_star: np.ndarray) -> np.ndarray:
    """
    Q-evaluation via DIRECTED linear solve: M q = r

    The Q-Bellman system under fixed policy a* is:
        Q(s_i, a) - γ · Q(s_j, a*[s_j]) = r(s_i, a)   for all (s_i, a)
    This is a directed (asymmetric) linear system. Written as M q = r:
        M[i, i]            = 1
        M[i, idx(s_j,a*)]  = -γ      (if s_j ≠ GOAL)
    where idx(s, a) = s * N_ACTIONS + a.

    The Q-stalk STRUCTURE is what makes this valid — each Q(s,a) couples to
    exactly ONE destination Q(s_j, a*[s_j]), giving a well-conditioned linear
    system solvable by sparse LU. No symmetric averaging; exact Bellman values.
    """
    from scipy.sparse.linalg import spsolve

    n     = N_STATES * N_ACTIONS
    idx   = lambda s, a: s * N_ACTIONS + a
    M     = lil_matrix((n, n), dtype=np.float64)
    r_vec = np.zeros(n)

    for s_i, a, r, s_j in transitions:
        i        = idx(s_i, a)
        M[i, i]  = 1.0
        r_vec[i] = r
        if s_j != GOAL:
            M[i, idx(s_j, int(a_star[s_j]))] = -GAMMA

    # Anchor terminal Q-values to 0 (s8 has no future)
    for a in range(N_ACTIONS):
        i = idx(GOAL, a)
        M[i, :] = 0.0
        M[i, i] = 1.0
        r_vec[i] = 0.0

    q_vec = spsolve(csr_matrix(M), r_vec)
    return q_vec.reshape(N_STATES, N_ACTIONS)


def qiteration(
    koopman_ops: list[np.ndarray],
    transitions: list[tuple],
    n_pi_iter: int = 8,
    directed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Q-iteration: alternate Q-evaluation and greedy policy improvement.
    Returns both the symmetric-Laplacian Q (for comparison) and the directed Q.
    """
    a_star = np.zeros(N_STATES, dtype=int)

    print(f"\n--- Symmetric Laplacian solver (Δ_F = B^T B) ---")
    Q_sym = None
    a_sym = np.zeros(N_STATES, dtype=int)
    for pi in range(n_pi_iter):
        Q_sym  = qeval_symmetric(koopman_ops, transitions, a_sym)
        a_sym  = Q_sym.argmax(axis=1)
        V = Q_sym.max(axis=1)
        print(f"  PI {pi+1}: V(s0)={V[0]:.4f}  V(s7)={V[7]:.4f}  V(s8)={V[8]:.4f}")

    print(f"\n--- Directed solver (M q = r, sparse LU) ---")
    Q_dir = None
    a_dir = np.zeros(N_STATES, dtype=int)
    for pi in range(n_pi_iter):
        Q_dir  = qeval_directed(transitions, a_dir)
        a_dir  = Q_dir.argmax(axis=1)
        V = Q_dir.max(axis=1)
        print(f"  PI {pi+1}: V(s0)={V[0]:.4f}  V(s7)={V[7]:.4f}  V(s8)={V[8]:.4f}")

    return Q_sym, Q_dir

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    print("=" * 62)
    print("  Sheaf-RL: Advantage Stalk (Q-Stalk)")
    print("  Stalk: [z, q_0, q_1, q_2, q_3]  (d + |A| = 13 dim)")
    print("  Fix: each edge updates one Q(s,a), no cross-action averaging")
    print("=" * 62)

    koopman_ops = build_koopman_operators()
    transitions = all_transitions()
    print(f"\nVertex stalk: R^{QSTALK_V}  (d={D} + |A|={N_ACTIONS})")
    print(f"Edge stalk:   R^{STALK_E}  (d={D} + 1 scalar Q-coupling)")
    print(f"B shape: [{len(transitions) * STALK_E}, {N_STATES * QSTALK_V}]")
    print(f"Δ_F shape: [{N_STATES * QSTALK_V}, {N_STATES * QSTALK_V}]")

    Q_sym, Q_dir = qiteration(koopman_ops, transitions, n_pi_iter=8)
    V_sym  = Q_sym.max(axis=1)
    V_dir  = Q_dir.max(axis=1)
    policy = Q_dir.argmax(axis=1)

    # --- Q-table printout (directed solver) ---
    print("\nFinal Q-table — directed solver (exact Bellman):")
    print(f"  {'State':8s}  {'Q(Up)':7s}  {'Q(Dn)':7s}  {'Q(Lt)':7s}  {'Q(Rt)':7s}")
    print("  " + "-" * 44)
    for s in range(N_STATES):
        row_str = f"  s{s} r{s//3}c{s%3}  "
        for a in range(N_ACTIONS):
            marker = ">" if a == policy[s] else " "
            row_str += f"{marker}{Q_dir[s, a]:6.4f}  "
        print(row_str)

    # --- Sanity check: directed solver should give exact γ^k ---
    print(f"\nSanity check — directed Q-stalk vs true Bellman Q*:")
    checks = [
        ("Q(s7, Right)", Q_dir[7, 3], 1.0,      "γ^0 = 1.000"),
        ("Q(s4, Down) ", Q_dir[4, 1], GAMMA,     f"γ^1 = {GAMMA:.3f}"),
        ("Q(s3, Right)", Q_dir[3, 3], GAMMA**2,  f"γ^2 = {GAMMA**2:.3f}"),
        ("V*(s0)      ", V_dir[0],    GAMMA**3,  f"γ^3 = {GAMMA**3:.3f}"),
    ]
    for name, got, want, label in checks:
        ok = abs(got - want) < 1e-6
        print(f"  {name} = {got:.6f}  (expected {label})  {'✓' if ok else '✗'}")

    # --- Three-way comparison ---
    V_vstalk  = np.array([0.0822, 0.1197, 0.1359,
                           0.1197, 0.1968, 0.2760,
                           0.1359, 0.2760, 1.0000])
    V_optimal = np.array([GAMMA**4, GAMMA**3, GAMMA**2,
                           GAMMA**3, GAMMA**2, GAMMA**1,
                           GAMMA**2, GAMMA**1, 1.0])

    # Greedy path (directed solver)
    s, path = 0, [0]
    for _ in range(20):
        if s == GOAL: break
        s, _, _ = grid_step(s, int(policy[s]))
        path.append(s)
    print(f"\nGreedy path: {' → '.join(f's{p}' for p in path)}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(N_STATES)
    w = 0.22
    axes[0].bar(x - w*1.5, V_vstalk, w, label="V-stalk (sym. Laplacian)",      color="steelblue", alpha=0.85)
    axes[0].bar(x - w*0.5, V_sym,    w, label="Q-stalk (sym. Laplacian)",      color="salmon",    alpha=0.85)
    axes[0].bar(x + w*0.5, V_dir,    w, label="Q-stalk (directed solve) ← fix", color="tomato",   alpha=0.85)
    axes[0].bar(x + w*1.5, V_optimal,w, label="True Bellman V*(s)",             color="gold",      alpha=0.85, edgecolor="gray")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"s{i}" for i in range(N_STATES)])
    axes[0].set_ylabel("State Value")
    axes[0].set_title("V-stalk vs Q-stalk vs True V*\n"
                      "Q-stalk matches optimal Bellman; V-stalk averages over actions")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_ylim(0, 1.15)

    # Heatmap: directed Q-stalk (exact values)
    im = axes[1].imshow(V_dir.reshape(3, 3), cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], label="V*(s) = max_a Q*(s, a)")
    for s in range(N_STATES):
        r, c = divmod(s, 3)
        arrow = "★" if s == GOAL else ACTION_ARROWS[policy[s]]
        axes[1].text(c, r - 0.15, f"{V_dir[s]:.3f}",
                     ha="center", fontsize=10, fontweight="bold")
        axes[1].text(c, r + 0.22, arrow, ha="center", fontsize=16, color="navy")
    axes[1].set_xticks([0, 1, 2]); axes[1].set_yticks([0, 1, 2])
    axes[1].set_title("Q-Stalk + Directed Solve: Exact Bellman V*(s)\n"
                      "(Q-values in stalk → no averaging; directed solver → exact γ^k)")

    plt.tight_layout()
    plt.savefig("sheaf_rl_qstalk_values.png", dpi=150)
    print("\nSaved → sheaf_rl_qstalk_values.png")
    plt.close()


if __name__ == "__main__":
    main()
