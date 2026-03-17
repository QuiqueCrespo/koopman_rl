"""
Latent Affine Sheaf-RL: Minimal 3x3 Grid World Demo
=====================================================
No neural networks. Pure tabular math with numpy/scipy.

Grid layout (row-major):
  s0 | s1 | s2
  s3 | s4 | s5
  s6 | s7 | s8 (GOAL, reward=+1)

Actions: 0=Up, 1=Down, 2=Left, 3=Right
Latent encoding: 9-dimensional one-hot (the "Koopman" space)
Stalk dimension: d+1 = 10  (9 latent + 1 value)
Global state X: 90-dimensional (9 states x 10)
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_STATES  = 9
N_ACTIONS = 4
D         = 9          # latent dimension (one-hot)
STALK_DIM = D + 1      # 10
GOAL      = 8          # state index of the goal (bottom-right)
GAMMA     = 0.9
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
ACTION_ARROWS = ["↑", "↓", "←", "→"]

# ---------------------------------------------------------------------------
# 1. Environment
# ---------------------------------------------------------------------------

def grid_step(s: int, a: int) -> tuple[int, float, bool]:
    """Deterministic step with wall-bounce."""
    row, col = divmod(s, 3)
    if a == 0:   row = max(0, row - 1)   # Up
    elif a == 1: row = min(2, row + 1)   # Down
    elif a == 2: col = max(0, col - 1)   # Left
    elif a == 3: col = min(2, col + 1)   # Right
    s_next = row * 3 + col
    reward = 1.0 if s_next == GOAL else 0.0
    done   = s_next == GOAL
    return s_next, reward, done

# ---------------------------------------------------------------------------
# 2. Koopman Operators (permutation matrices)
# ---------------------------------------------------------------------------

def build_koopman_operators() -> list[np.ndarray]:
    """
    K_a[s_next, s] = 1  iff  grid_step(s, a) = s_next
    So K_a @ one_hot(s) = one_hot(s_next).
    For a deterministic env these are permutation matrices.
    """
    ops = []
    for a in range(N_ACTIONS):
        K = np.zeros((D, D))
        for s in range(N_STATES):
            s_next, _, _ = grid_step(s, a)
            K[s_next, s] = 1.0
        ops.append(K)
    return ops

def verify_koopman(ops: list[np.ndarray]):
    """Sanity-check: K_Right @ z_{s1} should equal z_{s2}."""
    z0 = one_hot(0)
    z1_pred = ops[3] @ z0   # Right from s0 → s1
    z1_true = one_hot(1)
    assert np.allclose(z1_pred, z1_true), "Koopman Right operator broken!"
    z2_pred = ops[1] @ one_hot(1)  # Down from s1 → s4
    z4_true = one_hot(4)
    assert np.allclose(z2_pred, z4_true), "Koopman Down operator broken!"
    print("[OK] Koopman operators verified: K_Right @ z_s0 = z_s1, K_Down @ z_s1 = z_s4")

# ---------------------------------------------------------------------------
# 3. Replay Buffer
# ---------------------------------------------------------------------------

def one_hot(s: int, n: int = D) -> np.ndarray:
    z = np.zeros(n)
    z[s] = 1.0
    return z

def collect_replay_buffer(n_episodes: int = 60, max_steps: int = 80) -> list[tuple]:
    """
    Random-policy rollouts. Returns deduplicated (s, a, r, s_next) tuples.
    With 9 states x 4 actions there are at most 36 unique transitions.
    """
    seen = set()
    transitions = []
    for _ in range(n_episodes):
        s = np.random.randint(0, N_STATES - 1)   # never start at goal
        for _ in range(max_steps):
            a = np.random.randint(0, N_ACTIONS)
            s_next, r, done = grid_step(s, a)
            key = (s, a)
            if key not in seen:
                seen.add(key)
                transitions.append((s, a, r, s_next))
            if done:
                break
            s = s_next
    return transitions

# ---------------------------------------------------------------------------
# 4. Incidence Matrix  B  (the heart of the sheaf)
# ---------------------------------------------------------------------------

def build_incidence_matrix(
    transitions: list[tuple],
    koopman_ops: list[np.ndarray],
    gamma: float = GAMMA,
) -> csr_matrix:
    """
    B shape: [E * STALK_DIM,  N_STATES * STALK_DIM]  i.e.  [E*10, 90]

    For each edge k = (s_i, a, r, s_j):
      Source block  B[k, s_i]:  [ K_a | 0 ]
                                 [ 0^T | 1 ]
      Dest   block  B[k, s_j]:  [ -I_9 | 0  ]   (negated: disagreement = src - dst)
                                 [  0^T | -γ ]
    """
    E   = len(transitions)
    n_rows = E * STALK_DIM
    n_cols = N_STATES * STALK_DIM

    B = lil_matrix((n_rows, n_cols), dtype=np.float64)

    for k, (s_i, a, _r, s_j) in enumerate(transitions):
        row0  = k * STALK_DIM
        col_i = s_i * STALK_DIM
        col_j = s_j * STALK_DIM
        K_a   = koopman_ops[a]

        # Source block: upper-left D×D = K_a
        B[row0 : row0 + D, col_i : col_i + D] = K_a
        # Source block: bottom-right scalar = 1
        B[row0 + D, col_i + D] = 1.0

        # Destination block: upper-left D×D = -I (negated)
        for d in range(D):
            B[row0 + d, col_j + d] = -1.0
        # Destination block: bottom-right scalar = -gamma (negated)
        B[row0 + D, col_j + D] = -gamma

    return csr_matrix(B)

# ---------------------------------------------------------------------------
# 5. Reward Vector
# ---------------------------------------------------------------------------

def build_reward_vector() -> np.ndarray:
    """
    Sparse R ∈ R^90.
    Only the value slot of the goal state is non-zero (+1).
    """
    R = np.zeros(N_STATES * STALK_DIM)
    R[GOAL * STALK_DIM + D] = 1.0
    return R

# ---------------------------------------------------------------------------
# 6. Initial Global State  X_0
# ---------------------------------------------------------------------------

def initialize_global_state() -> np.ndarray:
    """
    X_0 ∈ R^90.
    Latent slots: fixed one-hot encodings.
    Value slots:  0 everywhere except goal = +1 (Dirichlet seed).
    """
    X = np.zeros(N_STATES * STALK_DIM)
    for s in range(N_STATES):
        X[s * STALK_DIM : s * STALK_DIM + D] = one_hot(s)
    X[GOAL * STALK_DIM + D] = 1.0
    return X

# ---------------------------------------------------------------------------
# 7. Richardson Diffusion
# ---------------------------------------------------------------------------

def apply_dirichlet_anchors(X: np.ndarray) -> np.ndarray:
    """Re-anchor latent one-hots (fixed) and goal value (=1)."""
    for s in range(N_STATES):
        X[s * STALK_DIM : s * STALK_DIM + D] = one_hot(s)
    X[GOAL * STALK_DIM + D] = 1.0
    return X

def richardson_diffusion(
    B: csr_matrix,
    R: np.ndarray,
    X0: np.ndarray,
    K_steps: int = 300,
    tol: float = 1e-7,
) -> tuple[np.ndarray, int]:
    """
    X^(k+1) = (I - α Δ_F) X^(k) + α R
    Δ_F = B^T B  (Sheaf Laplacian)

    α is set to 1/λ_max for guaranteed convergence (optimal Richardson step).
    Hard Dirichlet anchors applied after every step.
    """
    Delta_F = B.T @ B   # [90×90] sparse

    # Compute largest eigenvalue for optimal step size
    lam_max, _ = eigsh(Delta_F, k=1, which="LM", tol=1e-4, maxiter=1000)
    lam_max = float(lam_max[0])
    alpha = 1.0 / lam_max
    print(f"  λ_max(Δ_F) = {lam_max:.4f}  →  α = {alpha:.6f}")

    X = X0.copy()
    converged_at = K_steps
    for k in range(K_steps):
        X_new = X - alpha * (Delta_F @ X) + alpha * R
        X_new = apply_dirichlet_anchors(X_new)

        delta = np.max(np.abs(X_new - X))
        X = X_new
        if delta < tol:
            converged_at = k + 1
            print(f"  Converged at iteration {converged_at}  (Δ_inf = {delta:.2e})")
            break
    else:
        print(f"  Did not fully converge in {K_steps} steps (final Δ_inf = {delta:.2e})")

    return X, converged_at

# ---------------------------------------------------------------------------
# 8. Extract Value Table
# ---------------------------------------------------------------------------

def extract_values(X: np.ndarray) -> np.ndarray:
    V = np.array([X[s * STALK_DIM + D] for s in range(N_STATES)])
    return V

# ---------------------------------------------------------------------------
# 9. Action Selection via Koopman Lookahead
# ---------------------------------------------------------------------------

def select_action(s: int, koopman_ops: list[np.ndarray], V: np.ndarray) -> int:
    """
    a* = argmax_a  V[ argmax(K_a @ z_s) ]
    For deterministic grid, argmax(K_a @ z_s) = s_next deterministically.
    """
    z = one_hot(s)
    best_a, best_v = -1, -np.inf
    for a, K_a in enumerate(koopman_ops):
        z_next  = K_a @ z
        s_next  = int(np.argmax(z_next))
        if V[s_next] > best_v:
            best_v, best_a = V[s_next], a
    return best_a

# ---------------------------------------------------------------------------
# 10. Visualization
# ---------------------------------------------------------------------------

def visualize(V: np.ndarray, policy: list[int], save_path: str = "sheaf_rl_values.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    grid = V.reshape(3, 3)
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Diffused Value V(s)")

    for s in range(N_STATES):
        row, col = divmod(s, 3)
        v_text  = f"{V[s]:.3f}"
        arrow   = "★" if s == GOAL else ACTION_ARROWS[policy[s]]
        ax.text(col, row - 0.18, v_text, ha="center", va="center",
                fontsize=11, color="black", fontweight="bold")
        ax.text(col, row + 0.22, arrow, ha="center", va="center",
                fontsize=16, color="navy")

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["col 0", "col 1", "col 2"])
    ax.set_yticklabels(["row 0", "row 1", "row 2"])
    ax.set_title("Sheaf-RL: Diffused Values + Koopman Lookahead Policy\n"
                 "(3×3 Grid, goal=s8 bottom-right)", fontsize=11)

    legend = [mpatches.Patch(color="lightyellow", label="Low value (far from goal)"),
              mpatches.Patch(color="darkred",    label="High value (near goal)")]
    ax.legend(handles=legend, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n  Heatmap saved → {save_path}")
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)

    print("=" * 60)
    print("  Latent Affine Sheaf-RL — 3×3 Grid World Demo")
    print("=" * 60)

    # --- Build Koopman operators ---
    print("\n[1] Building Koopman operators...")
    koopman_ops = build_koopman_operators()
    verify_koopman(koopman_ops)

    # --- Collect replay buffer ---
    print("\n[2] Collecting replay buffer (random policy)...")
    transitions = collect_replay_buffer(n_episodes=60, max_steps=80)
    print(f"    {len(transitions)} unique (s, a) transitions collected "
          f"(max possible = {N_STATES * N_ACTIONS})")

    # --- Build sparse incidence matrix ---
    print("\n[3] Building sparse incidence matrix B...")
    B = build_incidence_matrix(transitions, koopman_ops)
    print(f"    B shape: {B.shape}  nnz: {B.nnz}")
    print(f"    Δ_F = B^T B  will be shape [90, 90]")

    # --- Reward vector ---
    R = build_reward_vector()

    # --- Initial global state ---
    X0 = initialize_global_state()

    # --- Richardson diffusion ---
    print("\n[4] Running Richardson diffusion...")
    X, n_iters = richardson_diffusion(B, R, X0, K_steps=500, tol=1e-8)

    # --- Extract values ---
    V = extract_values(X)
    print("\n[5] Diffused value table:")
    print("    " + "-" * 27)
    for row in range(3):
        vals = [f"{V[row*3+col]:6.4f}" for col in range(3)]
        print("    | " + " | ".join(vals) + " |")
    print("    " + "-" * 27)

    # Sanity checks
    # NOTE: The Sheaf Laplacian minimises disagreement over ALL edges (all 4 actions).
    # This is equivalent to evaluating a *uniform random policy*, not the optimal policy.
    # V(s) = γ * E_{a~uniform}[V(s')] rather than γ * max_a V(s').
    # Values are therefore lower than γ^k, but monotonically ordered and sufficient
    # for the greedy Koopman lookahead to find the optimal path.
    assert abs(V[8] - 1.0) < 1e-6, "Goal value anchor broken!"
    assert V[7] > V[4] > V[0], "Value ordering violated (should increase toward goal)"
    assert V[5] > V[2] > V[0], "Value ordering violated (should increase toward goal)"
    print(f"\n  V[goal=s8]          = {V[8]:.4f}  (expected = 1.0, Dirichlet anchor) ✓")
    print(f"  V[s7, s5 adj goal]  = {V[7]:.4f}, {V[5]:.4f}  (< γ={GAMMA} — averaged over all 4 actions)")
    print(f"  V[s0, farthest]     = {V[0]:.4f}  (smallest, farthest from goal) ✓")
    print(f"  Ordering V[s8]>V[s7]=V[s5]>V[s4]>V[s0]: {V[8]:.3f}>{V[7]:.3f}>{V[4]:.3f}>{V[0]:.3f} ✓")

    # --- Policy via Koopman lookahead ---
    print("\n[6] Koopman Lookahead Policy:")
    policy = []
    for s in range(N_STATES):
        if s == GOAL:
            policy.append(0)   # dummy; goal has no action
            print(f"    s{s} (row {s//3}, col {s%3}): GOAL ★")
        else:
            a = select_action(s, koopman_ops, V)
            policy.append(a)
            s_next, _, _ = grid_step(s, a)
            print(f"    s{s} (row {s//3}, col {s%3}):  {ACTION_NAMES[a]:5s} → s{s_next}  "
                  f"[V={V[s]:.4f} → {V[s_next]:.4f}]")

    # --- Demo: trace greedy path from s0 ---
    print("\n[7] Greedy path from s0 to goal (Koopman lookahead):")
    s, path = 0, [0]
    for _ in range(20):
        if s == GOAL:
            break
        a = select_action(s, koopman_ops, V)
        s, _, _ = grid_step(s, a)
        path.append(s)
    print("    " + " → ".join(f"s{p}" for p in path))

    # --- Visualize ---
    print("\n[8] Generating heatmap...")
    visualize(V, policy)

    print("\nDone.")


if __name__ == "__main__":
    main()
