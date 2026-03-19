"""
Latent Affine Koopman-RL: Neural Extension (3x3 Grid World)
==========================================================
Two changes from koopman_rl_gridworld.py:
  1. Encoder f_θ  — one_hot(s) → z ∈ R^d  (replaces fixed one-hot identity)
  2. Koopman K_a  — learned d×d matrices   (replaces hard-coded permutations)

Everything else is structurally identical:
  - build_incidence_matrix()  (same block layout, now uses learned K_a)
  - richardson_diffusion()    (same Richardson loop, same Dirichlet anchors)
  - select_action()           (same Koopman lookahead)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Grid world (identical to tabular version)
# ---------------------------------------------------------------------------
N_STATES  = 9
N_ACTIONS = 4
GOAL      = 8
GAMMA     = 0.9
ACTION_NAMES  = ["Up", "Down", "Left", "Right"]
ACTION_ARROWS = ["↑",  "↓",   "←",   "→"]

def grid_step(s: int, a: int) -> tuple[int, float, bool]:
    row, col = divmod(s, 3)
    if a == 0:   row = max(0, row - 1)
    elif a == 1: row = min(2, row + 1)
    elif a == 2: col = max(0, col - 1)
    elif a == 3: col = min(2, col + 1)
    s_next = row * 3 + col
    return s_next, (1.0 if s_next == GOAL else 0.0), (s_next == GOAL)

def all_transitions() -> list[tuple]:
    """All 36 unique (s, a, r, s_next) tuples for the 3x3 grid."""
    result = []
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            s_next, r, _ = grid_step(s, a)
            result.append((s, a, r, s_next))
    return result

X_ALL = torch.eye(N_STATES)   # [9, 9] one-hot inputs, shared constant

# ---------------------------------------------------------------------------
# CHANGE 1: Neural components (encoder + value net)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    f_θ: one_hot(s) ∈ R^9  →  z ∈ R^d

    Deliberately linear (no activation) so the Koopman loss can train
    a latent space where transitions are genuinely linear operators.
    A nonlinearity here fights the linearity assumption.
    """
    def __init__(self, n_states: int = N_STATES, d: int = 16):
        super().__init__()
        self.net = nn.Linear(n_states, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ValueNetwork(nn.Module):
    """V_ψ: z ∈ R^d → scalar value ∈ R"""
    def __init__(self, d: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)

class NeuralKoopmanRL(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.d = d

        self.encoder   = Encoder(N_STATES, d)
        self.value_net = ValueNetwork(d)

        # CHANGE 2: Learnable Koopman matrices K_a ∈ R^{d×d}, one per action.
        # Init: small random + scaled identity → stable early dynamics.
        self.K = nn.ParameterList([
            nn.Parameter(0.05 * torch.randn(d, d) + 0.2 * torch.eye(d))
            for _ in range(N_ACTIONS)
        ])

    # --- Convenience helpers ---

    @torch.no_grad()
    def encode_all_numpy(self) -> np.ndarray:
        """Z[s] = f_θ(one_hot(s))  →  [9, d] numpy (detached)."""
        return self.encoder(X_ALL).cpu().numpy()

    @torch.no_grad()
    def values_all_numpy(self, Z: np.ndarray) -> np.ndarray:
        """V_ψ(z) for every state  →  [9] numpy (detached)."""
        z_t = torch.tensor(Z, dtype=torch.float32)
        return self.value_net(z_t).cpu().numpy()

    @torch.no_grad()
    def K_numpy(self) -> list[np.ndarray]:
        return [K.cpu().numpy() for K in self.K]

# ---------------------------------------------------------------------------
# Incidence matrix — structurally IDENTICAL to tabular version.
# Only difference: K_numpy[a] is learned, not a permutation matrix.
# ---------------------------------------------------------------------------

def build_incidence_matrix(
    transitions: list[tuple],
    K_numpy: list[np.ndarray],
    d: int,
    gamma: float = GAMMA,
) -> csr_matrix:
    stalk_dim = d + 1
    E = len(transitions)
    B = lil_matrix((E * stalk_dim, N_STATES * stalk_dim), dtype=np.float64)

    for k, (s_i, a, _r, s_j) in enumerate(transitions):
        row0  = k * stalk_dim
        col_i = s_i * stalk_dim
        col_j = s_j * stalk_dim
        K_a   = K_numpy[a]

        # Source block: upper-left d×d = K_a, bottom-right = 1
        B[row0 : row0 + d, col_i : col_i + d] = K_a
        B[row0 + d, col_i + d] = 1.0

        # Dest block: upper-left d×d = -I, bottom-right = -γ  (negated: src − dst)
        for i in range(d):
            B[row0 + i, col_j + i] = -1.0
        B[row0 + d, col_j + d] = -gamma

    return csr_matrix(B)

# ---------------------------------------------------------------------------
# Richardson diffusion — structurally IDENTICAL to tabular version.
# Latents anchored to current f_θ output; only value slots evolve.
# ---------------------------------------------------------------------------

def diffuse_values(
    model: NeuralKoopmanRL,
    transitions: list[tuple],
    K_steps: int = 300,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Run the graph diffusion solver with current (detached) model parameters.
    Returns V_diff ∈ R^9  — the diffused value targets for all states.
    """
    d         = model.d
    stalk_dim = d + 1

    Z_all = model.encode_all_numpy()           # [9, d]  detached
    V_all = model.values_all_numpy(Z_all)      # [9]     detached
    K_np  = model.K_numpy()                   # list of [d, d]

    B       = build_incidence_matrix(transitions, K_np, d)
    Delta_F = B.T @ B                          # [9*(d+1), 9*(d+1)] sparse

    # Optimal Richardson step: α = 1 / λ_max(Δ_F)
    lam_max, _ = eigsh(Delta_F, k=1, which="LM", tol=1e-3, maxiter=500)
    alpha = 1.0 / float(lam_max[0])

    # Initialize X: fixed one-hot latents + current value estimates
    X = np.zeros(N_STATES * stalk_dim)
    for s in range(N_STATES):
        X[s * stalk_dim : s * stalk_dim + d] = Z_all[s]
        X[s * stalk_dim + d] = V_all[s]
    X[GOAL * stalk_dim + d] = 1.0   # Dirichlet seed

    R = np.zeros(N_STATES * stalk_dim)
    R[GOAL * stalk_dim + d] = 1.0   # reward source at goal value slot

    for _ in range(K_steps):
        X_new = X - alpha * (Delta_F @ X) + alpha * R
        # Dirichlet anchors: latents fixed to encoder output, goal value fixed
        for s in range(N_STATES):
            X_new[s * stalk_dim : s * stalk_dim + d] = Z_all[s]
        X_new[GOAL * stalk_dim + d] = 1.0
        if np.max(np.abs(X_new - X)) < tol:
            break
        X = X_new

    return np.array([X[s * stalk_dim + d] for s in range(N_STATES)])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    n_epochs: int = 1500,
    d: int = 16,
    lambda_val: float = 1.0,
    lr: float = 2e-3,
    diffuse_every: int = 30,
) -> tuple:
    transitions = all_transitions()
    model       = NeuralKoopmanRL(d=d)
    optimizer   = optim.Adam(model.parameters(), lr=lr)

    states_idx      = torch.tensor([t[0] for t in transitions])
    actions_idx     = [t[1] for t in transitions]
    next_states_idx = torch.tensor([t[3] for t in transitions])

    # Initial diffusion target
    V_diff   = diffuse_values(model, transitions)
    V_diff_t = torch.tensor(V_diff, dtype=torch.float32)

    k_losses, v_losses = [], []

    for epoch in range(n_epochs):
        # ----------------------------------------------------------------
        # Loss 1 — Koopman linearity  ||K_a f_θ(s) − f_θ(s')||²
        # Gradients flow through encoder (both sides) and K_a matrices.
        # ----------------------------------------------------------------
        z_src  = model.encoder(X_ALL[states_idx])       # [36, d]
        z_dst  = model.encoder(X_ALL[next_states_idx])  # [36, d]
        z_pred = torch.stack([model.K[a] @ z for a, z in zip(actions_idx, z_src)])
        loss_koopman = ((z_pred - z_dst) ** 2).mean()

        # ----------------------------------------------------------------
        # Loss 2 — Value targets  ||V_ψ(f_θ(s)) − stop_grad(V_diff[s])||²
        # Gradients flow through value_net and encoder; V_diff is detached.
        # ----------------------------------------------------------------
        z_all  = model.encoder(X_ALL)     # [9, d]
        V_pred = model.value_net(z_all)   # [9]
        loss_value = ((V_pred - V_diff_t.detach()) ** 2).mean()

        loss = loss_koopman + lambda_val * loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        k_losses.append(loss_koopman.item())
        v_losses.append(loss_value.item())

        # Periodically refresh diffusion targets with updated model
        if epoch % diffuse_every == 0:
            V_diff   = diffuse_values(model, transitions)
            V_diff_t = torch.tensor(V_diff, dtype=torch.float32)

        if epoch % 300 == 0:
            vmin, vmax = V_diff.min(), V_diff.max()
            print(f"  epoch {epoch:5d}  "
                  f"L_koopman={loss_koopman.item():.5f}  "
                  f"L_value={loss_value.item():.5f}  "
                  f"V_diff=[{vmin:.3f}, {vmax:.3f}]")

    # Final diffusion pass with converged model
    V_diff = diffuse_values(model, transitions)
    return model, k_losses, v_losses, V_diff

# ---------------------------------------------------------------------------
# Action selection — structurally IDENTICAL to tabular version.
# ---------------------------------------------------------------------------

def select_action(s: int, model: NeuralKoopmanRL) -> int:
    """a* = argmax_a  V_ψ(K_a · f_θ(s))"""
    with torch.no_grad():
        z = model.encoder(X_ALL[s])   # [d]
        best_a, best_v = -1, -float("inf")
        for a in range(N_ACTIONS):
            z_next = model.K[a] @ z
            v_next = model.value_net(z_next.unsqueeze(0)).item()
            if v_next > best_v:
                best_v, best_a = v_next, a
    return best_a

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 62)
    print("  Neural Koopman-RL  —  3×3 Grid World")
    print("  Changes from tabular: encoder f_θ + learned K_a")
    print("  Unchanged: incidence matrix, Richardson solver, anchors")
    print("=" * 62)

    model, k_losses, v_losses, V_diff = train(n_epochs=1500, d=16)

    print("\nFinal diffused value table:")
    print("  " + "-" * 29)
    for row in range(3):
        vals = [f"{V_diff[row*3+col]:7.4f}" for col in range(3)]
        print("  | " + " | ".join(vals) + " |")
    print("  " + "-" * 29)

    print(f"\n  V[goal=s8]      = {V_diff[8]:.4f}  (expected = 1.0) ✓")
    assert abs(V_diff[8] - 1.0) < 0.01, "Goal anchor drifted!"
    assert V_diff[8] > V_diff[7] > V_diff[4] > V_diff[0], "Ordering broken!"
    print(f"  Ordering s8>s7>s4>s0: "
          f"{V_diff[8]:.3f} > {V_diff[7]:.3f} > {V_diff[4]:.3f} > {V_diff[0]:.3f} ✓")

    print("\nKoopman lookahead policy (neural):")
    policy = []
    for s in range(N_STATES):
        if s == GOAL:
            policy.append(0)
            print(f"  s{s} (r{s//3},c{s%3}): GOAL ★")
        else:
            a = select_action(s, model)
            policy.append(a)
            s_next, _, _ = grid_step(s, a)
            print(f"  s{s} (r{s//3},c{s%3}): {ACTION_NAMES[a]:5s} → s{s_next}")

    s, path = 0, [0]
    for _ in range(20):
        if s == GOAL: break
        a = select_action(s, model)
        s, _, _ = grid_step(s, a)
        path.append(s)
    print(f"\nGreedy path s0 → goal: {' → '.join(f's{p}' for p in path)}")

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(k_losses, label="Koopman loss  $\\|K_a z_i - z_j\\|^2$", alpha=0.8)
    ax1.plot(v_losses, label="Value loss  $\\|V_\\psi - V_{\\mathrm{diff}}\\|^2$", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Neural Koopman-RL: Training Losses")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    im = ax2.imshow(V_diff.reshape(3, 3), cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax2, label="Diffused Value V(s)")
    for s in range(N_STATES):
        r, c = divmod(s, 3)
        arrow = "★" if s == GOAL else ACTION_ARROWS[policy[s]]
        ax2.text(c, r - 0.15, f"{V_diff[s]:.3f}",
                 ha="center", fontsize=10, fontweight="bold")
        ax2.text(c, r + 0.22, arrow, ha="center", fontsize=15, color="navy")
    ax2.set_xticks([0, 1, 2]);  ax2.set_yticks([0, 1, 2])
    ax2.set_title("Diffused Values + Koopman Lookahead Policy\n"
                  "(learned encoder + learned $K_a$)")

    plt.tight_layout()
    plt.savefig("koopman_neural_values.png", dpi=150)
    print("\nSaved → koopman_neural_values.png")
    plt.close()


if __name__ == "__main__":
    main()
