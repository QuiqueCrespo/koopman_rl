"""
Latent Affine Sheaf-RL: Neural Q-Stalk (3x3 Grid World)
========================================================
Combines the Q-stalk structural fix with learned neural components.

Two changes from sheaf_rl_qstalk.py (tabular):
  1. Encoder f_θ  — one_hot(s) → z ∈ R^d   (replaces fixed one-hot identity)
  2. Koopman K_a  — learned d×d matrices     (replaces hard-coded permutations)

Two changes from sheaf_rl_neural.py (V-stalk neural):
  3. QNetwork Q_ψ — z → R^|A|               (replaces scalar ValueNetwork)
  4. Q-evaluation — directed sparse solve    (replaces Richardson / symmetric Δ_F)

The directed Q-solver is unchanged from the tabular version: it is purely a
function of transitions and the current greedy policy a*; it does NOT use K_a
or f_θ. The neural components enter through three channels only:
  (a) Koopman loss trains f_θ and K_a jointly.
  (b) Q loss trains Q_ψ and f_θ to match the directed Bellman targets.
  (c) Policy a*[s] = argmax_a Q_ψ(f_θ(s)) — the neural network determines
      which arm of the Q-system the directed solver couples to.

Action selection:
  • Q-network:       a* = argmax_a Q_ψ(f_θ(s))[a]          (direct lookup)
  • Koopman lookahead: a* = argmax_a max_{a'} Q_ψ(K_a f_θ(s))[a']
                       = argmax_a V*(K_a z_s)               (one-step model)
Both should agree after convergence.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Grid world (identical to all prior versions)
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
    result = []
    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            s_next, r, _ = grid_step(s, a)
            result.append((s, a, r, s_next))
    return result

X_ALL = torch.eye(N_STATES)   # [9, 9] fixed one-hot inputs

# ---------------------------------------------------------------------------
# CHANGE 1 & 3: Neural components — Encoder, QNetwork, learned K_a
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    f_θ: one_hot(s) ∈ R^9 → z ∈ R^d

    Linear (no activation): keeps the latent space maximally compatible
    with the Koopman linearity assumption K_a z_i ≈ z_j.
    """
    def __init__(self, n_states: int = N_STATES, d: int = 16):
        super().__init__()
        self.net = nn.Linear(n_states, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(nn.Module):
    """
    Q_ψ: z ∈ R^d → q ∈ R^|A|

    Replaces the scalar ValueNetwork from sheaf_rl_neural.py.
    Output q[a] = Q(s, a) for every action simultaneously.
    V*(s) = max_a Q_ψ(f_θ(s))[a] — no separate value head needed.
    """
    def __init__(self, d: int = 16, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)              # [..., |A|]

    def value(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z).max(dim=-1).values   # V*(s) = max_a Q(s,a)


class NeuralQSheafRL(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.d = d

        self.encoder = Encoder(N_STATES, d)
        self.q_net   = QNetwork(d, N_ACTIONS)

        # CHANGE 2: Learned Koopman matrices K_a ∈ R^{d×d}, one per action.
        # Same initialisation as sheaf_rl_neural.py: small random + scaled identity.
        self.K = nn.ParameterList([
            nn.Parameter(0.05 * torch.randn(d, d) + 0.2 * torch.eye(d))
            for _ in range(N_ACTIONS)
        ])

    # --- Helpers (detached numpy, used by the directed Q-solver) ---

    @torch.no_grad()
    def encode_all_numpy(self) -> np.ndarray:
        return self.encoder(X_ALL).cpu().numpy()        # [9, d]

    @torch.no_grad()
    def q_all_numpy(self) -> np.ndarray:
        z = self.encoder(X_ALL)
        return self.q_net(z).cpu().numpy()             # [9, |A|]

    @torch.no_grad()
    def a_star_numpy(self) -> np.ndarray:
        return self.q_all_numpy().argmax(axis=1)        # [9] greedy policy

    @torch.no_grad()
    def K_numpy(self) -> list[np.ndarray]:
        return [K.cpu().numpy() for K in self.K]

# ---------------------------------------------------------------------------
# Directed Q-solver — UNCHANGED from sheaf_rl_qstalk.py (pure numpy).
# Does NOT use f_θ or K_a; the neural network enters only through a_star.
# ---------------------------------------------------------------------------

def qeval_directed(transitions: list[tuple], a_star: np.ndarray) -> np.ndarray:
    """
    Exact Q-policy-evaluation under fixed greedy policy a_star.
    Builds and solves: M q = r  (sparse LU, O(N·|A|) unknowns).

    M[idx(s,a), idx(s,a)]        = 1
    M[idx(s,a), idx(s', a*[s'])] = -γ    (if s' ≠ GOAL)
    r[idx(s,a)]                  = reward of transition (s,a)

    This is the DIRECTED Bellman system — no symmetric averaging.
    One policy-improvement outer loop converges to Q* in a few iterations.
    """
    n   = N_STATES * N_ACTIONS
    idx = lambda s, a: s * N_ACTIONS + a
    M   = lil_matrix((n, n), dtype=np.float64)
    r_vec = np.zeros(n)

    for s_i, a, r, s_j in transitions:
        i = idx(s_i, a)
        M[i, i]  = 1.0
        r_vec[i] = r
        if s_j != GOAL:
            M[i, idx(s_j, int(a_star[s_j]))] = -GAMMA

    for a in range(N_ACTIONS):           # anchor terminal Q-values to 0
        i = idx(GOAL, a)
        M[i, :] = 0.0;  M[i, i] = 1.0
        r_vec[i] = 0.0

    return spsolve(csr_matrix(M), r_vec).reshape(N_STATES, N_ACTIONS)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    n_epochs:     int   = 2000,
    d:            int   = 16,
    lambda_q:     float = 1.0,
    lr:           float = 2e-3,
    solve_every:  int   = 20,    # re-solve directed Q-system every N epochs
) -> tuple:
    transitions  = all_transitions()
    model        = NeuralQSheafRL(d=d)
    optimizer    = optim.Adam(model.parameters(), lr=lr)

    # Precompute index tensors for the 36 transitions
    states_t  = torch.tensor([t[0] for t in transitions])
    actions   = [t[1] for t in transitions]
    next_t    = torch.tensor([t[3] for t in transitions])

    # Bootstrap: initial Q-targets from random model
    Q_diff   = qeval_directed(transitions, model.a_star_numpy())
    Q_diff_t = torch.tensor(Q_diff, dtype=torch.float32)

    k_losses, q_losses = [], []

    for epoch in range(n_epochs):
        # ----------------------------------------------------------------
        # Loss 1 — Koopman linearity  ||K_a f_θ(s) - f_θ(s')||²
        # Gradient flows through encoder (both sides) and K_a matrices.
        # Drives the latent space to be linear w.r.t. transition dynamics.
        # ----------------------------------------------------------------
        z_src  = model.encoder(X_ALL[states_t])    # [36, d]
        z_dst  = model.encoder(X_ALL[next_t])      # [36, d]
        z_pred = torch.stack([model.K[a] @ z for a, z in zip(actions, z_src)])
        loss_koopman = (z_pred - z_dst).pow(2).mean()

        # ----------------------------------------------------------------
        # Loss 2 — Q-value alignment  ||Q_ψ(f_θ(s)) - stop_grad(Q_diff)||²
        # Gradient flows through q_net and encoder; Q_diff is a fixed target.
        # The stop_gradient is implicit: Q_diff_t is a plain tensor, not a
        # computation graph node.
        # ----------------------------------------------------------------
        Q_pred = model.q_net(model.encoder(X_ALL))    # [9, |A|]
        loss_q = (Q_pred - Q_diff_t).pow(2).mean()

        loss = loss_koopman + lambda_q * loss_q
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        k_losses.append(loss_koopman.item())
        q_losses.append(loss_q.item())

        # Refresh Q-targets: re-solve with updated greedy policy from Q_ψ
        if epoch % solve_every == 0:
            a_star   = model.a_star_numpy()
            Q_diff   = qeval_directed(transitions, a_star)
            Q_diff_t = torch.tensor(Q_diff, dtype=torch.float32)

        if epoch % 400 == 0:
            V = Q_diff.max(axis=1)
            print(f"  epoch {epoch:5d}  "
                  f"L_koop={loss_koopman.item():.5f}  "
                  f"L_q={loss_q.item():.5f}  "
                  f"V(s0)={V[0]:.4f}  V(s7)={V[7]:.4f}")

    # Final solve with converged policy
    Q_diff = qeval_directed(transitions, model.a_star_numpy())
    return model, Q_diff, k_losses, q_losses

# ---------------------------------------------------------------------------
# Action selection — two routes, both via the neural network
# ---------------------------------------------------------------------------

def select_action_qnet(s: int, model: NeuralQSheafRL) -> int:
    """Direct Q-lookup: a* = argmax_a Q_ψ(f_θ(s))[a]."""
    with torch.no_grad():
        z = model.encoder(X_ALL[s])
        return model.q_net(z).argmax().item()


def select_action_koopman(s: int, model: NeuralQSheafRL) -> int:
    """
    Koopman lookahead: a* = argmax_a V*(K_a f_θ(s))
                          = argmax_a max_{a'} Q_ψ(K_a z_s)[a']

    Predicts the next latent with K_a, evaluates it with Q_ψ.
    This is the same lookahead from sheaf_rl_neural.py, now using
    V*(z) = max_a Q_ψ(z) instead of a separate value head.
    """
    with torch.no_grad():
        z = model.encoder(X_ALL[s])
        best_a, best_v = -1, -float("inf")
        for a in range(N_ACTIONS):
            z_next = model.K[a] @ z
            v_next = model.q_net.value(z_next.unsqueeze(0)).item()
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
    print("  Neural Q-Stalk Sheaf-RL  —  3×3 Grid World")
    print("  Changes from tabular Q-stalk: encoder f_θ + QNetwork Q_ψ + learned K_a")
    print("  Q-evaluation:  directed sparse solve M q = r (unchanged)")
    print("=" * 62)

    model, Q_diff, k_losses, q_losses = train(n_epochs=2000, d=16)

    V      = Q_diff.max(axis=1)
    policy = Q_diff.argmax(axis=1)

    # --- Q-table (from directed solver targets) ---
    print("\nFinal Q-table (directed solver targets):")
    print(f"  {'State':8s}  {'Q(Up)':7s}  {'Q(Dn)':7s}  {'Q(Lt)':7s}  {'Q(Rt)':7s}")
    print("  " + "-" * 44)
    for s in range(N_STATES):
        row = f"  s{s} r{s//3}c{s%3}  "
        for a in range(N_ACTIONS):
            row += f"{'>' if a == policy[s] else ' '}{Q_diff[s, a]:6.4f}  "
        print(row)

    # --- Sanity checks: directed Q-targets should be exact ---
    print(f"\nQ-targets vs true Bellman Q*:")
    checks = [
        ("Q(s7, Right)", Q_diff[7, 3], 1.0,     "γ^0 = 1.000"),
        ("Q(s4, Down) ", Q_diff[4, 1], GAMMA,   f"γ^1 = {GAMMA:.3f}"),
        ("Q(s3, Right)", Q_diff[3, 3], GAMMA**2,f"γ^2 = {GAMMA**2:.3f}"),
        ("V*(s0)      ", V[0],         GAMMA**3,f"γ^3 = {GAMMA**3:.3f}"),
    ]
    for name, got, want, label in checks:
        ok = abs(got - want) < 1e-6
        print(f"  {name} = {got:.6f}  (expected {label})  {'✓' if ok else '✗'}")

    # --- Neural network Q-values vs solver targets ---
    with torch.no_grad():
        Q_net = model.q_net(model.encoder(X_ALL)).cpu().numpy()
    mae = np.abs(Q_net - Q_diff).mean()
    print(f"\nNeural Q_ψ(f_θ(s)) vs directed targets — MAE: {mae:.5f}")

    # --- Compare action selection routes ---
    print("\nAction selection: Q-network vs Koopman lookahead:")
    print(f"  {'State':8s}  {'Q-net':6s}  {'Koopman':8s}  {'Agree?':6s}")
    print("  " + "-" * 36)
    disagree = 0
    for s in range(N_STATES):
        if s == GOAL:
            print(f"  s{s} r{s//3}c{s%3}   GOAL ★")
            continue
        a_q = select_action_qnet(s, model)
        a_k = select_action_koopman(s, model)
        ok  = "✓" if a_q == a_k else "✗"
        if a_q != a_k: disagree += 1
        print(f"  s{s} r{s//3}c{s%3}   {ACTION_NAMES[a_q]:5s}   {ACTION_NAMES[a_k]:5s}    {ok}")
    if disagree == 0:
        print("  → Both routes agree on all states ✓")

    # --- Greedy path ---
    s, path = 0, [0]
    for _ in range(20):
        if s == GOAL: break
        s, _, _ = grid_step(s, select_action_qnet(s, model))
        path.append(s)
    print(f"\nGreedy path (Q-net): {' → '.join(f's{p}' for p in path)}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Training losses
    axes[0].plot(k_losses, label="Koopman  $\\|K_a z_i - z_j\\|^2$", alpha=0.8)
    axes[0].plot(q_losses, label="Q-align  $\\|Q_\\psi - Q_{\\rm diff}\\|^2$", alpha=0.8)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Neural Q-Stalk: Training Losses")
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Q-network output vs directed targets
    x = np.arange(N_STATES * N_ACTIONS)
    axes[1].bar(x - 0.2, Q_net.flatten(),  0.4, label="Q_ψ(f_θ(s))", color="steelblue", alpha=0.85)
    axes[1].bar(x + 0.2, Q_diff.flatten(), 0.4, label="Directed Q*",  color="gold",      alpha=0.85)
    axes[1].set_xticks(np.arange(N_STATES) * N_ACTIONS + 1.5)
    axes[1].set_xticklabels([f"s{s}" for s in range(N_STATES)], fontsize=8)
    axes[1].set_title("Q_ψ output vs Directed Bellman Targets\n(per state, 4 bars = 4 actions)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis="y")

    # Value heatmap
    im = axes[2].imshow(V.reshape(3, 3), cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[2], label="V*(s) = max_a Q*(s,a)")
    for s in range(N_STATES):
        r, c = divmod(s, 3)
        arrow = "★" if s == GOAL else ACTION_ARROWS[policy[s]]
        axes[2].text(c, r - 0.15, f"{V[s]:.3f}",
                     ha="center", fontsize=10, fontweight="bold")
        axes[2].text(c, r + 0.22, arrow, ha="center", fontsize=15, color="navy")
    axes[2].set_xticks([0, 1, 2]); axes[2].set_yticks([0, 1, 2])
    axes[2].set_title("Neural Q-Stalk: Optimal Values + Policy\n"
                      "(learned f_θ + Q_ψ + K_a, exact Bellman Q*)")

    plt.tight_layout()
    plt.savefig("sheaf_rl_qstalk_neural_values.png", dpi=150)
    print("\nSaved → sheaf_rl_qstalk_neural_values.png")
    plt.close()


if __name__ == "__main__":
    main()
