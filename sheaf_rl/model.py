"""
Neural components: Encoder, ValueNetwork, QNetwork, SheafAgent, TargetNetwork.
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sheaf_rl.config import Config, ModelConfig, EnvConfig

# Module-level defaults (backward compat)
N_ACTIONS = 4
STATE_DIM = 2
D         = 32
LR        = 3e-4
EMA_TAU   = 0.005


class Encoder(nn.Module):
    """
    f_θ: (x,y) ∈ R² → z ∈ R^d, ||z|| = 1  (unit hypersphere)

    L2 normalisation prevents constant-vector collapse without requiring
    orthogonal constraints on K_a. All latents live on S^{d-1}, consistent
    with F.normalize(K_a z) outputs from the linear dynamics model.
    """
    def __init__(self, state_dim: int = STATE_DIM, d: int = D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),        nn.Tanh(),
            nn.Linear(64, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out if getattr(self, "no_normalize", False) else F.normalize(out, dim=-1)


class QNetwork(nn.Module):
    """Q_ψ: z ∈ R^d → q ∈ R^|A|.  Kept for benchmark.py DQN baseline."""
    def __init__(self, d: int = D, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def value(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z).max(dim=-1).values


class ValueNetwork(nn.Module):
    """
    V_ψ: z ∈ R^d → v ∈ R  (scalar state value).

    Action selection is handled entirely by the linear dynamics matrices:
        a* = argmax_a  V_ψ(normalize(A z_t + B[:,a]))
    """
    def __init__(self, d: int = D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class SheafAgent(nn.Module):
    """
    Online network: encoder f_θ + value network V_ψ + linear dynamics (A, B).

    Latent transition model:
        z_{t+1} = normalize(A z_t + B a_t)

    A ∈ R^{d×d}  — shared state dynamics (orthogonal init: random rotation)
    B ∈ R^{d×|A|} — action input matrix (columns orthonormal at init)
    a_t           — one-hot for discrete actions

    Action selection — linear lookahead (vectorised):
        a* = argmax_a  V_ψ(normalize(A z_t + B e_a))
    """
    def __init__(self, state_dim: int = STATE_DIM, d: int = D,
                 n_actions: int = N_ACTIONS):
        super().__init__()
        self.d        = d
        self.n_actions = n_actions
        self.encoder  = Encoder(state_dim, d)
        self.v_net    = ValueNetwork(d)

        self.A = nn.Parameter(torch.eye(d))          # [d, d], identity init

        # B columns are orthonormal (d > n_actions, so orthogonal_ gives ortho cols)
        B = torch.empty(d, n_actions)
        nn.init.orthogonal_(B)
        self.B = nn.Parameter(B)                     # [d, n_actions]

    @classmethod
    def from_cfg(cls, cfg: Config) -> "SheafAgent":
        return cls(
            state_dim=cfg.env.state_dim,
            d=cfg.model.d,
            n_actions=cfg.env.n_actions,
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.v_net(self.encoder(state))

    def transition(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z' = A z + B a   (unnormalised — for use in L_koop only)

        L_koop = ‖A z + B a − z_dst‖² is NOT scale-invariant, so the loss itself
        bounds ‖A‖ and ‖B‖ without weight decay.
        """
        return z @ self.A.T + a @ self.B.T

    def transition_norm(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Normalised transition for action selection — consistent with V_ψ training."""
        return F.normalize(z @ self.A.T + a @ self.B.T, dim=-1)

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        # Push tensor to whatever device the model lives on
        device = next(self.parameters()).device
        z      = self.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        )  # [1, d]
        a_eye = torch.eye(self.n_actions, device=device)  # [A, A]

        # Normalised for V_ψ: [A, d] via broadcast [1,d] + [A,d]
        z_next = F.normalize(z @ self.A.T + a_eye @ self.B.T, dim=-1)  # [A, d]
        v_next = self.v_net(z_next)                                      # [A]

        v_next += torch.randn_like(v_next) * 1e-6
        return v_next.argmax().item()


class TargetNetwork:
    """
    EMA copy of encoder + V-net.  Not an nn.Module — excluded from optimizer
    parameters automatically.  A and B are NOT tracked: bootstrap value
    V_target(s') = V_ψ_target(enc_target(s')) doesn't require the dynamics model.
    """
    def __init__(self, agent: SheafAgent):
        self.encoder = copy.deepcopy(agent.encoder).eval()
        self.v_net   = copy.deepcopy(agent.v_net).eval()
        for p in self.encoder.parameters(): p.requires_grad_(False)
        for p in self.v_net.parameters():   p.requires_grad_(False)

    @torch.no_grad()
    def update(self, agent: SheafAgent, tau: float = EMA_TAU) -> None:
        for p_t, p_o in zip(self.encoder.parameters(), agent.encoder.parameters()):
            p_t.data.mul_(1 - tau).add_(p_o.data, alpha=tau)
        for p_t, p_o in zip(self.v_net.parameters(), agent.v_net.parameters()):
            p_t.data.mul_(1 - tau).add_(p_o.data, alpha=tau)

    @torch.no_grad()
    def v_target(self, states: torch.Tensor) -> torch.Tensor:
        return self.v_net(self.encoder(states))
