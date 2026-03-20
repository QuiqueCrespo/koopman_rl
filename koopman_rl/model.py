"""
Neural components: Encoder, ValueNetwork, QNetwork, KoopmanGradientPlanner, TargetNetwork.

NOTE — ortho_a device dispatch:
  ortho_a=True uses differentiable SVD Procrustes on CUDA and a soft penalty on MPS/CPU.

  CUDA → _SVDOrthogonal parametrization: A = U Vᵀ  from  W = U Σ Vᵀ.
    Covers all of O(d) (det=±1).  W initialised ~ N(0,1/d) so singular
    values are generically distinct → stable backward pass.
    _use_hard_ortho=True; soft penalty never applied.

  MPS / CPU → soft penalty  λ · ||AᵀA − I||²_F  added to the loss.
    torch.linalg.svd falls back to CPU on MPS; the penalty avoids the
    device round-trip.  _use_hard_ortho=False.

  Device is passed at construction via from_cfg(cfg, device=device).
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from koopman_rl.config import Config, ModelConfig, EnvConfig

# Module-level defaults (backward compat)
N_ACTIONS = 4
STATE_DIM = 2
D         = 32
LR        = 3e-4
EMA_TAU   = 0.005


class Encoder(nn.Module):
    """
    f_θ: state → z ∈ R^d.

    Three output modes (mutually exclusive, controlled at construction):
      tanh_out=True   : z = tanh(linear(h)), each dim ∈ (-1,1), ||z||≤√d
      no_normalize    : z = linear(h)  (raw, set as attribute after init)
      default         : z = normalize(linear(h))  (unit hypersphere)

    Last linear layer uses orthogonal init so initial encodings are
    well-spread across the output space from step 0.
    """
    def __init__(self, state_dim: int = STATE_DIM, d: int = D, tanh_out: bool = False):
        super().__init__()
        layers = [nn.Linear(state_dim, 64), nn.Tanh(),
                  nn.Linear(64, 64),        nn.Tanh(),
                  nn.Linear(64, d)]
        if tanh_out:
            layers.append(nn.Tanh())
        self.net      = nn.Sequential(*layers)
        self.tanh_out = tanh_out

        # Orthogonal init on the last linear: isometric map → well-spread
        # initial encodings on the sphere (or hypercube for tanh_out).
        last_linear = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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

    Output layer initialised with small weights so V starts near-zero and
    the value landscape is nearly linear at training onset (standard deep
    RL trick; improves early-training stability and OOD robustness for
    off-sphere Koopman predictions).
    """
    def __init__(self, d: int = D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class _SVDOrthogonal(nn.Module):
    """
    Differentiable Procrustes SVD parametrization:  W → U Vᵀ
    where  W = U Σ Vᵀ  (thin SVD via torch.linalg.svd).

    The result A = U Vᵀ is exactly in O(d).  Gradient flows through the SVD
    via PyTorch autograd.

    Gradient stability note:
      The SVD backward contains 1/(σᵢ²−σⱼ²) terms.  These blow up when any
      two singular values of W are equal.  We therefore require W to be
      initialised as a random Gaussian matrix (see KoopmanGradientPlanner),
      NOT as identity or an orthogonal matrix (all σ=1 → immediately NaN).
      After the first optimiser step, W leaves the degenerate manifold and
      singular values are generically distinct for all subsequent steps.
    """
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        return U @ Vh


class KoopmanGradientPlanner(nn.Module):
    """
    Formerly SheafAgent. Renamed to reflect primary usage: MPC via learned
    Koopman latent dynamics (A, B) + value network V_ψ.

    Latent transition model:
        ortho_a=False (default): z_{t+1} = normalize(A z_t + B a_t)
        ortho_a=True  (linear):  z_{t+1} = A z_t + B a_t,  A ∈ O(d)
            CUDA  → hard constraint via Cayley parametrization (exact, no penalty)
            MPS/CPU → soft penalty ||AᵀA − I||²_F in the loss (see ortho_penalty())

    A ∈ R^{d×d}   — shared state dynamics
    B ∈ R^{d×|A|} — action input matrix (cols orthonormal at init)
    a_t           — one-hot for discrete actions, or continuous vector for cont. envs

    Use dyn_step(z, b_vec) for all transition computations — it dispatches the
    normalize-or-not decision centrally so callers stay mode-agnostic.
    """
    def __init__(self, state_dim: int = STATE_DIM, d: int = D,
                 n_actions: int = N_ACTIONS, ortho_a: bool = False,
                 tanh_out: bool = False, device=None):
        super().__init__()
        self.d         = d
        self.n_actions = n_actions
        self._ortho_a  = ortho_a
        self.encoder   = Encoder(state_dim, d, tanh_out=tanh_out)
        self.v_net     = ValueNetwork(d)
        self.decoder   = nn.Linear(d, state_dim)

        # CUDA: SVD Procrustes hard constraint (covers full O(d), det=±1).
        # MPS/CPU: unconstrained A + soft penalty in loss (no linalg.svd round-trip).
        dev_type = torch.device(device).type if device is not None else "cpu"
        self._use_hard_ortho = ortho_a and dev_type == "cuda"

        if self._use_hard_ortho:
            # W ~ N(0, 1/d): generically distinct singular values → stable SVD backward.
            # Identity / orthogonal init gives all σ=1 → 1/(σᵢ²−σⱼ²) = NaN.
            self._A_layer = nn.Linear(d, d, bias=False)
            nn.init.normal_(self._A_layer.weight, std=1.0 / (d ** 0.5))
            parametrize.register_parametrization(self._A_layer, 'weight', _SVDOrthogonal())
        else:
            # Unconstrained; soft penalty applied in loss when ortho_a=True.
            self.A = nn.Parameter(torch.eye(d))

        B = torch.empty(d, n_actions)
        nn.init.orthogonal_(B)
        self.B = nn.Parameter(B)

    def __getattr__(self, name: str):
        # With hard constraint, 'A' is not a bare Parameter — return the O(d) weight.
        if name == 'A' and self.__dict__.get('_use_hard_ortho', False):
            return self._modules['_A_layer'].weight
        return super().__getattr__(name)

    def ortho_penalty(self) -> torch.Tensor:
        """Soft ||AᵀA − I||²_F penalty (MPS/CPU only — never called when _use_hard_ortho)."""
        A = self.A
        return (A.T @ A - torch.eye(self.d, device=A.device)).pow(2).sum()

    def ortho_error(self) -> float:
        """||AᵀA − I||²_F as a plain float. Device-safe — uses A.device for the identity."""
        with torch.no_grad():
            A = self.A
            return (A.T @ A - torch.eye(self.d, device=A.device)).pow(2).sum().item()

    def dyn_step(self, z: torch.Tensor, b_vec: torch.Tensor) -> torch.Tensor:
        """
        One Koopman step: z' = A z + b_vec  


        All callers (planner, train loop, act) go through here so switching
        modes requires changing only the config, not any call site.
        """
        return z @ self.A.T + b_vec

    def koop_parameters(self) -> list:
        """Parameter list for the Koopman optimizer group (A and B).
        With hard ortho, returns the Cayley pre-image parameters (not the read-only O(d) weight)."""
        A_params = list(self._A_layer.parameters()) if self._use_hard_ortho else [self.A]
        return A_params + [self.B]

    @classmethod
    def from_cfg(cls, cfg: Config, device=None) -> "KoopmanGradientPlanner":
        return cls(
            state_dim=cfg.env.state_dim,
            d=cfg.model.d,
            n_actions=cfg.env.n_actions,
            ortho_a=cfg.model.ortho_a,
            tanh_out=cfg.model.tanh_out,
            device=device,
        )

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.v_net(self.encoder(state))

    def transition(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """z' = A z + B a  (unnormalised — for L_koop loss)."""
        return z @ self.A.T + a @ self.B.T

    def transition_norm(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Mode-aware normalised transition."""
        return self.dyn_step(z, a @ self.B.T)

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)

        device = next(self.parameters()).device
        z      = self.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        )  # [1, d]
        a_eye  = torch.eye(self.n_actions, device=device)  # [A, A]

        # [A, d] via broadcast: dyn_step dispatches normalize/linear
        z_next = self.dyn_step(z, a_eye @ self.B.T)
        v_next = self.v_net(z_next)                         # [A]

        v_next += torch.randn_like(v_next) * 1e-6
        return v_next.argmax().item()

    def act_plan_discrete(self, state: np.ndarray,
                          horizon: int = 10, plan_iters: int = 20,
                          tau: float = 1.0) -> int:
        """Gumbel-Softmax MPC — discrete environments."""
        from koopman_rl.planner import plan_action_gumbel
        return plan_action_gumbel(self, state, horizon, plan_iters, tau=tau)

    def act_plan_toeplitz(self, state: np.ndarray,
                          horizon: int = 10, plan_iters: int = 20,
                          lr: float = 0.1, tau: float = 1.0,
                          gamma: float = 0.95) -> int:
        """Block-Toeplitz GEMM planner — requires ortho_a=True.
        Pre-computes ZIR and W_toeplitz outside the Adam loop; each iteration
        is a single dense GEMM instead of H sequential dyn_step calls."""
        from koopman_rl.planner import plan_action_toeplitz
        return plan_action_toeplitz(self, state, horizon, plan_iters,
                                    lr=lr, tau=tau, gamma=gamma)

    def act_plan_continuous(self, state: np.ndarray,
                            horizon: int = 10, plan_iters: int = 20,
                            action_scale: float = 1.0,
                            frozen_b: bool = False) -> np.ndarray:
        """tanh-squash MPC — continuous environments. Returns action ∈ [-scale, scale]^d."""
        from koopman_rl.planner import plan_action_continuous
        return plan_action_continuous(self, state, horizon, plan_iters,
                                      frozen_b=frozen_b) * action_scale

    def act_plan_toeplitz_continuous(self, state: np.ndarray,
                                     horizon: int = 10, plan_iters: int = 20,
                                     gamma: float = 0.95, action_scale: float = 1.0,
                                     cumulative: bool = True) -> np.ndarray:
        """Block-Toeplitz GEMM MPC — continuous environments, requires ortho_a=True.
        cumulative=False uses terminal V only (stable for data collection);
        cumulative=True uses discounted sum over horizon (better at eval)."""
        from koopman_rl.planner import plan_action_toeplitz_continuous
        return plan_action_toeplitz_continuous(self, state, horizon, plan_iters,
                                               gamma=gamma, action_scale=action_scale,
                                               cumulative=cumulative)

    def act_plan_toeplitz_continuous_batch(self, states: np.ndarray,
                                           horizon: int = 10, plan_iters: int = 20,
                                           gamma: float = 0.95, action_scale: float = 1.0,
                                           cumulative: bool = True) -> np.ndarray:
        """Batched Block-Toeplitz GEMM MPC for N states. W_toeplitz built once.
        Returns [N, action_dim]. Requires ortho_a=True."""
        from koopman_rl.planner import plan_action_toeplitz_continuous_batch
        return plan_action_toeplitz_continuous_batch(self, states, horizon, plan_iters,
                                                     gamma=gamma, action_scale=action_scale,
                                                     cumulative=cumulative)

    def act_plan_continuous_batch(self, states: np.ndarray,
                                  horizon: int = 10, plan_iters: int = 20,
                                  action_scale: float = 1.0,
                                  frozen_b: bool = False) -> np.ndarray:
        """Batched sequential MPC for N states. Returns [N, action_dim]."""
        from koopman_rl.planner import plan_action_continuous_batch
        return plan_action_continuous_batch(self, states, horizon, plan_iters,
                                            action_scale=action_scale, frozen_b=frozen_b)


# Backward-compatibility alias: KoopmanAgent is an alias for KoopmanGradientPlanner
KoopmanAgent = KoopmanGradientPlanner


class TargetNetwork:
    """
    EMA copy of encoder + V-net.  Not an nn.Module — excluded from optimizer
    parameters automatically.  A and B are NOT tracked: bootstrap value
    V_target(s') = V_ψ_target(enc_target(s')) doesn't require the dynamics model.
    """
    def __init__(self, agent: KoopmanGradientPlanner):
        self.encoder = copy.deepcopy(agent.encoder).eval()
        self.v_net   = copy.deepcopy(agent.v_net).eval()
        for p in self.encoder.parameters(): p.requires_grad_(False)
        for p in self.v_net.parameters():   p.requires_grad_(False)

    @torch.no_grad()
    def update(self, agent: KoopmanGradientPlanner, tau: float = EMA_TAU) -> None:
        for p_t, p_o in zip(self.encoder.parameters(), agent.encoder.parameters()):
            p_t.data.mul_(1 - tau).add_(p_o.data, alpha=tau)
        for p_t, p_o in zip(self.v_net.parameters(), agent.v_net.parameters()):
            p_t.data.mul_(1 - tau).add_(p_o.data, alpha=tau)

    @torch.no_grad()
    def v_target(self, states: torch.Tensor) -> torch.Tensor:
        return self.v_net(self.encoder(states))
