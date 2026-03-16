"""
Neural components: Encoder, ValueNetwork, QNetwork, KoopmanGradientPlanner, TargetNetwork.

NOTE — ortho_a on GPU vs MPS:
  Currently ortho_a=True uses a soft penalty ||AᵀA − I||²_F added to the loss
  (see ortho_penalty() and LAMBDA_ORTHO in algorithms.py).  This is a workaround
  for MPS: torch.linalg.svd and torch.linalg.qr both fall back to CPU on MPS,
  causing a device round-trip on every forward+backward pass.

  On a CUDA GPU, use PyTorch's native orthogonal parametrization instead.
  DO NOT use SVD on the forward pass — even on CUDA, svd is a sequential iterative
  algorithm that does not parallelise well across CUDA cores and will silently
  bottleneck throughput for any d ≥ 64.

  The correct CUDA approach uses the Cayley map  A = (I − S)(I + S)⁻¹  where S is
  skew-symmetric, or the matrix exponential.  Both are parallelisable matmuls and
  PyTorch exposes them via the native orthogonal parametrization:

    import torch.nn.utils.parametrizations as parametrizations  # note: not parametrize

    # in KoopmanGradientPlanner.__init__, replace self.A = nn.Parameter(torch.eye(d)) with:
    self.A_layer = nn.Linear(d, d, bias=False)
    parametrizations.orthogonal(self.A_layer, 'weight')   # Cayley/matrix-exp, NOT SVD
    # access as self.A_layer.weight; add a property 'A' for drop-in compatibility.

  This guarantees ||A z|| = ||z|| exactly at every step with near-zero forward-pass
  overhead.  The soft penalty (current default) only approximately enforces this and
  can drift under large learning rates.

  BONUS — parallel horizon unroll (CUDA only):
  With A strictly orthonormal you can pre-compute the matrix powers A¹…Aᴴ and
  evaluate the entire H-step lookahead as a single batched matmul (equivalent to a
  1-D causal convolution in latent space) rather than a Python for-loop.  This makes
  plan_action_* O(1) wall-clock in H instead of O(H) and is the right path for
  longer horizons / real-time inference.
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations

from sheaf_rl.config import Config, ModelConfig, EnvConfig

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
        out = self.net(x)
        if self.tanh_out or getattr(self, "no_normalize", False):
            return out
        return F.normalize(out, dim=-1)


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


class KoopmanGradientPlanner(nn.Module):
    """
    Formerly SheafAgent. Renamed to reflect primary usage: MPC via learned
    Koopman latent dynamics (A, B) + value network V_ψ.

    Latent transition model:
        ortho_a=False (default): z_{t+1} = normalize(A z_t + B a_t)
        ortho_a=True  (linear):  z_{t+1} = A z_t + B a_t   (A ≈ orthonormal via
                                            soft penalty ||AᵀA − I||²_F in loss)

    A ∈ R^{d×d}   — shared state dynamics (free parameter, penalised toward O(d))
    B ∈ R^{d×|A|} — action input matrix (free, cols orthonormal at init)
    a_t           — one-hot for discrete actions, or continuous vector for cont. envs

    Use dyn_step(z, b_vec) for all transition computations — it dispatches the
    normalize-or-not decision centrally so callers stay mode-agnostic.
    """
    def __init__(self, state_dim: int = STATE_DIM, d: int = D,
                 n_actions: int = N_ACTIONS, ortho_a: bool = False,
                 tanh_out: bool = False):
        super().__init__()
        self.d         = d
        self.n_actions = n_actions
        self._ortho_a  = ortho_a
        self.encoder   = Encoder(state_dim, d, tanh_out=tanh_out)
        self.v_net     = ValueNetwork(d)
        self.decoder   = nn.Linear(d, state_dim)     # reconstruction anchor (anti-collapse)

        if ortho_a:
            # Hard orthogonal constraint via Cayley map / matrix exponential.
            # parametrizations.orthogonal keeps A ∈ O(d) exactly after every step.
            # Uses matrix_exp (Taylor-series matmuls) — MPS/CUDA-native, no SVD.
            # Optimizer sees the unconstrained pre-image; .weight gives the O(d) matrix.
            self._A_layer = nn.Linear(d, d, bias=False)
            nn.init.eye_(self._A_layer.weight)       # start as identity
            parametrizations.orthogonal(self._A_layer, 'weight', orthogonal_map="matrix_exp")
        else:
            # Default: unconstrained A; normalisation applied in dyn_step instead.
            self.A = nn.Parameter(torch.eye(d))      # [d, d]

        # B columns orthonormal at init (both modes)
        B = torch.empty(d, n_actions)
        nn.init.orthogonal_(B)
        self.B = nn.Parameter(B)                     # [d, n_actions]

    def __getattr__(self, name: str):
        # When ortho_a=True, 'A' is not a Parameter — intercept and return
        # the parametrized weight (the actual O(d) matrix).
        if name == 'A' and self.__dict__.get('_ortho_a', False):
            return self._modules['_A_layer'].weight
        return super().__getattr__(name)

    def dyn_step(self, z: torch.Tensor, b_vec: torch.Tensor) -> torch.Tensor:
        """
        One Koopman step: z' = A z + b_vec  [+ optional normalisation].

        ortho_a=False: normalises z' to S^{d-1} (original sheaf approach).
        ortho_a=True:  fully linear — no normalisation. A is orthonormal so
                       ||A z|| = ||z||; only B a perturbs the norm.

        All callers (planner, train loop, act) go through here so switching
        modes requires changing only the config, not any call site.
        """
        raw = z @ self.A.T + b_vec
        return raw if self._ortho_a else F.normalize(raw, dim=-1)

    def koop_parameters(self) -> list:
        """Parameter list for the Koopman optimizer group (A and B).
        With ortho_a=True, returns the underlying pre-image parameters that the
        optimizer updates — not the projected weight (which is read-only)."""
        A_params = list(self._A_layer.parameters()) if self._ortho_a else [self.A]
        return A_params + [self.B]

    @classmethod
    def from_cfg(cls, cfg: Config) -> "KoopmanGradientPlanner":
        return cls(
            state_dim=cfg.env.state_dim,
            d=cfg.model.d,
            n_actions=cfg.env.n_actions,
            ortho_a=cfg.model.ortho_a,
            tanh_out=cfg.model.tanh_out,
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
        from sheaf_rl.planner import plan_action_gumbel
        return plan_action_gumbel(self, state, horizon, plan_iters, tau=tau)

    def act_plan_continuous(self, state: np.ndarray,
                            horizon: int = 10, plan_iters: int = 20,
                            action_scale: float = 1.0) -> np.ndarray:
        """tanh-squash MPC — continuous environments. Returns action ∈ [-scale, scale]^d."""
        from sheaf_rl.planner import plan_action_continuous
        return plan_action_continuous(self, state, horizon, plan_iters) * action_scale


# Backward-compatibility alias for any existing saved checkpoints or notebooks
SheafAgent = KoopmanGradientPlanner


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
