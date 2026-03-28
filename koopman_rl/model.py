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
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim


from koopman_rl.config import Config
from koopman_rl.planner import (
    _build_w_toeplitz,
    plan_action_gumbel
)

def _oscillator_init(d: int) -> torch.Tensor:
    """Skew-symmetric matrix with purely imaginary eigenvalues ±iβ_k, β_k ~ U[0.5, 1.5].
    J = block_diag([[0,-β₁],[β₁,0]], ...) rotated by a random orthogonal Q: A = Q J Qᵀ.
    Eigenvalues are purely imaginary → undamped oscillation inductive bias.
    Each mode gets a distinct frequency; random Q mixes modes across all latent dims."""
    if d % 2 != 0:
        print("Warning: d is odd. One eigenvalue will be 0.")
    J = torch.zeros(d, d)
    betas = torch.rand(d // 2) + 0.5   # β_k ~ U[0.5, 1.5], no zeros
    for i in range(d // 2):
        idx = i * 2
        b = betas[i].item()
        J[idx,   idx+1] = -b
        J[idx+1, idx  ] =  b
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    return Q @ J @ Q.T


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
                 tanh_out: bool = False, device=None, continuous: bool = False):
        super().__init__()
        self.d          = d
        self.n_actions  = n_actions
        self._ortho_a   = ortho_a
        self._continuous = continuous
        self.encoder    = Encoder(state_dim, d, tanh_out=tanh_out)
        self.decoder    = nn.Linear(d, state_dim)

        if continuous:
            # Continuous actor-critic heads.
            # r_net: R(z, a_norm) → scalar  (direct regression, no bootstrap)
            # q_net: Q(z, a_norm) → scalar  (Bellman TD with target network)
            # pi_net: π(z) → a_norm ∈ [-1,1]^action_dim  (DDPG-style)
            self.r_net  = nn.Sequential(
                nn.Linear(d + n_actions, 64), nn.ReLU(), nn.Linear(64, 1))
            self.q_net  = nn.Sequential(
                nn.Linear(d + n_actions, 64), nn.ReLU(), nn.Linear(64, 1))
            self.pi_net = nn.Sequential(
                nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())
        else:
            self.v_net = ValueNetwork(d)

        # CUDA: SVD Procrustes hard constraint (covers full O(d), det=±1).
        # MPS/CPU: unconstrained A + soft penalty in loss (no linalg.svd round-trip).
        dev_type = torch.device(device).type if device is not None else "cpu"
        self._use_hard_ortho = ortho_a and dev_type == "cuda"

        if self._use_hard_ortho:
            # Oscillator init + small noise: noise breaks σᵢ = σⱼ degeneracy required
            # for stable SVD backward (pure orthogonal init → all σ=1 → NaN gradient).
            self._A_layer = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                osc = _oscillator_init(d)
                self._A_layer.weight.copy_(osc + 0.01 * torch.randn(d, d))
            parametrize.register_parametrization(self._A_layer, 'weight', _SVDOrthogonal())
        else:
            # Unconstrained; soft penalty applied in loss when ortho_a=True.
            self.A = nn.Parameter(_oscillator_init(d))

        B = torch.empty(d, n_actions)
        nn.init.orthogonal_(B)
        self.B = nn.Parameter(B)

        # Toeplitz cache: keyed by (horizon, gamma) → (W_toeplitz, gammas, A_stack)
        # Valid until invalidate_toeplitz_cache() is called (after each opt.step()).
        self._toeplitz_cache: dict = {}

    def get_toeplitz_cache(self, horizon: int, gamma: float) -> tuple:
        """Return cached (W_toeplitz, gammas, A_stack), building if needed."""
        key = (horizon, gamma)
        if key not in self._toeplitz_cache:
            device = next(self.parameters()).device
            # Use A_eff = 2I - A to match the residual dyn_step formulation.
            A_eff = 2 * torch.eye(self.d, device=device) - self.A.detach()
            self._toeplitz_cache[key] = _build_w_toeplitz(
                A_eff, horizon, gamma, device)
        return self._toeplitz_cache[key]

    def invalidate_toeplitz_cache(self) -> None:
        """Call after opt.step() — A may have changed."""
        self._toeplitz_cache.clear()

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
        One Koopman step: z' = z + (I - A)z + b_vec  (residual form of Az + b_vec).

        Equivalent to z' = (2I - A)z + b_vec.  The residual formulation is
        friendlier to gradient-based planning: the skip connection z passes
        gradients directly back through the trajectory without going through A.
        """
        return z + (z - z @ self.A.T) + b_vec

    def koop_parameters(self) -> list:
        """Parameter list for the Koopman optimizer group (A and B).
        With hard ortho, returns the Cayley pre-image parameters (not the read-only O(d) weight)."""
        A_params = list(self._A_layer.parameters()) if self._use_hard_ortho else [self.A]
        return A_params + [self.B]

    @classmethod
    def from_cfg(cls, cfg: Config, device=None) -> "KoopmanGradientPlanner":
        agent = cls(
            state_dim=cfg.env.state_dim,
            d=cfg.model.d,
            n_actions=cfg.env.n_actions,
            ortho_a=cfg.model.ortho_a,
            tanh_out=cfg.model.tanh_out,
            device=device,
            continuous=cfg.env.continuous,
        )
        if cfg.env.obs_type == "pixels":
            from koopman_rl.visual_encoder import VisualEncoder
            agent.encoder = VisualEncoder(cfg.env.img_channels, cfg.env.img_size, cfg.model.d)
        return agent

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.v_net(self.encoder(state))

    @torch.no_grad()
    def act_policy_continuous_batch(self, states: np.ndarray,
                                    action_scale: float = 1.0) -> np.ndarray:
        """Direct policy rollout for N states. Returns [N, action_dim] in [-action_scale, action_scale]."""
        device = next(self.parameters()).device
        z = self.encoder(torch.tensor(states, dtype=torch.float32, device=device))
        return (self.pi_net(z) * action_scale).cpu().numpy()

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
        return plan_action_gumbel(self, state, horizon, plan_iters, tau=tau)

    def act_plan_continuous(
        agent,
        states:          np.ndarray,
        horizon:         int   = 10,
        plan_iters:      int   = 20,
        lr:              float = 0.1,
        gamma:           float = 0.95,
        action_scale:    float = 1.0,
        objective:       str   = "value",   # "value" | "reward"
        warm_start_u     = None,            # [N, H, action_dim] logit-space init
        return_final_u:  bool  = False,     # if True return (actions, final_u_logits)
    ) -> np.ndarray:
        """
        Batched Block-Toeplitz GEMM planner for continuous action spaces.

        Objective: Σ_{t=0}^{H-1} γ^t r_net(z_t, u_t)  +  γ^H Q(z_H, π(z_H))
        - Path costs come from r_net (no bootstrap, stable gradient).
        - Terminal uses Q + π to estimate future value beyond the horizon.

        W_toeplitz and A_stack read from agent.get_toeplitz_cache() — built once
        per (horizon, gamma) key; each plan_iter is a single batched GEMM.

        Returns actions [N, action_dim] ∈ [-action_scale, action_scale].
        Requires ortho_a=True (linear superposition Z = ZIR + W X must hold).
        """
        device     = next(agent.parameters()).device
        d          = agent.d
        action_dim = agent.B.shape[1]
        N          = len(states)

        with torch.no_grad():
            z0 = agent.encoder(
                torch.tensor(states, dtype=torch.float32, device=device)
            )                                                          # [N, d]

            B = agent.B.detach()                                       # [d, action_dim]

            # W_toeplitz and A_stack come from cache — free if A hasn't changed
            W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)
            ZIR = torch.einsum('kij,nj->nki', A_stack[1:], z0)        # [N, H, d]

        # γ^0 … γ^{H-1} for path discounting
        gammas_path = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)

        if warm_start_u is not None:
            u_logits = warm_start_u.to(device).clone().requires_grad_(True)
        else:
            u_logits = torch.zeros(N, horizon, action_dim, device=device, requires_grad=True)
        opt      = optim.Adam([u_logits], lr=lr)

        for _ in range(plan_iters):
            opt.zero_grad()
            u      = torch.tanh(u_logits)                              # [N, H, action_dim]
            X_flat = (u @ B.T).reshape(N, horizon * d)                 # [N, H*d]

            # Single batched GEMM replaces N × H sequential dyn_step calls
            Z = ZIR + (X_flat @ W_toeplitz.T).reshape(N, horizon, d)  # [N, H, d]

            # Explicit path rewards from r_net — pair current states z_0..z_{H-1} with actions
            Z_curr       = torch.cat([z0.unsqueeze(1), Z[:, :-1, :]], dim=1)  # [N, H, d]
            ZU           = torch.cat([Z_curr, u], dim=-1)              # [N, H, d+action_dim]
            path_rewards = agent.r_net(ZU).squeeze(-1)                 # [N, H]
            disc_path    = (gammas_path.unsqueeze(0) * path_rewards).sum(dim=1)  # [N]

            if objective == "value":
                # Terminal Q: Q(z_H, π(z_H)) — gradient flows through z_H and π
                z_H  = Z[:, -1, :]                                     # [N, d]
                a_H  = agent.pi_net(z_H)                               # [N, action_dim]
                q_H  = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
                loss = -(disc_path + gamma ** horizon * q_H).mean()
            else:  # reward — r_net path only, no terminal Q
                loss = -disc_path.mean()

            (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
            u_logits.grad = grad_u
            opt.step()

        with torch.no_grad():
            actions = (torch.tanh(u_logits[:, 0, :]) * action_scale).cpu().numpy()
        if return_final_u:
            return actions, u_logits.detach()
        return actions






class TargetNetwork:
    """
    EMA copies of encoder + value/critic heads.  Not an nn.Module — excluded
    from optimizer parameters automatically.  A and B are NOT tracked.

    Discrete path: copies v_net.
    Continuous path: copies q_net and pi_net (r_net is direct regression — no target needed).
    """
    def __init__(self, agent: KoopmanGradientPlanner):
        self.encoder = copy.deepcopy(agent.encoder).eval()
        for name in ['v_net', 'q_net', 'pi_net']:
            if hasattr(agent, name):
                setattr(self, name, copy.deepcopy(getattr(agent, name)).eval())
        # Freeze all target params
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for name in ['v_net', 'q_net', 'pi_net']:
            if hasattr(self, name):
                for p in getattr(self, name).parameters():
                    p.requires_grad_(False)

    @torch.no_grad()
    def update(self, agent: KoopmanGradientPlanner, tau: float = EMA_TAU) -> None:
        for p_t, p_o in zip(self.encoder.parameters(), agent.encoder.parameters()):
            p_t.data.mul_(1 - tau).add_(p_o.data, alpha=tau)
        for name in ['v_net', 'q_net', 'pi_net']:
            if hasattr(self, name):
                for p_t, p_o in zip(getattr(self, name).parameters(),
                                    getattr(agent, name).parameters()):
                    p_t.data.mul_(1 - tau).add_(p_o.data, alpha=tau)

    @torch.no_grad()
    def v_target(self, states: torch.Tensor) -> torch.Tensor:
        return self.v_net(self.encoder(states))
