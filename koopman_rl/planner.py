"""
Latent Planner for KoopmanGradientPlanner — discrete and continuous action spaces.

Discrete planners (gravity basin):
  plan_action_shooting                  — random shooting MPC
  plan_action_beam                      — beam search MPC
  plan_action_gumbel                    — Straight-Through Gumbel-Softmax MPC
                                          cumulative=True for discounted sum objective
  plan_action_toeplitz                  — Block-Toeplitz GEMM planner (requires ortho_a=True)

Continuous planners (pendulum, generic):
  KoopmanGradientPlanner.act_plan_continuous  — Toeplitz GEMM, N states (model.py)

Performance notes:
  - All Toeplitz planners read agent.get_toeplitz_cache() — W_toeplitz and A_stack are
    built once per (horizon, gamma) key and reused until agent.invalidate_toeplitz_cache()
    is called (which the trainer does after every opt.step()).
  - Continuous planners use Adam with manual grad assignment (u_logits.grad = grad; opt.step())
    to get bias-corrected second-moment normalization. Raw GD underperforms Adam for short
    (10-iter) planning because gradient magnitudes through B and A^k are O(1/√d), making
    raw GD steps ~10x smaller than Adam's normalized steps at the same lr.
  - Everything stays float32: A ∈ O(d) has condition number 1, so A^H is numerically
    stable at float32 precision regardless of horizon.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from koopman_rl.config import Config


# ---------------------------------------------------------------------------
# Private Toeplitz system builder (used by agent.get_toeplitz_cache)
# ---------------------------------------------------------------------------

def _build_w_toeplitz(A, horizon: int, gamma: float, device):
    """
    Precompute the state-independent Toeplitz components. All float32.

    W_toeplitz exploits linearity of A ∈ O(d):
      Z = ZIR + W_toeplitz X   (ZIR is state-dependent; computed per-call from A_stack)

    Args:
        A      : [d, d] orthogonal dynamics matrix, detached
        horizon: planning horizon H
        gamma  : discount factor
        device : torch device

    Returns:
        W_toeplitz — [H*d, H*d] causal block-Toeplitz matrix
        gammas     — [H] discount weights γ¹ … γᴴ
        A_stack    — [H+1, d, d] powers A^0 … A^H (reused to compute ZIR per call)
    """
    d = A.shape[0]

    # A^0 … A^H — O(H) d×d matmuls
    A_pows = [torch.eye(d, device=device)]
    for _ in range(horizon):
        A_pows.append(A_pows[-1] @ A)
    A_stack = torch.stack(A_pows)  # [H+1, d, d]

    # W_toeplitz[i,j] = A^{i−j} (i≥j), 0 otherwise
    row_idx   = torch.arange(horizon, device=device).unsqueeze(1)
    col_idx   = torch.arange(horizon, device=device).unsqueeze(0)
    power_idx = (row_idx - col_idx).clamp(min=0)
    causal    = (row_idx >= col_idx).float()
    W_blocks  = A_stack[power_idx] * causal.unsqueeze(-1).unsqueeze(-1)
    W_toeplitz = W_blocks.permute(0, 2, 1, 3).reshape(horizon * d, horizon * d)

    gammas = gamma ** torch.arange(1, horizon + 1, device=device)

    return W_toeplitz, gammas, A_stack


# ---------------------------------------------------------------------------
# Discrete planners (gravity basin)
# ---------------------------------------------------------------------------

def plan_action_shooting(
    agent,
    state:     np.ndarray,
    horizon:   int = 10,
    n_samples: int = 200,
    cfg: Config = None,
) -> int:
    """
    Random-shooting MPC over discrete action sequences.

    Samples n_samples random H-step action sequences, rolls each through
    the learned latent dynamics z_{t+1} = dyn_step(z_t, B[:,a_t]),
    scores by V_ψ(z_H), returns the first action of the best sequence.
    """
    n_actions = cfg.env.n_actions if cfg else agent.n_actions
    device    = next(agent.parameters()).device
    with torch.no_grad():
        z = agent.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        ).expand(n_samples, -1).clone()

        actions = torch.randint(0, n_actions, (n_samples, horizon), device=device)
        for t in range(horizon):
            b_t = agent.B[:, actions[:, t]].T
            z   = agent.dyn_step(z, b_t)

        best = agent.v_net(z).argmax()
        return actions[best, 0].item()


def plan_action_beam(
    agent,
    state:      np.ndarray,
    horizon:    int = 10,
    beam_width: int = 8,
    cfg: Config = None,
) -> int:
    """
    Beam-search MPC over discrete action sequences.
    Keeps top beam_width partial sequences at each step, scored by V_ψ.
    """
    n_actions = cfg.env.n_actions if cfg else agent.n_actions
    device    = next(agent.parameters()).device
    with torch.no_grad():
        z0      = agent.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        )

        b_all   = agent.B.T                                          # [A, d]
        z_beam  = agent.dyn_step(z0, b_all)                          # [A, d]
        v_beam  = agent.v_net(z_beam)
        first_a = torch.arange(n_actions, device=device)

        w       = min(beam_width, n_actions)
        idx     = v_beam.topk(w).indices
        z_beam  = z_beam[idx]
        first_a = first_a[idx]

        for _ in range(1, horizon):
            W      = z_beam.shape[0]
            z_exp  = z_beam.unsqueeze(1).expand(W, n_actions, -1).reshape(W * n_actions, -1)
            b_exp  = b_all.unsqueeze(0).expand(W, n_actions, -1).reshape(W * n_actions, -1)
            z_next = agent.dyn_step(z_exp, b_exp)
            v_next = agent.v_net(z_next)
            fa_exp = first_a.unsqueeze(1).expand(W, n_actions).reshape(W * n_actions)

            w       = min(beam_width, W * n_actions)
            idx     = v_next.topk(w).indices
            z_beam  = z_next[idx]
            first_a = fa_exp[idx]

        return first_a[0].item()


def plan_action_gumbel(
    agent,
    state:      np.ndarray,
    horizon:    int   = 10,
    plan_iters: int   = 20,
    lr:         float = 0.1,
    tau:        float = 1.0,
    gamma:      float = 0.95,
    cumulative: bool  = False,
    cfg: Config = None,
) -> int:
    """
    Straight-Through Gumbel-Softmax MPC for discrete action spaces.

    Forward pass: gumbel_softmax(hard=True) outputs strict one-hot vectors,
    so the rollout visits only real, reachable discrete states — no ghost
    blending. The STE backward flows gradient through logits normally.

    cumulative=False: max V_ψ(z_H)               terminal objective (default)
    cumulative=True:  max Σ_{t=1}^H γ^t V_ψ(z_t) rewards intermediate progress
    """
    n_acts = cfg.env.n_actions if cfg else agent.n_actions
    gamma  = cfg.algo.gamma if cfg else gamma
    device = next(agent.parameters()).device

    with torch.no_grad():
        z_start = agent.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        )

    logits = torch.zeros(horizon, n_acts, device=device, requires_grad=True)
    opt    = optim.Adam([logits], lr=lr)

    for _ in range(plan_iters):
        opt.zero_grad()
        z_t     = z_start
        probs   = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        v_accum = torch.tensor(0.0, device=device)
        for t in range(horizon):
            z_t = agent.dyn_step(z_t, (agent.B @ probs[t]).unsqueeze(0))
            if cumulative:
                v_accum = v_accum + (gamma ** (t + 1)) * agent.v_net(z_t).mean()
        loss = -(v_accum if cumulative else agent.v_net(z_t).mean())

        (grad_l,) = torch.autograd.grad(loss, logits, only_inputs=True)
        logits.grad = grad_l
        opt.step()

    with torch.no_grad():
        return logits[0].argmax().item()


# ---------------------------------------------------------------------------
# Continuous planners (pendulum, generic)
# ---------------------------------------------------------------------------
# Base gradient planner: KoopmanGradientPlanner.act_plan_continuous in model.py.


def plan_cem_gradient_batch(
    agent,
    states:          np.ndarray,
    horizon:         int   = 15,
    cem_iters:       int   = 3,
    n_samples:       int   = 300,
    n_elites:        int   = 30,
    grad_iters:      int   = 5,
    lr:              float = 0.1,
    gamma:           float = 0.95,
    action_scale:    float = 2.0,
    objective:       str   = "value",   # "value" | "value_free"
    z_goal                 = None,      # required when objective == "value_free"
    warm_start_mu          = None,      # [N, H, action_dim] logit-space init; None → zeros
    return_final_mu: bool  = False,     # if True, return (actions, final_mu_tensor)
) -> np.ndarray:
    """
    CEM + gradient hybrid planner for continuous action spaces.

    Phase 1 — CEM (zero-order, torch.no_grad()):
      Samples n_samples action sequences per env via a single batched Toeplitz
      GEMM, scores them, and iteratively refits the Gaussian over the top
      n_elites.  Evaluating N×S trajectories costs one matmul.

      Zero-order search escapes the zero-init local minima that trap gradient
      descent (e.g. energy-pumping in pendulum swing-up).

    Phase 2 — gradient polish (first-order):
      Warm-starts u_logits at the CEM mean, then runs grad_iters Adam steps.

    objective="value"      (value-based):
      Score/loss: Σ_t γᵗ r_net(z_t, u_t) + γᴴ Q(z_H, π(z_H))
      Identical objective to act_plan_continuous — directly comparable.

    objective="value_free" (value-free / latent-goal):
      Score/loss: −‖z_H − z_goal‖²   (no r_net or Q needed)
      z_goal must be provided as a [1, d] or [N, d] encoded tensor.

    warm_start_mu:
      Logit-space initial mean for the CEM distribution (replaces zeros).
      Typically the final u_logits from the previous timestep, shifted forward
      one step: [u_2, ..., u_H, 0].  See CEMPlannerWarmStart.
      When provided, initial sigma is tightened to 0.5 (vs 2.0 cold-start)
      so the prior actually constrains sampling rather than being drowned out.

    return_final_mu:
      When True returns (actions [N, action_dim], final_mu [N, H, action_dim])
      so the caller can carry the solution forward.  Default False preserves
      the original return type.
    """
    assert objective in ("value", "value_free", "reward"), \
        f"objective must be 'value', 'value_free', or 'reward', got {objective!r}"
    if objective == "value_free":
        assert z_goal is not None, "z_goal required for objective='value_free'"

    device     = next(agent.parameters()).device
    d          = agent.d
    action_dim = agent.B.shape[1]
    N          = len(states)

    with torch.no_grad():
        z0 = agent.encoder(
            torch.tensor(states, dtype=torch.float32, device=device)
        )  # [N, d]

        B = agent.B.detach()
        W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)
        ZIR = torch.einsum('kij,nj->nki', A_stack[1:], z0)  # [N, H, d]

        ZIR_s       = ZIR.unsqueeze(1).expand(N, n_samples, horizon, d)
        gammas_path = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)

        # ── Phase 1: CEM ──────────────────────────────────────────────────────
        # Cold start: mu=0, sigma=2.0 (tanh(±2)≈±0.96 — full action range).
        # Warm start: mu from previous solution, sigma=0.5 so the prior
        # actually constrains sampling.  sigma=2.0 with a non-zero mean is
        # mathematically equivalent to cold-start: the samples span the full
        # tanh range regardless of mu, so the warm-start mean would be ignored.
        if warm_start_mu is not None:
            mu    = warm_start_mu.to(device).clone()
            sigma = torch.ones(N, horizon, action_dim, device=device) * 1.0
        else:
            # Cold start: bias mean toward committed action (logit=1 → tanh≈0.76).
            # Zero mean → "do nothing" plan → worst saddle from hanging position.
            mu    = torch.zeros(N, horizon, action_dim, device=device)
            sigma = torch.ones(N, horizon, action_dim, device=device) * 2.0

        for _ in range(cem_iters):
            u_logits_s = (mu.unsqueeze(1)
                          + sigma.unsqueeze(1)
                          * torch.randn(N, n_samples, horizon, action_dim, device=device))
            u_s = torch.tanh(u_logits_s)

            NS     = N * n_samples
            X_flat = (u_s @ B.T).reshape(NS, horizon * d)
            Z_path = (ZIR_s.reshape(NS, horizon, d)
                      + (X_flat @ W_toeplitz.T).reshape(NS, horizon, d)
                      ).reshape(N, n_samples, horizon, d)

            if objective in ("value", "reward"):
                z0_s  = z0.unsqueeze(1).unsqueeze(2).expand(N, n_samples, 1, d)
                Z_curr = torch.cat([z0_s, Z_path[:, :, :-1, :]], dim=2)
                ZU    = torch.cat([Z_curr, u_s], dim=-1)
                disc_path = (gammas_path * agent.r_net(ZU).squeeze(-1)).sum(dim=2)
                if objective == "value":
                    z_H   = Z_path[:, :, -1, :]
                    a_H   = agent.pi_net(z_H)
                    q_H   = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
                    scores = disc_path + gamma ** horizon * q_H          # [N, S]
                else:  # reward — no terminal Q
                    scores = disc_path                                    # [N, S]
            else:  # value_free — score = −‖z_H − z_goal‖²
                z_H    = Z_path[:, :, -1, :]                             # [N, S, d]
                scores = -((z_H - z_goal.unsqueeze(1)) ** 2).sum(dim=-1) # [N, S]

            elite_idx    = scores.topk(n_elites, dim=1).indices
            batch_idx    = torch.arange(N, device=device).unsqueeze(1).expand(N, n_elites)
            elite_logits = u_logits_s[batch_idx, elite_idx]
            mu    = elite_logits.mean(dim=1)
            sigma = elite_logits.std(dim=1).clamp(min=1e-3)

    # ── Phase 2: gradient polish warm-started at CEM mean ─────────────────────
    u_logits = mu.clone().requires_grad_(True)
    opt = optim.Adam([u_logits], lr=lr)

    for _ in range(grad_iters):
        opt.zero_grad()
        u      = torch.tanh(u_logits)
        X_flat = (u @ B.T).reshape(N, horizon * d)
        Z_path = ZIR + (X_flat @ W_toeplitz.T).reshape(N, horizon, d)

        if objective in ("value", "reward"):
            Z_curr    = torch.cat([z0.unsqueeze(1), Z_path[:, :-1, :]], dim=1)
            ZU        = torch.cat([Z_curr, u], dim=-1)
            disc_path = (gammas_path.unsqueeze(0) * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
            if objective == "value":
                z_H  = Z_path[:, -1, :]
                a_H  = agent.pi_net(z_H)
                q_H  = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
                loss = -(disc_path + gamma ** horizon * q_H).mean()
            else:  # reward — no terminal Q
                loss = -disc_path.mean()
        else:  # value_free
            z_H  = Z_path[:, -1, :]
            loss = ((z_H - z_goal.detach()) ** 2).sum(dim=-1).mean()

        (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
        u_logits.grad = grad_u
        opt.step()

    with torch.no_grad():
        actions = (torch.tanh(u_logits[:, 0, :]) * action_scale).cpu().numpy()
    if return_final_mu:
        return actions, u_logits.detach()
    return actions


# ---------------------------------------------------------------------------
# Stateful receding-horizon CEM wrapper
# ---------------------------------------------------------------------------

class CEMPlannerWarmStart:
    """
    Stateful CEM planner with receding-horizon warm-start.

    After each call the committed action u_1 is discarded and the remaining
    solution [u_2, ..., u_H] is shifted forward by one step, zero-padded to
    length H, and used to initialise the *mean* of the next CEM distribution.
    This replaces cold zero-initialisation with a temporally consistent prior,
    producing smoother action sequences and faster CEM convergence.

    Call reset() at every episode boundary so stale solutions from the
    previous episode do not pollute the new one.

    Usage
    -----
        planner = CEMPlannerWarmStart(agent, **kwargs)
        planner.reset()                       # at episode start
        action = planner(state[np.newaxis])   # [1, action_dim] → [action_dim]
    """

    def __init__(
        self,
        agent,
        horizon:      int,
        cem_iters:    int,
        n_samples:    int,
        n_elites:     int,
        grad_iters:   int,
        lr:           float,
        gamma:        float,
        action_scale: float,
        objective:    str   = "value",
        z_goal              = None,
    ):
        self._agent       = agent
        self._action_dim  = agent.B.shape[1]
        self._horizon     = horizon
        self._device      = next(agent.parameters()).device
        self._kwargs      = dict(
            horizon=horizon, cem_iters=cem_iters, n_samples=n_samples,
            n_elites=n_elites, grad_iters=grad_iters, lr=lr, gamma=gamma,
            action_scale=action_scale, objective=objective, z_goal=z_goal,
        )
        self._mu: torch.Tensor | None = None  # [N, H, action_dim], logit space

    def __call__(self, states: np.ndarray) -> np.ndarray:
        # Track whether caller passed a single unbatched state so we can match
        # the return shape — prevents Gym obs corruption from (1,1) actions.
        unbatched = states.ndim == 1
        states = np.atleast_2d(states)
        actions, new_mu = plan_cem_gradient_batch(
            self._agent, states,
            warm_start_mu=self._mu,
            return_final_mu=True,
            **self._kwargs,
        )
        # Shift: drop committed step, slide remainder forward, zero-pad last step.
        N = len(states)
        self._mu = torch.cat([
            new_mu[:, 1:, :],
            torch.zeros(N, 1, self._action_dim, device=self._device),
        ], dim=1)
        return actions[0] if unbatched else actions

    def reset(self) -> None:
        """Discard warm-start state. Call at the start of every episode."""
        self._mu = None


# ---------------------------------------------------------------------------
# Stateful receding-horizon wrapper for the Toeplitz gradient planner
# ---------------------------------------------------------------------------

class ToeplitzPlannerWarmStart:
    """
    Receding-horizon warm-start for act_plan_continuous.

    After each call the committed action u_1 is discarded and the remaining
    solution [u_2, ..., u_H] is shifted forward, zero-padded to length H,
    and used as the initial u_logits for the next planning call.  This
    replaces cold zero-initialisation: the gradient planner starts from a
    temporally consistent prior instead of from "do nothing", improving both
    convergence speed and solution quality.

    Call reset() at every episode boundary.
    """

    def __init__(self, agent, horizon, plan_iters, lr, gamma, action_scale,
                 objective="value"):
        self._agent       = agent
        self._action_dim  = agent.B.shape[1]
        self._horizon     = horizon
        self._device      = next(agent.parameters()).device
        self._kwargs      = dict(
            horizon=horizon, plan_iters=plan_iters, lr=lr, gamma=gamma,
            action_scale=action_scale, objective=objective,
        )
        self._u: torch.Tensor | None = None  # [N, H, action_dim], logit space

    def __call__(self, states: np.ndarray) -> np.ndarray:
        unbatched = states.ndim == 1
        states    = np.atleast_2d(states)
        actions, new_u = self._agent.act_plan_continuous(
            states,
            warm_start_u=self._u,
            return_final_u=True,
            **self._kwargs,
        )
        N = len(states)
        # Shift: drop committed step, slide forward, zero-pad last.
        self._u = torch.cat([
            new_u[:, 1:, :],
            torch.zeros(N, 1, self._action_dim, device=self._device),
        ], dim=1)
        return actions[0] if unbatched else actions

    def reset(self) -> None:
        self._u = None
