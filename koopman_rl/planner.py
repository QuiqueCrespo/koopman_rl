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
# Implemented as KoopmanGradientPlanner.act_plan_continuous in model.py.
