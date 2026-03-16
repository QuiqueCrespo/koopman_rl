"""
Latent Planner for KoopmanGradientPlanner — discrete and continuous action spaces.

Planners in order of robustness:

  plan_action_shooting              — random shooting MPC (recommended)
  plan_action_beam                  — beam search MPC
  plan_action_gumbel                — Straight-Through Gumbel-Softmax MPC (gradient, no ghost states)
  plan_action_gumbel_cumulative     — Gumbel with discounted cumulative objective
  plan_action_softmax               — gradient-based, convex-hull relaxation (can cheat)
  plan_action_softmax_cumulative    — softmax with discounted cumulative objective
  plan_action_continuous            — continuous-action MPC (tanh-squash)
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from sheaf_rl.config import Config


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
    the learned latent dynamics z_{t+1} = normalize(A z_t + B[:,a_t]),
    scores by V_ψ(z_H), returns the first action of the highest-scoring sequence.
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
        z0     = agent.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        )

        b_all   = agent.B.T                                        # [A, d]
        z_beam  = agent.dyn_step(z0, b_all)                        # [A, d]
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
    cfg: Config = None,
) -> int:
    """
    Straight-Through Gumbel-Softmax MPC.

    Forward pass: F.gumbel_softmax(hard=True) outputs a strict one-hot vector,
    so the Koopman rollout z_{t+h} = normalize(A z + B e_a) visits only real,
    reachable discrete states — no ghost blending.

    Backward pass: the STE gradient treats this as a soft softmax, so the
    value gradient flows back to the logits Θ normally.

    The optimizer cannot cheat: if the goal requires "Up then Right", the
    gradient has to find that exact integer sequence.
    """
    n_acts = cfg.env.n_actions if cfg else agent.n_actions
    device = next(agent.parameters()).device

    with torch.no_grad():
        z_start = agent.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        )

    logits = torch.zeros(horizon, n_acts, device=device, requires_grad=True)
    opt    = optim.Adam([logits], lr=lr)

    for _ in range(plan_iters):
        opt.zero_grad()
        z_t   = z_start
        probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)  # strict one-hot forward
        for t in range(horizon):
            step = (agent.B @ probs[t]).unsqueeze(0)
            z_t  = agent.dyn_step(z_t, step)
        loss = -agent.v_net(z_t).mean()

        (grad_l,) = torch.autograd.grad(loss, logits, only_inputs=True)
        logits.grad = grad_l
        opt.step()

    with torch.no_grad():
        return logits[0].argmax().item()


def plan_action_gumbel_cumulative(
    agent,
    state:      np.ndarray,
    horizon:    int   = 10,
    plan_iters: int   = 20,
    lr:         float = 0.1,
    tau:        float = 1.0,
    gamma:      float = 0.95,
    cfg: Config = None,
) -> int:
    """
    Gumbel-Softmax planner with discounted cumulative value objective:
        max Σ_{t=1}^{H} γ^t V_ψ(z_t)

    Uses hard=True Gumbel-Softmax: same no-ghost-state guarantee as
    plan_action_gumbel but rewards intermediate progress toward goal.
    """
    n_acts = cfg.env.n_actions if cfg else agent.n_actions
    if cfg:
        gamma = cfg.algo.gamma
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
        v_cumul = torch.tensor(0.0, device=device)
        for t in range(horizon):
            step    = (agent.B @ probs[t]).unsqueeze(0)
            z_t     = agent.dyn_step(z_t, step)
            v_cumul = v_cumul + (gamma ** (t + 1)) * agent.v_net(z_t).mean()
        loss = -v_cumul

        (grad_l,) = torch.autograd.grad(loss, logits, only_inputs=True)
        logits.grad = grad_l
        opt.step()

    with torch.no_grad():
        return logits[0].argmax().item()


def plan_action_softmax(
    agent,
    state:      np.ndarray,
    horizon:    int   = 10,
    plan_iters: int   = 20,
    lr:         float = 0.1,
    tau:        float = 1.0,
    cfg: Config = None,
) -> int:
    """
    Differentiable MPC via convex-hull relaxation.
    Parameterises over action logits Θ ∈ R^{H × |A|}.

    Warning: softmax blends B columns into ghost states that don't exist on
    the reachable discrete manifold — the planner can find trajectories
    through infeasible states.  Prefer plan_action_gumbel (hard=True) instead.
    tau controls temperature: lower → sharper / more discrete.
    """
    n_acts = cfg.env.n_actions if cfg else agent.n_actions
    device = next(agent.parameters()).device

    with torch.no_grad():
        z_start = agent.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        )

    logits = torch.zeros(horizon, n_acts, device=device, requires_grad=True)
    opt    = optim.Adam([logits], lr=lr)

    for _ in range(plan_iters):
        opt.zero_grad()
        z_t   = z_start
        probs = F.softmax(logits / tau, dim=-1)
        for t in range(horizon):
            step = (agent.B @ probs[t]).unsqueeze(0)
            z_t  = agent.dyn_step(z_t, step)
        loss = -agent.v_net(z_t).mean()

        (grad_l,) = torch.autograd.grad(loss, logits, only_inputs=True)
        logits.grad = grad_l
        opt.step()

    with torch.no_grad():
        return logits[0].argmax().item()


def plan_action_softmax_cumulative(
    agent,
    state:      np.ndarray,
    horizon:    int   = 10,
    plan_iters: int   = 20,
    lr:         float = 0.1,
    tau:        float = 1.0,
    gamma:      float = 0.95,
    cfg: Config = None,
) -> int:
    """
    Softmax planner with discounted cumulative value objective:
        max Σ_{t=1}^{H} γ^t V_ψ(z_t)

    tau controls temperature: lower → sharper / more discrete.
    """
    n_acts = cfg.env.n_actions if cfg else agent.n_actions
    if cfg:
        gamma = cfg.algo.gamma
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
        probs   = F.softmax(logits / tau, dim=-1)
        v_cumul = torch.tensor(0.0, device=device)
        for t in range(horizon):
            step    = (agent.B @ probs[t]).unsqueeze(0)
            z_t     = agent.dyn_step(z_t, step)
            v_cumul = v_cumul + (gamma ** (t + 1)) * agent.v_net(z_t).mean()
        loss = -v_cumul

        (grad_l,) = torch.autograd.grad(loss, logits, only_inputs=True)
        logits.grad = grad_l
        opt.step()

    with torch.no_grad():
        return logits[0].argmax().item()


def plan_action_continuous(
    agent,
    state:      np.ndarray,
    horizon:    int   = 10,
    plan_iters: int   = 20,
    lr:         float = 0.1,
    cfg: Config = None,
) -> np.ndarray:
    """Differentiable MPC for continuous action spaces. Returns action vector."""
    device     = next(agent.parameters()).device
    action_dim = agent.B.shape[1]

    with torch.no_grad():
        z_start = agent.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        )

    u_logits = torch.randn(horizon, action_dim, device=device) * 1e-4
    u_logits.requires_grad_(True)
    opt = optim.Adam([u_logits], lr=lr)

    for _ in range(plan_iters):
        opt.zero_grad()
        z_t = z_start
        u   = torch.tanh(u_logits)
        for t in range(horizon):
            latent_step = (agent.B @ u[t]).unsqueeze(0)
            z_t         = agent.dyn_step(z_t, latent_step)
        loss = -agent.v_net(z_t).mean()

        (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
        u_logits.grad = grad_u
        opt.step()

    with torch.no_grad():
        return torch.tanh(u_logits[0]).cpu().numpy()
