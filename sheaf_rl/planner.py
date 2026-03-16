"""
Latent Planner for SheafAgent — discrete action spaces.

Three planners, in order of robustness:

  plan_action_shooting  — random shooting MPC (recommended)
  plan_action_beam      — beam search MPC
  plan_action_softmax / plan_action_softmax_cumulative — gradient-based
  plan_action_continuous — continuous-action MPC (reference)
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
            z   = F.normalize(z @ agent.A.T + b_t, dim=-1)

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
        z_beam  = F.normalize(z0 @ agent.A.T + b_all, dim=-1)     # [A, d]
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
            z_next = F.normalize(z_exp @ agent.A.T + b_exp, dim=-1)
            v_next = agent.v_net(z_next)
            fa_exp = first_a.unsqueeze(1).expand(W, n_actions).reshape(W * n_actions)

            w       = min(beam_width, W * n_actions)
            idx     = v_next.topk(w).indices
            z_beam  = z_next[idx]
            first_a = fa_exp[idx]

        return first_a[0].item()


def plan_action_softmax(
    agent,
    state:      np.ndarray,
    horizon:    int   = 10,
    plan_iters: int   = 20,
    lr:         float = 0.1,
    cfg: Config = None,
) -> int:
    """
    Differentiable MPC via convex-hull relaxation.
    Parameterises over action logits Θ ∈ R^{H × |A|}.
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
        probs = F.softmax(logits, dim=-1)
        for t in range(horizon):
            step = (agent.B @ probs[t]).unsqueeze(0)
            z_t  = F.normalize(z_t @ agent.A.T + step, dim=-1)
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
    gamma:      float = 0.95,
    cfg: Config = None,
) -> int:
    """
    Softmax planner with discounted cumulative value objective:
        max Σ_{t=1}^{H} γ^t V_ψ(z_t)
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
        probs   = F.softmax(logits, dim=-1)
        v_cumul = torch.tensor(0.0, device=device)
        for t in range(horizon):
            step    = (agent.B @ probs[t]).unsqueeze(0)
            z_t     = F.normalize(z_t @ agent.A.T + step, dim=-1)
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
            z_t         = F.normalize(z_t @ agent.A.T + latent_step, dim=-1)
        loss = -agent.v_net(z_t).mean()

        (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
        u_logits.grad = grad_u
        opt.step()

    with torch.no_grad():
        return torch.tanh(u_logits[0]).cpu().numpy()
