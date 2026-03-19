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

from koopman_rl.config import Config


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


def plan_action_toeplitz(
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
    Block-Toeplitz GEMM planner — O(H²d²) precompute, single GEMM per grad step.

    Exploits exact orthogonality (ortho_a=True) to unroll the entire H-step
    latent trajectory as a closed-form linear system:

        Z = ZIR + W_toeplitz X

    where:
      ZIR[k]           = A^{k+1} z₀        (free / zero-input response)
      W_toeplitz[i,j]  = A^{i-j}  (i≥j)   (lower-triangular Block-Toeplitz)
      X[t]             = B aₜ               (latent action vectors, [H, d])

    Both ZIR and W_toeplitz are pre-computed once outside the Adam loop with
    torch.no_grad(). Each plan_iter reduces to a single dense GEMM
    (W_toeplitz @ X_flat) plus one V-net forward pass — no sequential rollout.

    The backward pass tracks gradient only through the GEMM input X (which
    depends on logits via Gumbel-Softmax), not through H chained dyn_step calls.
    Memory footprint of the autograd graph is O(H d) instead of O(H² d).

    Requires ortho_a=True; for normalised (spherical) dynamics the linear
    superposition principle does not hold.
    """
    if cfg:
        gamma = cfg.algo.gamma
    n_acts = cfg.env.n_actions if cfg else agent.n_actions
    device = next(agent.parameters()).device
    d      = agent.d

    with torch.no_grad():
        z0 = agent.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        ).squeeze(0)  # [d]

        A = agent.A.detach()  # [d, d], exact O(d) matrix via SVD parametrisation

        # Build A^0 … A^H iteratively — O(H) matmuls of size d×d
        A_pows = [torch.eye(d, device=device)]
        for _ in range(horizon):
            A_pows.append(A_pows[-1] @ A)
        A_stack = torch.stack(A_pows)  # [H+1, d, d]

        # Zero-Input Response: ZIR[k] = A^{k+1} z₀  (≡ z₀ @ (Aᵀ)^{k+1} for 1-D)
        ZIR = torch.einsum('kij,j->ki', A_stack[1:], z0)  # [H, d]

        # Causal indices for Block-Toeplitz
        row_idx   = torch.arange(horizon, device=device).unsqueeze(1)   # [H, 1]
        col_idx   = torch.arange(horizon, device=device).unsqueeze(0)   # [1, H]
        power_idx = (row_idx - col_idx).clamp(min=0)                    # [H, H]
        causal    = (row_idx >= col_idx).float()                         # lower-tri mask

        # Gather blocks → [H, H, d, d], zero upper triangle
        W_blocks = A_stack[power_idx] * causal.unsqueeze(-1).unsqueeze(-1)

        # Reshape to [H·d, H·d]: one dense GEMM replaces H sequential matmuls
        W_toeplitz = W_blocks.permute(0, 2, 1, 3).reshape(horizon * d, horizon * d)

        # Discount weights γ¹ … γᴴ
        gammas = (gamma ** torch.arange(1, horizon + 1, device=device)).unsqueeze(1)

    B = agent.B.detach()  # [d, n_acts]

    logits = torch.zeros(horizon, n_acts, device=device, requires_grad=True)
    opt    = optim.Adam([logits], lr=lr)

    for _ in range(plan_iters):
        opt.zero_grad()

        probs  = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)   # [H, n_acts]
        X      = probs @ B.T                                             # [H, d]
        X_flat = X.reshape(horizon * d, 1)

        # Single cuBLAS GEMM — entire horizon parallelised on GPU
        ZSR  = (W_toeplitz @ X_flat).reshape(horizon, d)
        Z    = ZIR + ZSR                                                 # [H, d]
        loss = -(gammas * agent.v_net(Z)).sum()

        (grad_l,) = torch.autograd.grad(loss, logits, only_inputs=True)
        logits.grad = grad_l
        opt.step()

    with torch.no_grad():
        return logits[0].argmax().item()


class WarmStartToeplitzPlanner:
    """
    Stateful wrapper around plan_action_toeplitz_continuous that warm-starts
    each planning call by shifting the previous solution one step forward.

    On each call:
      u_logits[0..H-2] ← previous u_logits[1..H-1]  (already-committed actions)
      u_logits[H-1]    ← randn * 1e-4                (fresh guess for new horizon tail)
    Adam first/second moments are shifted identically so curvature information
    from previous steps carries over for the actions that remain in the plan.

    Call reset() at episode boundaries (or after any discontinuity).
    """
    def __init__(self, agent, horizon: int = 10, plan_iters: int = 20,
                 lr: float = 0.1, gamma: float = 0.95,
                 action_scale: float = 1.0, cumulative: bool = False):
        self.agent        = agent
        self.horizon      = horizon
        self.plan_iters   = plan_iters
        self.lr           = lr
        self.gamma        = gamma
        self.action_scale = action_scale
        self.cumulative   = cumulative
        self._u_logits    = None   # [H, action_dim] float64
        self._opt         = None

    def reset(self):
        self._u_logits = None
        self._opt      = None

    def __call__(self, state: np.ndarray) -> np.ndarray:
        agent      = self.agent
        device     = next(agent.parameters()).device
        action_dim = agent.B.shape[1]
        H          = self.horizon

        # Shift previous solution or cold-start
        if self._u_logits is not None:
            tail = torch.randn(1, action_dim, device=device,
                               dtype=torch.float64) * 1e-4
            u_init = torch.cat([self._u_logits[1:].detach(), tail], dim=0)
        else:
            u_init = torch.randn(H, action_dim, device=device,
                                 dtype=torch.float64) * 1e-4

        u_logits = u_init.clone().requires_grad_(True)

        if self._opt is not None:
            # Rebuild Adam, transplant shifted moment estimates
            opt = optim.Adam([u_logits], lr=self.lr)
            prev_state = self._opt.state[self._opt.param_groups[0]['params'][0]]
            if prev_state:
                new_state   = opt.state[u_logits]
                new_state['step']        = prev_state['step']
                new_state['exp_avg']     = torch.cat(
                    [prev_state['exp_avg'][1:], torch.zeros(1, action_dim, device=device, dtype=torch.float64)], dim=0)
                new_state['exp_avg_sq']  = torch.cat(
                    [prev_state['exp_avg_sq'][1:], torch.zeros(1, action_dim, device=device, dtype=torch.float64)], dim=0)
        else:
            opt = optim.Adam([u_logits], lr=self.lr)

        # Run the optimization (same body as plan_action_toeplitz_continuous)
        d = agent.d
        with torch.no_grad():
            z0 = agent.encoder(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            ).squeeze(0).to(torch.float64)
            A = agent.A.detach().to(torch.float64)
            B = agent.B.detach().to(torch.float64)

            A_pows = [torch.eye(d, device=device, dtype=torch.float64)]
            for _ in range(H):
                A_pows.append(A_pows[-1] @ A)
            A_stack = torch.stack(A_pows)

            ZIR = torch.einsum('kij,j->ki', A_stack[1:], z0)

            row_idx    = torch.arange(H, device=device).unsqueeze(1)
            col_idx    = torch.arange(H, device=device).unsqueeze(0)
            power_idx  = (row_idx - col_idx).clamp(min=0)
            causal     = (row_idx >= col_idx).to(torch.float64)
            W_blocks   = A_stack[power_idx] * causal.unsqueeze(-1).unsqueeze(-1)
            W_toeplitz = W_blocks.permute(0, 2, 1, 3).reshape(H * d, H * d)

            gammas = (self.gamma ** torch.arange(1, H + 1, device=device,
                                                 dtype=torch.float64)).unsqueeze(1)

        for _ in range(self.plan_iters):
            opt.zero_grad()
            u      = torch.tanh(u_logits)
            X_flat = (u @ B.T).reshape(H * d, 1)
            Z      = ZIR + (W_toeplitz @ X_flat).reshape(H, d)
            Z32    = Z.to(torch.float32)
            if self.cumulative:
                loss = -(gammas * agent.v_net(Z32)).sum()
            else:
                loss = -agent.v_net(Z32[-1]).unsqueeze(0).mean()
            (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
            u_logits.grad = grad_u
            opt.step()

        self._u_logits = u_logits.detach()
        self._opt      = opt
        with torch.no_grad():
            return (torch.tanh(u_logits[0]) * self.action_scale).to(torch.float32).cpu().numpy()


def plan_action_toeplitz_continuous(
    agent,
    state:        np.ndarray,
    horizon:      int   = 10,
    plan_iters:   int   = 20,
    lr:           float = 0.1,
    gamma:        float = 0.95,
    action_scale: float = 1.0,
    cumulative:   bool  = True,
    cfg: Config = None,
) -> np.ndarray:
    """
    Block-Toeplitz GEMM planner for continuous action spaces.

    Identical structure to plan_action_toeplitz (discrete) but parameterises
    actions as tanh-squashed logits instead of Gumbel-Softmax one-hots.

        X[t] = B tanh(u_logits[t]) ∈ R^d

    ZIR and W_toeplitz are pre-computed once; each plan_iter is a single
    dense GEMM (W_toeplitz @ X_flat) instead of H sequential dyn_step calls.

    Requires ortho_a=True for the linear superposition Z = ZIR + W X to hold.
    Returns action ∈ [-action_scale, action_scale]^action_dim.
    """
    if cfg:
        gamma = cfg.algo.gamma
    device     = next(agent.parameters()).device
    d          = agent.d
    action_dim = agent.B.shape[1]

    with torch.no_grad():
        z0 = agent.encoder(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        ).squeeze(0).to(torch.float64)  # [d]

        A = agent.A.detach().to(torch.float64)
        B = agent.B.detach().to(torch.float64)  # [d, action_dim]

        A_pows = [torch.eye(d, device=device, dtype=torch.float64)]
        for _ in range(horizon):
            A_pows.append(A_pows[-1] @ A)
        A_stack = torch.stack(A_pows)  # [H+1, d, d]

        ZIR = torch.einsum('kij,j->ki', A_stack[1:], z0)  # [H, d]

        row_idx   = torch.arange(horizon, device=device).unsqueeze(1)
        col_idx   = torch.arange(horizon, device=device).unsqueeze(0)
        power_idx = (row_idx - col_idx).clamp(min=0)
        causal    = (row_idx >= col_idx).to(torch.float64)
        W_blocks  = A_stack[power_idx] * causal.unsqueeze(-1).unsqueeze(-1)
        W_toeplitz = W_blocks.permute(0, 2, 1, 3).reshape(horizon * d, horizon * d)

        gammas = (gamma ** torch.arange(1, horizon + 1, device=device,
                                        dtype=torch.float64)).unsqueeze(1)

    u_logits = torch.randn(horizon, action_dim, device=device,
                           dtype=torch.float64) * 1e-4
    u_logits.requires_grad_(True)
    opt = optim.Adam([u_logits], lr=lr)

    for _ in range(plan_iters):
        opt.zero_grad()

        u      = torch.tanh(u_logits)           # [H, action_dim] float64
        X      = u @ B.T                         # [H, d]
        X_flat = X.reshape(horizon * d, 1)

        ZSR  = (W_toeplitz @ X_flat).reshape(horizon, d)
        Z    = ZIR + ZSR                         # [H, d] float64

        # v_net expects float32 — cast Z for the forward pass only
        Z32 = Z.to(torch.float32)
        if cumulative:
            loss = -(gammas * agent.v_net(Z32)).sum()
        else:
            loss = -agent.v_net(Z32[-1]).unsqueeze(0).mean()

        (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
        u_logits.grad = grad_u
        opt.step()

    with torch.no_grad():
        return (torch.tanh(u_logits[0]) * action_scale).to(torch.float32).cpu().numpy()


def plan_action_continuous(
    agent,
    state:      np.ndarray,
    horizon:    int   = 10,
    plan_iters: int   = 20,
    lr:         float = 0.1,
    frozen_b:   bool  = False,
    cfg: Config = None,
) -> np.ndarray:
    """Differentiable MPC for continuous action spaces. Returns action vector.

    frozen_b=True detaches B before the planning loop (matches Toeplitz behaviour)
    to isolate whether live vs detached B explains the training performance gap.
    """
    device     = next(agent.parameters()).device
    action_dim = agent.B.shape[1]
    B          = agent.B.detach() if frozen_b else agent.B

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
            latent_step = (B @ u[t]).unsqueeze(0)
            z_t         = agent.dyn_step(z_t, latent_step)
        loss = -agent.v_net(z_t).mean()

        (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
        u_logits.grad = grad_u
        opt.step()

    with torch.no_grad():
        return torch.tanh(u_logits[0]).cpu().numpy()
