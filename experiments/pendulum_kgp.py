"""
Pendulum-v1 sanity check for KoopmanGradientPlanner (continuous MPC).

Architecture:
  state_dim=3 — [cos θ, sin θ, θ̇]    n_actions=1 — scalar torque, B ∈ R^{d×1}
  action_scale=2.0                      torque ∈ [-2, 2]

Visualizations produced (saved to ./viz_pendulum/):
  dashboard_step{N}.png  — 6-panel training summary every VIZ_EVERY steps
  filmstrip_step{N}.png  — 8 rendered frames from an eval episode
  final_summary.png      — value map, phase portraits, plan vs actual, koopman quality

Success criterion: episode returns consistently above -300 in 20k steps.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

from koopman_rl.model import KoopmanGradientPlanner, TargetNetwork

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STATE_DIM    = 3
ACTION_DIM   = 1
ACTION_SCALE = 2.0
D            = 32
GAMMA        = 0.99
LR           = 3e-4
KOOP_LR_SCALE = 0.5
BATCH_SIZE   = 256
BUFFER_SIZE  = 100_000
N_STEPS      = 20_000
WARMUP       = 10_000
NOISE_START  = 1.0
NOISE_END    = 0.1
NOISE_DECAY  = 15_000
LAMBDA_KOOP  = 1.0
LAMBDA_RECON = 0.5
LAMBDA_V     = 1.0
EMA_TAU      = 0.005
PLAN_HORIZON     = 5     # horizon during training (speed)
PLAN_ITERS       = 10    # gradient iters during training
VIZ_PLAN_HORIZON = 20    # longer horizon for viz plan rollouts
VIZ_PLAN_ITERS   = 50    # more iters for viz (offline, no speed pressure)
REWARD_SCALE     = 10.0  # divide rewards before TD: keeps V in [-150, 0] range
VIZ_EVERY        = 5_000 # dashboard + filmstrip interval
VIZ_DIR          = "output/viz/pendulum"
CKPT_DIR         = "output/checkpoints/pendulum"

# Discrete-action Toeplitz test
N_DISC_ACTIONS = 9   # torque levels uniformly spaced in [-ACTION_SCALE, ACTION_SCALE]
N_EVAL_PLAN    = 20  # episodes per planner in the speed/quality benchmark

os.makedirs(VIZ_DIR,  exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck noise
# ---------------------------------------------------------------------------

class OUNoise:
    """
    Ornstein-Uhlenbeck process: dx = θ(μ - x)dt + σ dW

    Produces temporally correlated noise that mean-reverts to μ=0.
    Compared to i.i.d. Gaussian noise, OU generates smoother action
    trajectories that explore contiguous regions of the state space —
    important for pendulum swing-up where sustained torque in one
    direction is needed to build momentum.

    sigma decays externally by passing sigma= on each call to sample().
    """
    def __init__(self, size: int, theta: float = 0.15, dt: float = 0.05):
        self.size  = size
        self.theta = theta
        self.dt    = dt
        self.state = np.zeros(size, dtype=np.float32)

    def reset(self):
        self.state[:] = 0.0

    def sample(self, sigma: float) -> np.ndarray:
        self.state += -self.theta * self.state * self.dt + \
                      sigma * np.sqrt(self.dt) * np.random.randn(self.size).astype(np.float32)
        return self.state.copy()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(agent, target, episode_returns, koop_log, v_log,
                    mean_eval_return, path=None):
    """Save agent weights + training history to .pt file."""
    path = path or os.path.join(CKPT_DIR, "kgp_pendulum.pt")
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "target_encoder":   target.encoder.state_dict(),
        "target_v_net":     target.v_net.state_dict(),
        "episode_returns":  episode_returns,
        "koop_log":         koop_log,
        "v_log":            v_log,
        "config": {
            "state_dim": STATE_DIM, "action_dim": ACTION_DIM,
            "d": D, "n_steps": N_STEPS, "gamma": GAMMA,
            "reward_scale": REWARD_SCALE,
            "mean_eval_return": mean_eval_return,
        },
    }, path)
    print(f"  [ckpt] {path}")


def load_checkpoint(path=None):
    """
    Reload a saved agent from disk.

    Usage:
        agent, ckpt = load_checkpoint("viz_pendulum/kgp_pendulum.pt")
        plot_final_summary(agent, ckpt["episode_returns"], buf)
    """
    path = path or os.path.join(CKPT_DIR, "kgp_pendulum.pt")
    ckpt  = torch.load(path, map_location="cpu")
    cfg   = ckpt["config"]
    agent = KoopmanGradientPlanner(
        state_dim=cfg["state_dim"], d=cfg["d"], n_actions=cfg["action_dim"]
    )
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    return agent, ckpt


# ---------------------------------------------------------------------------
# Minimal continuous replay buffer
# ---------------------------------------------------------------------------

class ContinuousReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.states   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.next_s   = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.ptr = self.size = 0

    def push(self, s, a, r, ns, d):
        self.states[self.ptr]  = s
        self.next_s[self.ptr]  = ns
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.dones[self.ptr]   = float(d)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) -> dict:
        idx = np.random.randint(0, self.size, n)
        return {k: getattr(self, k)[idx]
                for k in ("states", "next_s", "actions", "rewards", "dones")}


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _to_theta_thetadot(states: np.ndarray):
    """Convert [cos θ, sin θ, θ̇] → (θ, θ̇).  θ ∈ [-π, π]."""
    theta    = np.arctan2(states[:, 1], states[:, 0])
    thetadot = states[:, 2]
    return theta, thetadot


def _value_grid(agent, n_theta=60, n_tdot=50):
    """
    Evaluate V_ψ(enc(s)) over a regular (θ, θ̇) grid.
    Returns theta_vals, tdot_vals, V_grid [n_tdot, n_theta].
    """
    theta_vals = np.linspace(-np.pi, np.pi, n_theta, dtype=np.float32)
    tdot_vals  = np.linspace(-8.0,   8.0,   n_tdot,  dtype=np.float32)

    cos_t = np.cos(theta_vals)
    sin_t = np.sin(theta_vals)
    grid_states = []
    for td in tdot_vals:
        row = np.column_stack([cos_t, sin_t, np.full(n_theta, td)])
        grid_states.append(row)
    grid_np = np.concatenate(grid_states, axis=0).astype(np.float32)  # [n_t*n_td, 3]

    device = next(agent.parameters()).device
    with torch.no_grad():
        z = agent.encode(torch.tensor(grid_np, device=device))
        v = agent.v_net(z).cpu().numpy()

    V_grid = v.reshape(n_tdot, n_theta)
    return theta_vals, tdot_vals, V_grid


def _decode_plan_rollout_dyn(agent, start_state: np.ndarray,
                             horizon: int = 15, plan_iters: int = 30, lr: float = 0.05):
    """
    Like _decode_plan_rollout but uses agent.dyn_step — works for both
    ortho_a=True (linear, no normalise) and ortho_a=False (normalised sphere).
    """
    device = next(agent.parameters()).device
    with torch.no_grad():
        z0 = agent.encoder(
            torch.tensor(start_state, dtype=torch.float32).unsqueeze(0).to(device)
        )

    u_logits = torch.zeros(horizon, ACTION_DIM, device=device, requires_grad=True)
    opt_p = optim.Adam([u_logits], lr=lr)

    for _ in range(plan_iters):
        opt_p.zero_grad()
        z_t = z0
        u   = torch.tanh(u_logits)
        for t in range(horizon):
            z_t = agent.dyn_step(z_t, (agent.B @ u[t]).unsqueeze(0))
        loss = -agent.v_net(z_t).mean()
        (g,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
        u_logits.grad = g
        opt_p.step()

    with torch.no_grad():
        u_opt = torch.tanh(u_logits)
        zs, z_t = [z0], z0
        for t in range(horizon):
            z_t = agent.dyn_step(z_t, (agent.B @ u_opt[t]).unsqueeze(0))
            zs.append(z_t)
        decoded   = agent.decoder(torch.cat(zs, dim=0)).cpu().numpy()
        thetas    = np.arctan2(decoded[:, 1], decoded[:, 0])
        thetadots = decoded[:, 2]
        actions   = (u_opt * ACTION_SCALE).cpu().numpy().flatten()
    return thetas, thetadots, actions


def _decode_plan_rollout(agent, start_state: np.ndarray,
                         horizon: int = 20, plan_iters: int = 50, lr: float = 0.05):
    """
    Run the continuous MPC optimizer from start_state, then replay the
    optimized action sequence through the Koopman model and decode each
    latent z_t → [cos θ, sin θ, θ̇] via agent.decoder.

    Returns:
      thetas    — [horizon+1] decoded θ values (rad)
      thetadots — [horizon+1] decoded θ̇ values
      actions   — [horizon]   optimized torques ∈ [-ACTION_SCALE, ACTION_SCALE]
    """
    device = next(agent.parameters()).device
    with torch.no_grad():
        z0 = agent.encoder(
            torch.tensor(start_state, dtype=torch.float32).unsqueeze(0).to(device)
        )

    u_logits = torch.randn(horizon, ACTION_DIM, device=device) * 1e-4
    u_logits.requires_grad_(True)
    opt = optim.Adam([u_logits], lr=lr)

    for _ in range(plan_iters):
        opt.zero_grad()
        z_t = z0
        u   = torch.tanh(u_logits)
        for t in range(horizon):
            z_t = F.normalize(z_t @ agent.A.T + (agent.B @ u[t]).unsqueeze(0), dim=-1)
        loss = -agent.v_net(z_t).mean()
        (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
        u_logits.grad = grad_u
        opt.step()

    with torch.no_grad():
        u_opt = torch.tanh(u_logits)  # [H, 1], normalised
        zs    = [z0]
        z_t   = z0
        for t in range(horizon):
            z_t = F.normalize(z_t @ agent.A.T + (agent.B @ u_opt[t]).unsqueeze(0), dim=-1)
            zs.append(z_t)
        Z       = torch.cat(zs, dim=0)             # [H+1, d]
        decoded = agent.decoder(Z).cpu().numpy()   # [H+1, 3]
        thetas    = np.arctan2(decoded[:, 1], decoded[:, 0])
        thetadots = decoded[:, 2]
        actions   = (u_opt * ACTION_SCALE).cpu().numpy().flatten()

    return thetas, thetadots, actions


def _run_real_episode(agent, max_steps=200):
    """
    Run one greedy episode, collecting (states, rewards).
    Returns states_np [T, 3] and total_return.
    """
    ev = gym.make("Pendulum-v1")
    s, _ = ev.reset()
    states, ret = [s.copy()], 0.0
    for _ in range(max_steps):
        a = agent.act_plan_continuous(s, horizon=PLAN_HORIZON,
                                      plan_iters=PLAN_ITERS,
                                      action_scale=ACTION_SCALE)
        s, r, term, trunc, _ = ev.step(a)
        states.append(s.copy())
        ret += r
        if term or trunc:
            break
    ev.close()
    return np.array(states, dtype=np.float32), ret


def _render_episode(agent, n_frames: int = 8):
    """
    Render one greedy episode, return (frames list of HxWx3 arrays, return).
    Frames are sampled evenly over the episode.
    """
    ev = gym.make("Pendulum-v1", render_mode="rgb_array")
    s, _ = ev.reset()
    all_frames, ret = [], 0.0
    all_frames.append(ev.render())
    for _ in range(200):
        a = agent.act_plan_continuous(s, horizon=PLAN_HORIZON,
                                      plan_iters=PLAN_ITERS,
                                      action_scale=ACTION_SCALE)
        s, r, term, trunc, _ = ev.step(a)
        all_frames.append(ev.render())
        ret += r
        if term or trunc:
            break
    ev.close()

    # Sample n_frames evenly
    indices = np.linspace(0, len(all_frames) - 1, n_frames, dtype=int)
    return [all_frames[i] for i in indices], ret


def _colorline(ax, x, y, cmap="plasma", lw=2.0):
    """Draw a line with colour fading from start (dark) to end (bright)."""
    t    = np.linspace(0, 1, len(x))
    pts  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, array=t, cmap=cmap, linewidth=lw, zorder=3)
    ax.add_collection(lc)
    return lc


# ---------------------------------------------------------------------------
# Dashboard: 6-panel training summary
# ---------------------------------------------------------------------------

def plot_dashboard(agent, episode_returns, koop_log, v_log, buf, step):
    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)
    fig.suptitle(f"KoopmanGradientPlanner — Pendulum-v1   step {step:,}", fontsize=13)

    # ── [0,0] Episode returns + EMA ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(episode_returns, alpha=0.35, color="#4c8bc9", lw=1.0, label="raw")
    if len(episode_returns) >= 5:
        alpha = 2 / (min(20, len(episode_returns)) + 1)
        ema   = [episode_returns[0]]
        for r in episode_returns[1:]:
            ema.append(ema[-1] * (1 - alpha) + r * alpha)
        ax.plot(ema, color="#4c8bc9", lw=2.0, label="EMA")
    ax.axhline(-300, color="#2ca02c", linestyle="--", lw=1.2, label="target −300")
    ax.axhline(-1200, color="#d62728", linestyle=":", lw=1.0, label="random")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Episode Returns")
    ax.legend(fontsize=7, loc="lower right")

    # ── [0,1] Loss curves ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if koop_log:
        ax.semilogy(koop_log, color="#e377c2", lw=1.5, label="L_koop")
    if v_log:
        ax.semilogy(v_log,    color="#17becf", lw=1.5, label="L_v")
    ax.set_xlabel("Log step (×1k)")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Training Losses")
    ax.legend(fontsize=8)

    # ── [0,2] Value heatmap V(θ, θ̇) ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    try:
        theta_g, tdot_g, V_grid = _value_grid(agent)
        im = ax.imshow(
            V_grid,
            extent=[-np.pi, np.pi, -8, 8],
            aspect="auto", origin="lower",
            cmap="viridis",
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("θ (rad)")
        ax.set_ylabel("θ̇ (rad/s)")
        ax.set_title("Value Function V(θ, θ̇)")
        ax.axvline(0, color="white", lw=0.8, alpha=0.5)   # θ=0 = upright
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    except Exception as e:
        ax.text(0.5, 0.5, f"value grid failed\n{e}", ha="center", va="center",
                transform=ax.transAxes)

    # ── [1,0] State visitation scatter ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if buf.size > 0:
        s_all   = buf.states[:buf.size]
        th, td  = _to_theta_thetadot(s_all)
        t_color = np.arange(buf.size) / buf.size   # time → colour
        sc = ax.scatter(th, td, c=t_color, cmap="cool",
                        s=1.5, alpha=0.4, rasterized=True)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="time (normalised)")
    ax.set_xlabel("θ (rad)")
    ax.set_ylabel("θ̇ (rad/s)")
    ax.set_title("State Visitation (buffer)")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-8, 8)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["-π", "0", "π"])

    # ── [1,1] Plans overlay on value heatmap ────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    try:
        theta_g, tdot_g, V_grid = _value_grid(agent)
        ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                  aspect="auto", origin="lower", cmap="viridis", alpha=0.75)
        ax.axvline(0, color="white", lw=0.8, alpha=0.4)

        # Starting states: hanging, π/2 with spin, −π/2
        starts = [
            ("hang",   np.array([ -1.0,  0.0,  0.0], np.float32)),
            ("+π/2",   np.array([  0.0,  1.0,  2.0], np.float32)),
            ("−π/2",   np.array([  0.0, -1.0, -2.0], np.float32)),
        ]
        colors = ["#ff7f0e", "#2ca02c", "#d62728"]
        for (label, s0), col in zip(starts, colors):
            try:
                th_p, td_p, _ = _decode_plan_rollout(agent, s0, horizon=VIZ_PLAN_HORIZON, plan_iters=VIZ_PLAN_ITERS)
                _colorline(ax, th_p, td_p, cmap="Oranges" if col == "#ff7f0e"
                           else "Greens" if col == "#2ca02c" else "Reds", lw=2.2)
                ax.plot(th_p[0], td_p[0], "o", color=col, ms=6, zorder=4, label=label)
                ax.plot(th_p[-1], td_p[-1], "*", color=col, ms=8, zorder=4)
            except Exception:
                pass
        ax.legend(fontsize=7, loc="upper right")
    except Exception as e:
        ax.text(0.5, 0.5, f"plan viz failed\n{e}", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("θ (rad)")
    ax.set_ylabel("θ̇ (rad/s)")
    ax.set_title("Planned Trajectories (decoded)")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-8, 8)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["-π", "0", "π"])

    # ── [1,2] Value histogram from buffer ───────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    if buf.size > 0:
        s_sub = buf.states[:min(buf.size, 4096)]
        device = next(agent.parameters()).device
        with torch.no_grad():
            z   = agent.encode(torch.tensor(s_sub, device=device))
            vs  = agent.v_net(z).cpu().numpy()
        ax.hist(vs, bins=50, color="#9467bd", edgecolor="none", alpha=0.8)
        ax.axvline(vs.mean(), color="white", lw=1.5, linestyle="--",
                   label=f"mean {vs.mean():.1f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("V(s)")
    ax.set_ylabel("Count")
    ax.set_title("Value Distribution (buffer)")

    path = os.path.join(VIZ_DIR, f"dashboard_step{step:05d}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {path}")


# ---------------------------------------------------------------------------
# Filmstrip: rendered episode frames
# ---------------------------------------------------------------------------

def plot_filmstrip(agent, step, n_frames: int = 8):
    try:
        frames, ret = _render_episode(agent, n_frames=n_frames)
    except Exception as e:
        print(f"  [viz] filmstrip failed: {e}")
        return

    fig, axes = plt.subplots(1, n_frames, figsize=(2.5 * n_frames, 3))
    fig.suptitle(f"Rendered episode — return {ret:.1f}   (step {step:,})", fontsize=11)
    for ax, frame, i in zip(axes, frames, range(n_frames)):
        ax.imshow(frame)
        ax.axis("off")
        ax.set_title(f"t={i * 200 // (n_frames - 1)}", fontsize=7)
    plt.tight_layout()
    path = os.path.join(VIZ_DIR, f"filmstrip_step{step:05d}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {path}")


# ---------------------------------------------------------------------------
# Final summary: value map + phase portrait + plan-vs-actual + Koopman quality
# ---------------------------------------------------------------------------

def plot_final_summary(agent, episode_returns, buf):
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.38)
    fig.suptitle("Final Summary — KoopmanGradientPlanner Pendulum-v1", fontsize=14)

    # ── [0,0] Episode returns full history ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(episode_returns, alpha=0.3, color="#4c8bc9", lw=1)
    if len(episode_returns) >= 5:
        w  = max(1, len(episode_returns) // 20)
        ma = np.convolve(episode_returns, np.ones(w) / w, mode="valid")
        ax.plot(np.arange(w - 1, len(episode_returns)), ma, color="#4c8bc9", lw=2)
    ax.axhline(-300, color="#2ca02c", lw=1.2, ls="--", label="−300")
    ax.set_title("Episode Returns")
    ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.legend(fontsize=8)

    # ── [0,1] Value heatmap (high-res) ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    theta_g, tdot_g, V_grid = _value_grid(agent, n_theta=80, n_tdot=60)
    im = ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                   aspect="auto", origin="lower", cmap="plasma")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axvline(0, color="white", lw=1.0, alpha=0.6, label="θ=0 (up)")
    ax.set_title("Value Function V(θ, θ̇)")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])

    # ── [0,2] Phase portrait: real episode trajectory overlaid on value map ─
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
              aspect="auto", origin="lower", cmap="plasma", alpha=0.6)
    try:
        states_real, ret_real = _run_real_episode(agent, max_steps=200)
        th_r, td_r = _to_theta_thetadot(states_real)
        lc = _colorline(ax, th_r, td_r, cmap="cool", lw=2.5)
        ax.plot(th_r[0],  td_r[0],  "wo", ms=7, zorder=5, label="start")
        ax.plot(th_r[-1], td_r[-1], "w*", ms=9, zorder=5, label="end")
        ax.set_title(f"Phase Portrait (return {ret_real:.0f})")
    except Exception as e:
        ax.set_title(f"Phase Portrait (failed: {e})")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
    ax.legend(fontsize=7)

    # ── [0,3] Plan vs actual: predicted and real trajectory from hang ───────
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
              aspect="auto", origin="lower", cmap="plasma", alpha=0.55)
    hang_state = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # θ = π
    _plan_rollout = _decode_plan_rollout_dyn if agent._ortho_a else _decode_plan_rollout
    try:
        th_p, td_p, acts = _plan_rollout(agent, hang_state,
                                         horizon=VIZ_PLAN_HORIZON, plan_iters=VIZ_PLAN_ITERS)
        _colorline(ax, th_p, td_p, cmap="Oranges", lw=2.5)
        ax.plot(th_p[0], td_p[0], "o", color="#ff7f0e", ms=7, zorder=5,
                label="plan start")
        ax.plot(th_p[-1], td_p[-1], "*", color="#ff7f0e", ms=9, zorder=5,
                label="plan end")
    except Exception:
        pass
    ax.set_title("Plan from Hanging (decoded)")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
    ax.legend(fontsize=7)

    # ── [1,0–1] Koopman prediction quality: predicted vs target next state ──
    ax = fig.add_subplot(gs[1, :2])
    if buf.size > 0:
        n = min(buf.size, 512)
        idx = np.random.choice(buf.size, n, replace=False)
        s_np  = buf.states[idx]
        ns_np = buf.next_s[idx]
        a_np  = buf.actions[idx]
        device = next(agent.parameters()).device
        with torch.no_grad():
            s_t  = torch.tensor(s_np,  device=device)
            ns_t = torch.tensor(ns_np, device=device)
            a_t  = torch.tensor(a_np,  device=device)
            z    = agent.encode(s_t)
            z_pred_lat = agent.dyn_step(z, a_t @ agent.B.T)
            z_next_lat = agent.encode(ns_t)
            # Decode predicted and actual next states
            s_pred = agent.decoder(z_pred_lat).cpu().numpy()
            s_next = agent.decoder(z_next_lat).cpu().numpy()

        state_labels = ["cos θ", "sin θ", "θ̇"]
        colors_comp  = ["#e377c2", "#17becf", "#bcbd22"]
        for i, (label, col) in enumerate(zip(state_labels, colors_comp)):
            ax.scatter(s_next[:, i], s_pred[:, i], s=4, alpha=0.35,
                       color=col, label=label, rasterized=True)
        lo = min(s_next.min(), s_pred.min())
        hi = max(s_next.max(), s_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="ideal")
        ax.set_xlabel("z_target (decoded actual next state)")
        ax.set_ylabel("z_pred  (Koopman prediction)")
        ax.set_title("Koopman Prediction Quality: predicted vs. actual next state")
        ax.legend(fontsize=8, markerscale=3)

    # ── [1,2] Planned action sequence from hanging ──────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    try:
        th_p, td_p, acts = _plan_rollout(agent, hang_state,
                                         horizon=VIZ_PLAN_HORIZON, plan_iters=VIZ_PLAN_ITERS)
        t_ax = np.arange(len(acts))
        ax.bar(t_ax, acts, color="#ff7f0e", alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Plan step")
        ax.set_ylabel("Torque")
        ax.set_title("Planned Action Sequence (from hang)")
        ax.set_ylim(-ACTION_SCALE * 1.1, ACTION_SCALE * 1.1)
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)

    # ── [1,3] Value along planned trajectory ────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    try:
        device = next(agent.parameters()).device
        with torch.no_grad():
            z0 = agent.encoder(
                torch.tensor(hang_state, dtype=torch.float32).unsqueeze(0).to(device)
            )
        # Replay decoded plan and compute V at each step
        u_logits = torch.randn(VIZ_PLAN_HORIZON, ACTION_DIM, device=device) * 1e-4
        u_logits.requires_grad_(True)
        opt_tmp = optim.Adam([u_logits], lr=0.05)
        for _ in range(VIZ_PLAN_ITERS):
            opt_tmp.zero_grad()
            z_t = z0
            u   = torch.tanh(u_logits)
            for t in range(VIZ_PLAN_HORIZON):
                z_t = agent.dyn_step(z_t, (agent.B @ u[t]).unsqueeze(0))
            loss = -agent.v_net(z_t).mean()
            (g,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
            u_logits.grad = g
            opt_tmp.step()
        with torch.no_grad():
            u_opt = torch.tanh(u_logits)
            z_t   = z0
            v_traj = [agent.v_net(z_t).item()]
            for t in range(VIZ_PLAN_HORIZON):
                z_t = agent.dyn_step(z_t, (agent.B @ u_opt[t]).unsqueeze(0))
                v_traj.append(agent.v_net(z_t).item())
        ax.plot(v_traj, "o-", color="#9467bd", lw=2, ms=5)
        ax.set_xlabel("Plan step")
        ax.set_ylabel("V_ψ(z_t)")
        ax.set_title("Value Along Planned Trajectory")
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)

    path = os.path.join(VIZ_DIR, "final_summary.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {path}")


# ---------------------------------------------------------------------------
# Discrete Toeplitz test: train with ortho_a=True, compare planners
# ---------------------------------------------------------------------------

def _save_live_plot(episode_returns, koop_log, v_log, step, ortho_err,
                    agent=None, buf=None, path="pendulum_toeplitz_live.png"):
    """
    2×3 live dashboard: training metrics (top) + state-space panels (bottom).
    Saved to a fixed path every VIZ_EVERY steps for easy monitoring.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Pendulum Koopman (ortho_a=True, SVD) — step {step:,}"
                 f"   ‖AᵀA−I‖²={ortho_err:.1e}", fontsize=12)

    # ── [0,0] Episode returns ────────────────────────────────────────────────
    ax = axes[0, 0]
    if episode_returns:
        ax.plot(episode_returns, alpha=0.3, color="#4c8bc9", lw=0.8)
        w = min(30, len(episode_returns))
        if len(episode_returns) >= w:
            ma = np.convolve(episode_returns, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(w - 1, len(episode_returns)), ma,
                    color="#4c8bc9", lw=2, label=f"MA-{w}")
    ax.axhline(-300,  color="#2ca02c", ls="--", lw=1.2, label="−300 target")
    ax.axhline(-1200, color="#d62728", ls=":",  lw=1.0, label="random")
    ax.set_title("Episode Returns"); ax.set_xlabel("Episode")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── [0,1] Training losses ────────────────────────────────────────────────
    ax = axes[0, 1]
    if koop_log:
        ax.semilogy(koop_log, color="#e377c2", lw=1.5, label="L_koop")
    if v_log:
        ax.semilogy(v_log,    color="#17becf", lw=1.5, label="L_v")
    ax.set_title("Training Losses"); ax.set_xlabel("Log step (×1k)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── [0,2] Solve rate (ret > −300, rolling) ───────────────────────────────
    ax = axes[0, 2]
    if len(episode_returns) >= 10:
        succ = [1.0 if r > -300 else 0.0 for r in episode_returns]
        w    = min(30, len(succ))
        roll = np.convolve(succ, np.ones(w) / w, mode="valid") * 100
        ax.plot(np.arange(w - 1, len(succ)), roll, color="#ff7f0e", lw=2)
    ax.set_ylim(0, 105); ax.set_title("Solve Rate (ret>−300, MA-30)")
    ax.set_xlabel("Episode"); ax.set_ylabel("%"); ax.grid(alpha=0.3)

    # ── [1,0] Value heatmap V(θ, θ̇) ─────────────────────────────────────────
    ax = axes[1, 0]
    if agent is not None:
        try:
            theta_g, tdot_g, V_grid = _value_grid(agent)
            im = ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                           aspect="auto", origin="lower", cmap="viridis")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axvline(0, color="white", lw=0.8, alpha=0.5)
            ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
    ax.set_title("Value Function V(θ, θ̇)")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")

    # ── [1,1] State visitation on phase portrait ──────────────────────────────
    ax = axes[1, 1]
    if agent is not None:
        try:
            theta_g, tdot_g, V_grid = _value_grid(agent)
            ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                      aspect="auto", origin="lower", cmap="viridis", alpha=0.55)
        except Exception:
            pass
    if buf is not None and buf.size > 0:
        s_sub  = buf.states[:buf.size]
        th, td = _to_theta_thetadot(s_sub)
        t_col  = np.arange(buf.size) / buf.size
        ax.scatter(th, td, c=t_col, cmap="cool", s=1.5, alpha=0.4, rasterized=True)
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
    ax.set_title("State Visitation (phase portrait)")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")

    # ── [1,2] Planned trajectories (Toeplitz decoded) ────────────────────────
    ax = axes[1, 2]
    if agent is not None:
        try:
            theta_g, tdot_g, V_grid = _value_grid(agent)
            ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                      aspect="auto", origin="lower", cmap="viridis", alpha=0.6)
            ax.axvline(0, color="white", lw=0.8, alpha=0.4)
        except Exception:
            pass
        starts = [
            ("hang",  np.array([-1.0,  0.0,  0.0], np.float32)),
            ("+π/2",  np.array([ 0.0,  1.0,  2.0], np.float32)),
            ("-π/2",  np.array([ 0.0, -1.0, -2.0], np.float32)),
        ]
        cmaps  = ["Oranges", "Greens", "Reds"]
        colors = ["#ff7f0e", "#2ca02c", "#d62728"]
        for (label, s0), cmap_name, col in zip(starts, cmaps, colors):
            try:
                th_p, td_p, _ = _decode_plan_rollout_dyn(agent, s0)
                _colorline(ax, th_p, td_p, cmap=cmap_name, lw=2.0)
                ax.plot(th_p[0],  td_p[0],  "o", color=col, ms=6, zorder=4, label=label)
                ax.plot(th_p[-1], td_p[-1], "*", color=col, ms=8, zorder=4)
            except Exception:
                pass
        ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
    ax.set_title("Planned Trajectories (Toeplitz decoded)")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [live] {path}", flush=True)


def run_continuous_toeplitz(device: torch.device, n_steps: int = N_STEPS,
                            cumulative: bool = False, sequential: bool = False,
                            frozen_b: bool = False, seed: int = 0,
                            ou_noise: bool = False):
    """
    Train KoopmanGradientPlanner on Pendulum-v1 with continuous torque actions
    and ortho_a=True (SVD Procrustes on CUDA → exact A ∈ O(d), linear dynamics).

    Training uses act_plan_continuous (tanh-squash MPC) for exploration.
    At the end, benchmarks plan_action_continuous vs plan_action_toeplitz_continuous
    over N_EVAL_PLAN episodes each, reporting quality (mean return) and wall time.
    """
    from koopman_rl.model import KoopmanGradientPlanner, TargetNetwork
    from koopman_rl.planner import (plan_action_continuous,
                                   plan_action_toeplitz_continuous,
                                   WarmStartToeplitzPlanner)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env    = gym.make("Pendulum-v1")
    agent  = KoopmanGradientPlanner(state_dim=STATE_DIM, d=D,
                                    n_actions=ACTION_DIM,
                                    ortho_a=True, device=device)
    agent.encoder.no_normalize = True   # linear dynamics — no L2 normalisation
    target = TargetNetwork(agent)       # deepcopy inherits no_normalize
    buf    = ContinuousReplayBuffer(BUFFER_SIZE, STATE_DIM, ACTION_DIM)

    neural_params = (list(agent.encoder.parameters()) +
                     list(agent.v_net.parameters()) +
                     list(agent.decoder.parameters()))
    koop_params   = agent.koop_parameters()   # handles SVD parametrisation on CUDA
    opt = optim.Adam([
        {"params": neural_params, "lr": LR},
        {"params": koop_params,   "lr": LR * KOOP_LR_SCALE},
    ])

    agent.to(device)
    target.encoder.to(device)
    target.v_net.to(device)

    planner_tag = "seq" if sequential else "toe"
    if sequential and frozen_b:
        planner_tag = "seq_frozb"
    if not sequential and cumulative:
        planner_tag = "toe_cumul"
    noise_tag = "_ou" if ou_noise else ""
    run_tag  = f"{planner_tag}{noise_tag}_s{seed}"
    live_png = os.path.join(VIZ_DIR, f"pendulum_live_{run_tag}.png")

    print("=" * 64)
    print("  Pendulum-v1 — Continuous Koopman + Toeplitz planner test")
    print(f"  device={device}  hard_ortho={agent._use_hard_ortho}")
    print(f"  action_dim={ACTION_DIM}  d={D}  steps={n_steps:,}")
    print(f"  ortho_a=True → linear dynamics (no L2 normalisation)")
    print(f"  run_tag={run_tag}")
    print("=" * 64)
    print(f"\n[Warmup: {WARMUP} random steps...]\n")

    state, _ = env.reset()
    ep_return = 0.0
    episode_returns = []
    koop_log, v_log = [], []
    recent_koop, recent_v = [], []
    ou = OUNoise(ACTION_DIM) if ou_noise else None
    warm_planner = None
    best_ret     = -float("inf")
    best_ckpt    = os.path.join(CKPT_DIR, f"kgp_pendulum_{run_tag}_best.pt")
    t0 = time.time()

    for step in range(1, n_steps + 1):
        noise = max(NOISE_END,
                    NOISE_START - (NOISE_START - NOISE_END) * step / NOISE_DECAY)

        if step < WARMUP:
            action = np.random.uniform(-ACTION_SCALE, ACTION_SCALE, (ACTION_DIM,))
        else:
            if sequential:
                action = agent.act_plan_continuous(
                    state, horizon=PLAN_HORIZON, plan_iters=PLAN_ITERS,
                    action_scale=ACTION_SCALE, frozen_b=frozen_b,
                )
            else:
                action = agent.act_plan_toeplitz_continuous(
                    state, horizon=PLAN_HORIZON, plan_iters=PLAN_ITERS,
                    gamma=GAMMA, action_scale=ACTION_SCALE,
                    cumulative=cumulative,
                )
            exploration = (ou.sample(sigma=noise * ACTION_SCALE) if ou_noise
                           else np.random.normal(0, noise * ACTION_SCALE, size=action.shape))
            action = np.clip(action + exploration, -ACTION_SCALE, ACTION_SCALE)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buf.push(state, action, reward, next_state, done)
        ep_return += reward

        if done:
            episode_returns.append(ep_return)
            ep_return = 0.0
            state, _ = env.reset()
            if ou is not None:
                ou.reset()
            if warm_planner is not None:
                warm_planner.reset()
        else:
            state = next_state

        if buf.size < BATCH_SIZE:
            continue

        batch = buf.sample(BATCH_SIZE)
        s_b   = torch.from_numpy(batch["states"]).to(device)
        ns_b  = torch.from_numpy(batch["next_s"]).to(device)
        a_b   = torch.from_numpy(batch["actions"]).to(device)   # [Bs, ACTION_DIM]
        r_b   = torch.from_numpy(batch["rewards"]).to(device)
        d_b   = torch.from_numpy(batch["dones"]).to(device)

        z_src = agent.encode(s_b)
        with torch.no_grad():
            z_dst_tgt = target.encoder(ns_b)

        # Koopman loss — linear dynamics (dyn_step skips normalisation when ortho_a=True)
        z_pred = agent.dyn_step(z_src, a_b @ agent.B.T)
        L_koop = ((z_pred - z_dst_tgt.detach()).pow(2)
                  .sum(dim=-1) * (1.0 - d_b)).mean()

        # Reconstruction anchor
        L_recon = (agent.decoder(z_src) - s_b).pow(2).mean()

        # 1-step TD value loss
        with torch.no_grad():
            V_next = target.v_net(z_dst_tgt)
            y_td   = r_b / REWARD_SCALE + GAMMA * V_next * (1.0 - d_b)
        L_v = (agent.v_net(z_src) - y_td).pow(2).mean()

        # Soft ortho penalty (MPS/CPU fallback; never applied on CUDA)
        L_ortho = (agent.ortho_penalty()
                   if (agent._ortho_a and not agent._use_hard_ortho)
                   else torch.tensor(0.0, device=device))

        loss = LAMBDA_KOOP * L_koop + LAMBDA_RECON * L_recon + LAMBDA_V * L_v + L_ortho
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
        opt.step()
        target.update(agent, tau=EMA_TAU)

        recent_koop.append(L_koop.item())
        recent_v.append(L_v.item())

        if step % 1_000 == 0:
            elapsed = time.time() - t0; t0 = time.time()
            mk = np.mean(recent_koop); mv = np.mean(recent_v)
            koop_log.append(mk); v_log.append(mv)
            recent20 = episode_returns[-20:] if episode_returns else []
            ret20    = np.mean(recent20) if recent20 else float("nan")
            ortho_err = agent.ortho_error()
            print(f"  step {step:5d}  noise={noise:.3f}  L_koop={mk:.4f}  L_v={mv:.4f}"
                  f"  ret/20={ret20:7.1f}"
                  f"  ‖AᵀA-I‖²={ortho_err:.1e}  sps={1000/elapsed:.0f}", flush=True)
            recent_koop.clear(); recent_v.clear()

            if ret20 > best_ret:
                best_ret = ret20
                save_checkpoint(agent, target, episode_returns, koop_log, v_log,
                                best_ret, path=best_ckpt)
                print(f"  [best] ret/20={best_ret:.1f} → {best_ckpt}", flush=True)

        if step % VIZ_EVERY == 0:
            _save_live_plot(episode_returns, koop_log, v_log, step,
                            agent.ortho_error(), agent=agent, buf=buf,
                            path=live_png)

    env.close()

    # ── Planner speed / quality benchmark ────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  Planner benchmark — {N_EVAL_PLAN} episodes each")
    print(f"  H={PLAN_HORIZON}  iters={PLAN_ITERS}  γ={GAMMA}")
    print("=" * 64)

    eval_env = gym.make("Pendulum-v1")
    planners = [
        ("sequential (baseline)",
         lambda s: plan_action_continuous(agent, s, PLAN_HORIZON, PLAN_ITERS)),
        ("toeplitz  (GEMM)",
         lambda s: plan_action_toeplitz_continuous(agent, s, PLAN_HORIZON, PLAN_ITERS,
                                                   gamma=GAMMA, action_scale=ACTION_SCALE)),
    ]

    for name, fn in planners:
        t_plan = time.time()
        rets = []
        for _ in range(N_EVAL_PLAN):
            s, _ = eval_env.reset()
            ret = 0.0
            for _ in range(200):
                a = fn(s)
                s, r, term, trunc, _ = eval_env.step(a)
                ret += r
                if term or trunc:
                    break
            rets.append(ret)
        elapsed = time.time() - t_plan
        print(f"  {name:30s}  mean={np.mean(rets):8.1f}  std={np.std(rets):6.1f}"
              f"  wall={elapsed:.1f}s  ({elapsed/N_EVAL_PLAN:.2f}s/ep)")

    eval_env.close()
    save_checkpoint(agent, target, episode_returns, koop_log, v_log,
                    np.mean(episode_returns[-20:]) if episode_returns else float("nan"),
                    path=os.path.join(CKPT_DIR, f"kgp_pendulum_{run_tag}.pt"))
    _save_live_plot(episode_returns, koop_log, v_log, n_steps,
                    agent.ortho_error(), agent=agent, buf=buf, path=live_png)
    plot_final_summary(agent, episode_returns, buf)


# ---------------------------------------------------------------------------
# Main: setup + training + evaluation + final viz
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cumulative", action="store_true",
                        help="Use cumulative value objective during training (default: terminal-only)")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential planner for data collection (default: Toeplitz GEMM)")
    parser.add_argument("--frozen_b", action="store_true",
                        help="Detach B in sequential planner (isolates live-vs-frozen-B hypothesis)")
    parser.add_argument("--ou_noise", action="store_true",
                        help="Use Ornstein-Uhlenbeck noise instead of i.i.d. Gaussian")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        _dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _dev = torch.device(args.device)

    run_continuous_toeplitz(_dev, n_steps=args.steps, cumulative=args.cumulative,
                            sequential=args.sequential, frozen_b=args.frozen_b,
                            seed=args.seed, ou_noise=args.ou_noise)
