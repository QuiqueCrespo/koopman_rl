"""
Pendulum-v1 entry point for KoopmanGradientPlanner (continuous MPC).

Architecture:
  state_dim=3 — [cos θ, sin θ, θ̇]    n_actions=1 — scalar torque, B ∈ R^{d×1}
  action_scale=2.0                      torque ∈ [-2, 2]

Visualizations produced (saved to output/viz/pendulum/):
  pendulum_live_{run_name}.png  — live 2×3 dashboard updated every VIZ_EVERY steps
  final_summary.png             — value map, phase portraits, plan vs actual, koopman quality

Success criterion: episode returns consistently above -300 in 30k steps.
"""

import argparse
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

from koopman_rl.config import (Config, EnvConfig, ModelConfig, BufferConfig,
                                AlgoConfig, TrainConfig, PlannerConfig)
from koopman_rl.trainer_continuous import train_continuous
from koopman_rl.planner import plan_cem_gradient_batch, CEMPlannerWarmStart, ToeplitzPlannerWarmStart

# ---------------------------------------------------------------------------
# Pendulum-specific constants
# ---------------------------------------------------------------------------
ACTION_DIM       = 1
ACTION_SCALE     = 2.0
PLAN_HORIZON     = 5   # online planning (benchmark + data collection)
PLAN_ITERS       = 100   # online planning iterations
VIZ_PLAN_HORIZON = 30   # offline viz: longer horizon so swing-up is visible
VIZ_PLAN_ITERS   = 100   # offline viz: more iterations, no speed pressure
VIZ_DIR          = "output/viz/pendulum"

# Pendulum goal: upright at rest — [cos(0), sin(0), θ̇=0]
PENDULUM_GOAL = np.array([1.0, 0.0, 0.0], dtype=np.float32)

os.makedirs(VIZ_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_pendulum_cfg(args) -> Config:
    """Build Config from parsed argparse args."""
    sequential = args.sequential
    frozen_b   = args.frozen_b
    ou_noise   = args.ou_noise
    seed       = args.seed

    noise_tag = "_ou" if ou_noise else ""
    run_name  = f"pendulum_policy{noise_tag}_s{seed}"

    return Config(
        env=EnvConfig(
            state_dim=3,
            n_actions=ACTION_DIM,
            action_scale=ACTION_SCALE,
            continuous=True,
        ),
        model=ModelConfig(
            d=16,
            lr=3e-4,
            ema_tau=0.005,
            ortho_a=False,
        ),
        buffer=BufferConfig(
            capacity=100_000,
            batch_size=512,
        ),
        algo=AlgoConfig(
            gamma=0.99,
            lambda_koop=1.0,
            lambda_recon=1.0,
            lambda_v=1.0,
            koop_lr_scale=1.0,
            reward_scale=10.0,
            n_envs=10,
            noise_z_std=0.01,
        ),
        train=TrainConfig(
            n_steps=args.steps,
            warmup=20_000,
            noise_start=1.0,
            noise_end=0.1,
            noise_decay=15_000,
            viz_every=5_000,
            viz_dir=VIZ_DIR,
            ckpt_dir="output/checkpoints/pendulum",
            planner_type="policy",
            frozen_b=frozen_b,
            ou_noise=ou_noise,
        ),
        planner=PlannerConfig(
            horizon=PLAN_HORIZON,
            plan_iters=PLAN_ITERS,
            lr=0.1,
            cem_iters=10,
            cem_samples=200,
            cem_elites=20,
            cem_grad_iters=PLAN_ITERS,
        ),
        run_name=run_name,
        seed=seed,
        device=args.device,
    )


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
    Evaluate value over a regular (θ, θ̇) grid.
    Continuous agent: Q(enc(s), π(enc(s))) — on-policy Q-value.
    Discrete agent:   V(enc(s)).
    Returns theta_vals, tdot_vals, V_grid [n_tdot, n_theta].
    """
    agent.eval()
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
        if hasattr(agent, 'q_net') and hasattr(agent, 'pi_net'):
            a = agent.pi_net(z)
            v = agent.q_net(torch.cat([z, a], -1)).squeeze(-1).cpu().numpy()
        else:
            v = agent.v_net(z).cpu().numpy()

    V_grid = v.reshape(n_tdot, n_theta)
    return theta_vals, tdot_vals, V_grid


def _decode_plan_rollout_dyn(agent, start_state: np.ndarray,
                             horizon: int = 15, plan_iters: int = 30, lr: float = 0.05):
    """
    Run the continuous MPC optimizer from start_state using agent.dyn_step
    agent.eval() is set on entry — this is pure inference.
    (works for both ortho_a=True and ortho_a=False), then decode each
    latent z_t → [cos θ, sin θ, θ̇] via agent.decoder.

    Returns:
      thetas    — [horizon+1] decoded θ values (rad)
      thetadots — [horizon+1] decoded θ̇ values
      actions   — [horizon]   optimized torques ∈ [-ACTION_SCALE, ACTION_SCALE]
    """
    agent.eval()
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
        # Terminal: Q(z_H, π(z_H)) for continuous; V(z_H) for discrete
        if hasattr(agent, 'q_net') and hasattr(agent, 'pi_net'):
            a_t  = agent.pi_net(z_t)
            loss = -agent.q_net(torch.cat([z_t, a_t], -1)).mean()
        else:
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


def _run_real_episode(agent, cfg, max_steps=200):
    """
    Run one greedy episode, collecting (states, rewards).
    Returns states_np [T, 3] and total_return.
    """
    agent.eval()
    ev = gym.make("Pendulum-v1")
    s, _ = ev.reset()
    states, ret = [s.copy()], 0.0
    for _ in range(max_steps):
        a = agent.act_policy_continuous_batch(
            s[np.newaxis], action_scale=cfg.env.action_scale)[0]
        s, r, term, trunc, _ = ev.step(a)
        states.append(s.copy())
        ret += r
        if term or trunc:
            break
    ev.close()
    return np.array(states, dtype=np.float32), ret


def _render_episode(agent, cfg, n_frames: int = 8):
    """
    Render one greedy episode, return (frames list of HxWx3 arrays, return).
    Frames are sampled evenly over the episode.
    """
    agent.eval()
    ev = gym.make("Pendulum-v1", render_mode="rgb_array")
    s, _ = ev.reset()
    all_frames, ret = [], 0.0
    all_frames.append(ev.render())
    for _ in range(200):
        a = agent.act_policy_continuous_batch(
            s[np.newaxis], action_scale=cfg.env.action_scale)[0]
        s, r, term, trunc, _ = ev.step(a)
        all_frames.append(ev.render())
        ret += r
        if term or trunc:
            break
    ev.close()

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
# Live dashboard (2×3): training metrics + state-space panels
# ---------------------------------------------------------------------------

def _save_live_plot(episode_returns, koop_log, q_log=None, step=0, ortho_err=0.0,
                    agent=None, buf=None, path="pendulum_toeplitz_live.png",
                    **_kw):  # absorb any extra kwargs from old callers
    """
    2×3 live dashboard saved to a fixed path every viz_every steps.
    Called via the on_viz callback from train_continuous.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Pendulum Koopman (ortho_a=False) — step {step:,}"
                 f"   ‖AᵀA−I‖²={ortho_err:.1e}", fontsize=12)

    # ── [0,0] Episode returns ─────────────────────────────────────────────────
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

    # ── [0,1] Training losses ─────────────────────────────────────────────────
    ax = axes[0, 1]
    if koop_log:
        ax.semilogy(koop_log, color="#e377c2", lw=1.5, label="L_koop")
    if q_log:
        ax.semilogy(q_log,    color="#17becf", lw=1.5, label="L_q")
    ax.set_title("Training Losses"); ax.set_xlabel("Log step (×1k)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── [0,2] Solve rate (ret > −300, rolling) ────────────────────────────────
    ax = axes[0, 2]
    if len(episode_returns) >= 10:
        succ = [1.0 if r > -300 else 0.0 for r in episode_returns]
        w    = min(30, len(succ))
        roll = np.convolve(succ, np.ones(w) / w, mode="valid") * 100
        ax.plot(np.arange(w - 1, len(succ)), roll, color="#ff7f0e", lw=2)
    ax.set_ylim(0, 105); ax.set_title("Solve Rate (ret>−300, MA-30)")
    ax.set_xlabel("Episode"); ax.set_ylabel("%"); ax.grid(alpha=0.3)

    # Pre-compute value grid once (shared across the three bottom panels)
    _vg = None
    if agent is not None:
        try:
            _vg = _value_grid(agent)
        except Exception:
            pass

    # ── [1,0] Value heatmap V(θ, θ̇) ──────────────────────────────────────────
    ax = axes[1, 0]
    if _vg is not None:
        theta_g, tdot_g, V_grid = _vg
        im = ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                       aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axvline(0, color="white", lw=0.8, alpha=0.5)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.set_title("Value Function V(θ, θ̇)")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")

    # ── [1,1] State visitation on phase portrait ───────────────────────────────
    ax = axes[1, 1]
    if _vg is not None:
        theta_g, tdot_g, V_grid = _vg
        ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                  aspect="auto", origin="lower", cmap="viridis", alpha=0.55)
    if buf is not None and buf.size > 0:
        s_sub  = buf.states[:buf.size]
        th, td = _to_theta_thetadot(s_sub)
        t_col  = np.arange(buf.size) / buf.size
        ax.scatter(th, td, c=t_col, cmap="cool", s=1.5, alpha=0.4, rasterized=True)
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
    ax.set_title("State Visitation (phase portrait)")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")

    # ── [1,2] Planned trajectories (decoded) ──────────────────────────────────
    ax = axes[1, 2]
    if _vg is not None:
        theta_g, tdot_g, V_grid = _vg
        ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                  aspect="auto", origin="lower", cmap="viridis", alpha=0.6)
        ax.axvline(0, color="white", lw=0.8, alpha=0.4)
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
    ax.set_title("Planned Trajectories (decoded)")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [live] {path}", flush=True)


# ---------------------------------------------------------------------------
# Final summary: value map + phase portrait + plan-vs-actual + Koopman quality
# ---------------------------------------------------------------------------

def plot_final_summary(agent, episode_returns, buf, cfg=None):
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.38)
    fig.suptitle("Final Summary — KoopmanGradientPlanner Pendulum-v1", fontsize=14)

    # ── [0,0] Episode returns full history ────────────────────────────────────
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

    # ── [0,1] Value heatmap (high-res) ───────────────────────────────────────
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

    # ── [0,2] Phase portrait: real episode trajectory overlaid on value map ──
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
              aspect="auto", origin="lower", cmap="plasma", alpha=0.6)
    try:
        _cfg = cfg  # may be None; _run_real_episode handles defaults below
        states_real, ret_real = _run_real_episode(agent, _cfg or _default_pendulum_cfg(),
                                                  max_steps=200)
        th_r, td_r = _to_theta_thetadot(states_real)
        _colorline(ax, th_r, td_r, cmap="cool", lw=2.5)
        ax.plot(th_r[0],  td_r[0],  "wo", ms=7, zorder=5, label="start")
        ax.plot(th_r[-1], td_r[-1], "w*", ms=9, zorder=5, label="end")
        ax.set_title(f"Phase Portrait (return {ret_real:.0f})")
    except Exception as e:
        ax.set_title(f"Phase Portrait (failed: {e})")
    ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")
    ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
    ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"])
    ax.legend(fontsize=7)

    # ── [0,3] Plan vs actual: predicted and real trajectory from hang ─────────
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
              aspect="auto", origin="lower", cmap="plasma", alpha=0.55)
    hang_state = np.array([-1.0, 0.0, 0.0], dtype=np.float32)  # θ = π
    try:
        th_p, td_p, acts = _decode_plan_rollout_dyn(agent, hang_state,
                                                     horizon=VIZ_PLAN_HORIZON,
                                                     plan_iters=VIZ_PLAN_ITERS)
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

    # ── [1,0–1] Koopman prediction quality ────────────────────────────────────
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

    # ── [1,2] Planned action sequence from hanging ────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    try:
        th_p, td_p, acts = _decode_plan_rollout_dyn(agent, hang_state,
                                                     horizon=VIZ_PLAN_HORIZON,
                                                     plan_iters=VIZ_PLAN_ITERS)
        t_ax = np.arange(len(acts))
        ax.bar(t_ax, acts, color="#ff7f0e", alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Plan step")
        ax.set_ylabel("Torque")
        ax.set_title("Planned Action Sequence (from hang)")
        ax.set_ylim(-ACTION_SCALE * 1.1, ACTION_SCALE * 1.1)
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)

    # ── [1,3] Value along planned trajectory ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 3])
    try:
        device = next(agent.parameters()).device
        with torch.no_grad():
            z0 = agent.encoder(
                torch.tensor(hang_state, dtype=torch.float32).unsqueeze(0).to(device)
            )
        u_logits = torch.zeros(VIZ_PLAN_HORIZON, ACTION_DIM, device=device, requires_grad=True)
        opt_tmp = optim.Adam([u_logits], lr=0.05)
        for _ in range(VIZ_PLAN_ITERS):
            opt_tmp.zero_grad()
            z_t = z0
            u   = torch.tanh(u_logits)
            for t in range(VIZ_PLAN_HORIZON):
                z_t = agent.dyn_step(z_t, (agent.B @ u[t]).unsqueeze(0))
            if hasattr(agent, 'q_net') and hasattr(agent, 'pi_net'):
                a_t  = agent.pi_net(z_t)
                loss = -agent.q_net(torch.cat([z_t, a_t], -1)).mean()
            else:
                loss = -agent.v_net(z_t).mean()
            (g,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
            u_logits.grad = g
            opt_tmp.step()
        with torch.no_grad():
            u_opt = torch.tanh(u_logits)
            z_t   = z0
            def _val(z):
                if hasattr(agent, 'q_net') and hasattr(agent, 'pi_net'):
                    a = agent.pi_net(z)
                    return agent.q_net(torch.cat([z, a], -1)).item()
                return agent.v_net(z).item()
            v_traj = [_val(z_t)]
            for t in range(VIZ_PLAN_HORIZON):
                z_t = agent.dyn_step(z_t, (agent.B @ u_opt[t]).unsqueeze(0))
                v_traj.append(_val(z_t))
        ax.plot(v_traj, "o-", color="#9467bd", lw=2, ms=5)
        ax.set_xlabel("Plan step")
        ax.set_ylabel("Q(z_t, π(z_t))" if hasattr(agent, 'q_net') else "V_ψ(z_t)")
        ax.set_title("Value Along Planned Trajectory")
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha="center", va="center", transform=ax.transAxes)

    path = os.path.join(VIZ_DIR, "final_summary.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {path}")


def plot_plan_evolution(agent, cfg, capture_at=None):
    """
    Show how the Toeplitz plan (decoded latent trajectory + action sequence) evolves
    over Adam iterations for a set of diverse starting states.

    Layout: 2 rows per starting state (top = phase trajectory, bottom = action bars),
    one column per captured iteration snapshot.
    """
    agent.eval()
    device     = next(agent.parameters()).device
    horizon    = cfg.planner.horizon
    plan_iters = cfg.planner.plan_iters
    gamma      = cfg.algo.gamma
    action_scale = cfg.env.action_scale
    d          = agent.d
    action_dim = agent.B.shape[1]

    if capture_at is None:
        capture_at = sorted({0, max(1, plan_iters // 4), plan_iters // 2,
                             max(1, 3 * plan_iters // 4), plan_iters})

    starts = [
        ("hanging\nω=0",    np.array([-1.0,  0.0,  0.0], np.float32)),
        ("side\nω=+3",      np.array([ 0.0,  1.0,  3.0], np.float32)),
        ("side\nω=−3",      np.array([ 0.0, -1.0, -3.0], np.float32)),
    ]
    n_snaps = len(capture_at)
    n_starts = len(starts)

    fig, axes = plt.subplots(
        n_starts * 2, n_snaps,
        figsize=(2.6 * n_snaps, 3.2 * n_starts),
        squeeze=False,
    )
    fig.suptitle(
        f"Toeplitz Plan Evolution  (H={horizon}  plan_iters={plan_iters}  γ={gamma})",
        fontsize=11,
    )

    with torch.no_grad():
        B = agent.B.detach()
        W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)

    gammas_path = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)
    h_ax        = np.arange(horizon)

    for s_idx, (label, s0) in enumerate(starts):
        row_traj = s_idx * 2
        row_act  = s_idx * 2 + 1

        with torch.no_grad():
            z0  = agent.encoder(
                torch.tensor(s0[np.newaxis], dtype=torch.float32, device=device)
            )
            ZIR = torch.einsum('kij,nj->nki', A_stack[1:], z0)   # [1, H, d]

        u_logits = torch.zeros(1, horizon, action_dim, device=device, requires_grad=True)
        opt      = optim.Adam([u_logits], lr=0.1)

        snap_col = 0
        for it in range(plan_iters + 1):
            if it in capture_at:
                with torch.no_grad():
                    u      = torch.tanh(u_logits)
                    X_flat = (u @ B.T).reshape(1, horizon * d)
                    Z      = ZIR + (X_flat @ W_toeplitz.T).reshape(1, horizon, d)
                    all_z  = torch.cat([z0.unsqueeze(1), Z], dim=1).squeeze(0)  # [H+1, d]
                    dec    = agent.decoder(all_z).cpu().numpy()
                    thetas    = np.arctan2(dec[:, 1], dec[:, 0])
                    thetadots = dec[:, 2]
                    actions   = (u[0, :, 0] * action_scale).cpu().numpy()

                # ── phase trajectory ───────────────────────────────────────
                ax = axes[row_traj, snap_col]
                ax.plot(thetas, thetadots, 'o-', lw=1.5, ms=3, color='#4c8bc9')
                ax.plot(thetas[0],  thetadots[0],  'go', ms=7, zorder=5)
                ax.plot(thetas[-1], thetadots[-1], 'r*', ms=9, zorder=5)
                ax.axvline(0, color='gray', lw=0.6, alpha=0.5)
                ax.set_xlim(-np.pi, np.pi)
                ax.set_ylim(-8, 8)
                ax.set_xticks([-np.pi, 0, np.pi])
                ax.set_xticklabels(['-π', '0', 'π'], fontsize=7)
                ax.tick_params(labelsize=7)
                if snap_col == 0:
                    ax.set_ylabel(f"{label}\nθ̇", fontsize=8)
                if s_idx == 0:
                    ax.set_title(f"iter {it}", fontsize=9)

                # ── action bars ────────────────────────────────────────────
                ax = axes[row_act, snap_col]
                bar_colors = ['#e377c2' if a >= 0 else '#17becf' for a in actions]
                ax.bar(h_ax, actions, color=bar_colors, alpha=0.85, width=0.7)
                ax.axhline(0, color='black', lw=0.6)
                ax.set_ylim(-action_scale * 1.15, action_scale * 1.15)
                ax.set_xlim(-0.5, horizon - 0.5)
                ax.set_xticks(h_ax)
                ax.tick_params(labelsize=7)
                if snap_col == 0:
                    ax.set_ylabel("torque", fontsize=8)
                ax.set_xlabel("step", fontsize=7)

                snap_col += 1

            if it < plan_iters:
                opt.zero_grad()
                u      = torch.tanh(u_logits)
                X_flat = (u @ B.T).reshape(1, horizon * d)
                Z      = ZIR + (X_flat @ W_toeplitz.T).reshape(1, horizon, d)
                ZU     = torch.cat([Z, u], dim=-1)
                disc_path = (gammas_path * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
                z_H   = Z[:, -1, :]
                a_H   = agent.pi_net(z_H)
                q_H   = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
                loss  = -(disc_path + gamma ** horizon * q_H).mean()
                (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
                u_logits.grad = grad_u
                opt.step()

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "plan_evolution.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {path}")


def plot_policy_vs_planner(agent, cfg, max_steps=200):
    """
    Run real Pendulum-v1 episodes from diverse starting positions using:
      - direct policy  (π only)
      - Toeplitz MPC   (r_net path + Q terminal)
    Plot phase-portrait trajectories side by side, plus a return bar chart.
    """
    agent.eval()
    plan_horizon = cfg.planner.horizon
    plan_iters   = cfg.planner.plan_iters
    gamma        = cfg.algo.gamma
    action_scale = cfg.env.action_scale

    starts = [
        ("hanging ω=0",   np.array([-1.0,  0.0,  0.0],                    np.float32)),
        ("hanging ω=+2",  np.array([-1.0,  0.0,  2.0],                    np.float32)),
        ("hanging ω=−2",  np.array([-1.0,  0.0, -2.0],                    np.float32)),
        ("side ω=0",      np.array([ 0.0,  1.0,  0.0],                    np.float32)),
        ("near top ω=0",  np.array([np.cos(0.4), np.sin(0.4),  0.0],      np.float32)),
        ("near top ω=+2", np.array([np.cos(0.4), np.sin(0.4),  2.0],      np.float32)),
    ]
    colors = ['#e377c2', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']

    def _run_from(s0, action_fn):
        """Run one episode from s0. Returns (states, return, mean_ms_per_step)."""
        if hasattr(action_fn, "reset"):
            action_fn.reset()   # discard warm-start state from previous episode
        ev = gym.make("Pendulum-v1")
        ev.reset()
        theta = np.arctan2(float(s0[1]), float(s0[0]))
        ev.unwrapped.state = np.array([theta, float(s0[2])])
        s = s0.copy()
        states, ret, step_times = [s.copy()], 0.0, []
        for _ in range(max_steps):
            t0 = time.perf_counter()
            a = action_fn(s)
            step_times.append(time.perf_counter() - t0)
            s, r, term, trunc, _ = ev.step(np.atleast_1d(a).astype(np.float32))
            states.append(s.copy())
            ret += r
            if term or trunc:
                break
        ev.close()
        return np.array(states, dtype=np.float32), ret, float(np.mean(step_times)) * 1e3

    try:
        _vg = _value_grid(agent)
    except Exception:
        _vg = None

    device = next(agent.parameters()).device
    with torch.no_grad():
        z_goal = agent.encoder(
            torch.tensor(PENDULUM_GOAL[np.newaxis], dtype=torch.float32, device=device)
        )  # [1, d]

    cem_iters   = cfg.planner.cem_iters
    cem_samples = cfg.planner.cem_samples
    cem_elites  = cfg.planner.cem_elites
    cem_grad    = cfg.planner.cem_grad_iters

    fig, (ax_pol, ax_plan, ax_vfree, ax_cem, ax_ret) = plt.subplots(1, 5, figsize=(30, 7))
    fig.suptitle(
        f"Direct Policy vs Grad Value-Based vs Value-Free vs CEM Value-Based vs CEM Value-Free  "
        f"(H={plan_horizon}  plan_iters={plan_iters}  cem={cem_iters}×{cem_samples}+{cem_grad}grad)",
        fontsize=11,
    )

    # 2×4 grid: [Policy | Grad Value-Based | Value-Free (closed-form) | Value-Free (grad)]
    #           [CEM Value-Based | CEM Value-Free | Returns Bar | (empty)]
    fig_cmp, axes_cmp = plt.subplots(2, 4, figsize=(24, 12))
    fig_cmp.suptitle(
        f"Planner Comparison — Value-Based vs Value-Free × Grad vs CEM  "
        f"(H={plan_horizon}  plan_iters={plan_iters}  cem={cem_iters}×{cem_samples}+{cem_grad}grad)",
        fontsize=11,
    )
    ax_pol    = axes_cmp[0, 0]
    ax_plan   = axes_cmp[0, 1]
    ax_vfree  = axes_cmp[0, 2]   # closed-form
    ax_vfg    = axes_cmp[0, 3]   # gradient descent
    ax_cem_v  = axes_cmp[1, 0]
    ax_cem_f  = axes_cmp[1, 1]
    ax_ret    = axes_cmp[1, 2]
    axes_cmp[1, 3].set_visible(False)

    traj_axes = [
        (ax_pol,   "Policy (π direct)"),
        (ax_plan,  "Grad Value-Based (r_net+Q)"),
        (ax_vfree, "Value-Free (‖z_H−z_goal‖² closed-form)"),
        (ax_vfg,   f"Value-Free (‖z_H−z_goal‖² grad, {plan_iters} iters)"),
        (ax_cem_v, f"CEM Value-Based ({cem_iters}×{cem_samples}+{cem_grad}g)"),
        (ax_cem_f, f"CEM Value-Free  ({cem_iters}×{cem_samples}+{cem_grad}g)"),
    ]
    for ax, title in traj_axes:
        if _vg is not None:
            _, _, V_grid = _vg
            ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                      aspect="auto", origin="lower", cmap="viridis", alpha=0.35)
        ax.axvline(0, color="white", lw=0.8, alpha=0.6)
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels(["-π", "-π/2", "0", "π/2", "π"])
        ax.set_xlabel("θ (rad)"); ax.set_ylabel("θ̇ (rad/s)")
        ax.set_title(title, fontsize=9)

    pol_rets, plan_rets, vfree_rets, vfg_rets, cem_val_rets, cem_vfree_rets = [], [], [], [], [], []
    pol_ms,   plan_ms,  vfree_ms,  vfg_ms,   cem_val_ms,   cem_vfree_ms   = [], [], [], [], [], []

    for (label, s0), col in zip(starts, colors):
        # ── policy ───────────────────────────────────────────────────────────
        pol_states, pol_ret, ms = _run_from(
            s0, lambda s: agent.act_policy_continuous_batch(s[np.newaxis], action_scale)[0])
        pol_ms.append(ms)
        th, td = _to_theta_thetadot(pol_states)
        ax_pol.plot(th, td, color=col, lw=1.6, alpha=0.85)
        ax_pol.plot(th[0], td[0], "o", color=col, ms=7, zorder=5)
        ax_pol.plot(th[-1], td[-1], "*", color=col, ms=9, zorder=5,
                    label=f"{label}  {pol_ret:.0f}")

        # ── grad value-based ─────────────────────────────────────────────────
        plan_states, plan_ret, ms = _run_from(
            s0, lambda s: agent.act_plan_continuous(
                s[np.newaxis], plan_horizon, plan_iters,
                gamma=gamma, action_scale=action_scale)[0])
        plan_ms.append(ms)
        th, td = _to_theta_thetadot(plan_states)
        ax_plan.plot(th, td, color=col, lw=1.6, alpha=0.85)
        ax_plan.plot(th[0], td[0], "o", color=col, ms=7, zorder=5)
        ax_plan.plot(th[-1], td[-1], "*", color=col, ms=9, zorder=5,
                     label=f"{label}  {plan_ret:.0f}")

        # ── value-free closed-form ────────────────────────────────────────────
        vfree_states, vfree_ret, ms = _run_from(
            s0, lambda s: _value_free_act_batch(
                agent, s[np.newaxis], z_goal,
                plan_horizon, plan_iters, gamma, action_scale)[0])
        vfree_ms.append(ms)
        th, td = _to_theta_thetadot(vfree_states)
        ax_vfree.plot(th, td, color=col, lw=1.6, alpha=0.85)
        ax_vfree.plot(th[0], td[0], "o", color=col, ms=7, zorder=5)
        ax_vfree.plot(th[-1], td[-1], "*", color=col, ms=9, zorder=5,
                      label=f"{label}  {vfree_ret:.0f}")

        # ── value-free gradient descent ───────────────────────────────────────
        vfg_states, vfg_ret, ms = _run_from(
            s0, lambda s: _value_free_grad_act_batch(
                agent, s[np.newaxis], z_goal,
                plan_horizon, plan_iters, gamma, action_scale)[0])
        vfg_ms.append(ms)
        th, td = _to_theta_thetadot(vfg_states)
        ax_vfg.plot(th, td, color=col, lw=1.6, alpha=0.85)
        ax_vfg.plot(th[0], td[0], "o", color=col, ms=7, zorder=5)
        ax_vfg.plot(th[-1], td[-1], "*", color=col, ms=9, zorder=5,
                    label=f"{label}  {vfg_ret:.0f}")

        # ── CEM value-based (warm-start) ─────────────────────────────────────
        cem_v_states, cem_val_ret, ms = _run_from(
            s0, CEMPlannerWarmStart(
                agent, plan_horizon, cem_iters, cem_samples, cem_elites,
                cem_grad, 0.1, gamma, action_scale, objective="value"))
        cem_val_ms.append(ms)
        th, td = _to_theta_thetadot(cem_v_states)
        ax_cem_v.plot(th, td, color=col, lw=1.6, alpha=0.85)
        ax_cem_v.plot(th[0], td[0], "o", color=col, ms=7, zorder=5)
        ax_cem_v.plot(th[-1], td[-1], "*", color=col, ms=9, zorder=5,
                      label=f"{label}  {cem_val_ret:.0f}")

        # ── CEM value-free (warm-start) ──────────────────────────────────────
        cem_f_states, cem_vfree_ret, ms = _run_from(
            s0, CEMPlannerWarmStart(
                agent, plan_horizon, cem_iters, cem_samples, cem_elites,
                cem_grad, 0.1, gamma, action_scale,
                objective="value_free", z_goal=z_goal))
        cem_vfree_ms.append(ms)
        th, td = _to_theta_thetadot(cem_f_states)
        ax_cem_f.plot(th, td, color=col, lw=1.6, alpha=0.85)
        ax_cem_f.plot(th[0], td[0], "o", color=col, ms=7, zorder=5)
        ax_cem_f.plot(th[-1], td[-1], "*", color=col, ms=9, zorder=5,
                      label=f"{label}  {cem_vfree_ret:.0f}")

        pol_rets.append(pol_ret);       plan_rets.append(plan_ret)
        vfree_rets.append(vfree_ret);   vfg_rets.append(vfg_ret)
        cem_val_rets.append(cem_val_ret); cem_vfree_rets.append(cem_vfree_ret)

    # Update trajectory panel titles with measured per-step planning time
    timing_labels = [
        (ax_pol,   "Policy (π direct)",                                    pol_ms),
        (ax_plan,  "Grad value-based (r_net+Q)",                           plan_ms),
        (ax_vfree, "Value-free (‖z_H−z_goal‖², closed-form)",              vfree_ms),
        (ax_vfg,   f"Value-free (‖z_H−z_goal‖², grad {plan_iters} iters)", vfg_ms),
        (ax_cem_v, f"CEM value-based ({cem_iters}×{cem_samples}+{cem_grad}g)", cem_val_ms),
        (ax_cem_f, f"CEM value-free  ({cem_iters}×{cem_samples}+{cem_grad}g)", cem_vfree_ms),
    ]
    print("\n  [planner timing]")
    for ax, title, ms_list in timing_labels:
        mean_ms = float(np.mean(ms_list))
        ax.set_title(f"{title}\n{mean_ms:.1f} ms/step", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        print(f"    {title:<45s}  {mean_ms:6.2f} ms/step")

    # ── return bar chart ──────────────────────────────────────────────────────
    x = np.arange(len(starts))
    w = 0.12
    ax_ret.bar(x - 2.5*w, pol_rets,       w, label="Policy",                  color="#4c8bc9", alpha=0.85)
    ax_ret.bar(x - 1.5*w, plan_rets,      w, label="Grad value-based",        color="#ff7f0e", alpha=0.85)
    ax_ret.bar(x - 0.5*w, vfree_rets,     w, label="VF closed-form",          color="#9467bd", alpha=0.85)
    ax_ret.bar(x + 0.5*w, vfg_rets,       w, label="VF grad descent",         color="#8c564b", alpha=0.85)
    ax_ret.bar(x + 1.5*w, cem_val_rets,   w, label="CEM value-based",         color="#2ca02c", alpha=0.85)
    ax_ret.bar(x + 2.5*w, cem_vfree_rets, w, label="CEM value-free",          color="#d62728", alpha=0.85)
    ax_ret.axhline(-300, color="green", ls="--", lw=1.2, label="−300 target")
    ax_ret.set_xticks(x)
    ax_ret.set_xticklabels([l for l, _ in starts], rotation=20, ha="right", fontsize=8)
    ax_ret.set_ylabel("Episode Return")
    ax_ret.set_title("Return by Starting Position")
    ax_ret.legend(fontsize=9)
    ax_ret.grid(alpha=0.3)

    fig_cmp.tight_layout()
    path = os.path.join(VIZ_DIR, "policy_vs_planner.png")
    fig_cmp.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig_cmp)
    print(f"  [viz] {path}")


def _plan_objectives(agent, z0, z_goal, horizon, plan_iters, gamma):
    """
    Run both planners from z0, capturing their objectives at each Adam iteration.

    Value-MPC   (maximised): J = Σ_t γ^t r_net(z_t, u_t) + γ^H Q(z_H, π(z_H))
    Value-free  (minimised): D = ‖z_H − z_goal‖²

    Returns val_curve [plan_iters+1], vfree_curve [plan_iters+1], u_val, u_vfree.
    """
    device = z0.device
    d = agent.d
    action_dim = agent.B.shape[1]

    with torch.no_grad():
        B = agent.B.detach()
        W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)
        ZIR = torch.einsum('kij,nj->nki', A_stack[1:], z0)  # [1, H, d]

    gammas_path = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)

    u_val  = torch.zeros(1, horizon, action_dim, device=device, requires_grad=True)
    u_jepa = torch.zeros(1, horizon, action_dim, device=device, requires_grad=True)
    opt_val  = optim.Adam([u_val],  lr=0.1)
    opt_jepa = optim.Adam([u_jepa], lr=0.1)

    val_curve, vfree_curve = [], []

    for it in range(plan_iters + 1):
        # snapshot both objectives before taking the step
        with torch.no_grad():
            u = torch.tanh(u_val)
            X_flat = (u @ B.T).reshape(1, horizon * d)
            Z = ZIR + (X_flat @ W_toeplitz.T).reshape(1, horizon, d)
            Z_curr = torch.cat([z0.unsqueeze(1), Z[:, :-1, :]], dim=1)
            ZU = torch.cat([Z_curr, u], dim=-1)
            disc = (gammas_path * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
            z_H = Z[:, -1, :]
            a_H = agent.pi_net(z_H)
            q_H = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
            val_curve.append((disc + gamma ** horizon * q_H).mean().item())

            u_j = torch.tanh(u_jepa)
            X_j = (u_j @ B.T).reshape(1, horizon * d)
            Z_j = ZIR + (X_j @ W_toeplitz.T).reshape(1, horizon, d)
            z_H_j = Z_j[:, -1, :]
            vfree_curve.append(((z_H_j - z_goal) ** 2).sum(dim=-1).mean().item())

        if it < plan_iters:
            # value step
            opt_val.zero_grad()
            u = torch.tanh(u_val)
            X_flat = (u @ B.T).reshape(1, horizon * d)
            Z = ZIR + (X_flat @ W_toeplitz.T).reshape(1, horizon, d)
            Z_curr = torch.cat([z0.unsqueeze(1), Z[:, :-1, :]], dim=1)
            ZU = torch.cat([Z_curr, u], dim=-1)
            disc = (gammas_path * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
            z_H = Z[:, -1, :]
            a_H = agent.pi_net(z_H)
            q_H = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
            loss_val = -(disc + gamma ** horizon * q_H).mean()
            (g_val,) = torch.autograd.grad(loss_val, u_val, only_inputs=True)
            u_val.grad = g_val
            opt_val.step()

            # value-free step
            opt_jepa.zero_grad()
            u_j = torch.tanh(u_jepa)
            X_j = (u_j @ B.T).reshape(1, horizon * d)
            Z_j = ZIR + (X_j @ W_toeplitz.T).reshape(1, horizon, d)
            z_H_j = Z_j[:, -1, :]
            loss_jepa = ((z_H_j - z_goal.detach()) ** 2).sum(dim=-1).mean()
            (g_jepa,) = torch.autograd.grad(loss_jepa, u_jepa, only_inputs=True)
            u_jepa.grad = g_jepa
            opt_jepa.step()

    return val_curve, vfree_curve, u_val.detach(), u_jepa.detach()


def _value_free_grad_act_batch(agent, states: np.ndarray, z_goal,
                    horizon, plan_iters, gamma, action_scale) -> np.ndarray:
    """
    Gradient-descent planner: min_u ‖z_H − z_goal‖² − Σ_t γ^t R_φ(z_t, u_t)

    Combines the terminal latent-distance objective with a discounted reward sum
    from the learned R_φ.  Solved with Adam on tanh-squashed logits (soft bound).

    Returns actions [N, action_dim] ∈ [-action_scale, action_scale].
    """
    device = z_goal.device
    d = agent.d
    action_dim = agent.B.shape[1]
    N = len(states)

    with torch.no_grad():
        z0 = agent.encoder(torch.tensor(states, dtype=torch.float32, device=device))
        B = agent.B.detach()
        W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)
        ZIR = torch.einsum('kij,nj->nki', A_stack[1:], z0)  # [N, H, d]
        gammas_path = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)  # [H]

    u_logits = torch.zeros(N, horizon, action_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([u_logits], lr=0.1)

    for _ in range(plan_iters):
        opt.zero_grad()
        u = torch.tanh(u_logits)
        X_flat = (u @ B.T).reshape(N, horizon * d)
        Z = ZIR + (X_flat @ W_toeplitz.T).reshape(N, horizon, d)
        z_H = Z[:, -1, :]

        # discounted reward sum: R_φ(z_t, u_t) for t = 0..H-1
        Z_curr    = torch.cat([z0.unsqueeze(1), Z[:, :-1, :]], dim=1)  # state before each action [N, H, d]
        ZU        = torch.cat([Z_curr, u], dim=-1)                      # [N, H, d+action_dim]
        disc_path = (gammas_path * agent.r_net(ZU).squeeze(-1)).sum(dim=1)  # [N]

        goal_loss = ((z_H - z_goal.detach()) ** 2).sum(dim=-1)  # [N]
        loss = (goal_loss - disc_path).mean()
        (grad_u,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
        u_logits.grad = grad_u
        opt.step()

    with torch.no_grad():
        return (torch.tanh(u_logits[:, 0, :]) * action_scale).cpu().numpy()


def _value_free_act_batch(agent, states: np.ndarray, z_goal,
                    horizon, plan_iters, gamma, action_scale) -> np.ndarray:
    """
    Closed-form value-free planner: min_u ‖z_H − z_goal‖²

    Unrolling the Koopman recursion to horizon H gives:
        z_H = A^H z_0  +  W_reach · u_flat

    W_reach column k = γ^k · A^{H-1-k} · B  [d, H*action_dim]
    The γ^k factor discounts action u_k consistently with the discounted
    value objective: an action fired at step k costs γ^k in the discounted
    system, so its effective contribution to z_H is scaled accordingly.
    W_toeplitz stores undiscounted A^{i-j}; the gamma column-scaling is
    applied here independently.

    Closed-form OLS (no iterations):
        u* = (z_goal − A^H z_0) @ pinv(W_reach).T

    Returns actions [N, action_dim] ∈ [-action_scale, action_scale].
    """
    device = z_goal.device
    d = agent.d
    action_dim = agent.B.shape[1]
    N = len(states)

    with torch.no_grad():
        z0 = agent.encoder(torch.tensor(states, dtype=torch.float32, device=device))
        B = agent.B.detach()
        W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)

        # Free-flight terminal state: A^H z_0  [N, d]
        ZIR_H = torch.einsum('ij,nj->ni', A_stack[horizon], z0)

        # Reachability matrix with discount: W_reach[:,k] = γ^k · A^{H-1-k} · B
        # W_toeplitz[-d:, k*d:(k+1)*d] = A^{H-1-k} (last block row, column-block k)
        # Scale each d-wide column-block by γ^k → [H] discount weights
        IB          = torch.block_diag(*[B] * horizon)                      # [H*d, H*action_dim]
        gamma_cols  = (gamma ** torch.arange(horizon, device=device))        # [H]  γ^0 … γ^{H-1}
        # Broadcast γ^k across the d rows of each column-block of IB
        gamma_scale = gamma_cols.repeat_interleave(d)                        # [H*d]
        IB_disc     = IB * gamma_scale.unsqueeze(1)                          # [H*d, H*action_dim]
        W_reach     = W_toeplitz[-d:] @ IB_disc                             # [d, H*action_dim]

        # Closed-form OLS: u* = residual @ pinv(W_reach).T
        residual = z_goal.expand(N, -1) - ZIR_H             # [N, d]
        u_flat   = residual @ torch.linalg.pinv(W_reach).T  # [N, H*action_dim]
        u0 = u_flat.reshape(N, horizon, action_dim)[:, 0, :]
        return u0.clamp(-action_scale, action_scale).cpu().numpy()


def _cem_best_score(agent, z0, horizon, cem_iters, n_samples, n_elites, gamma):
    """
    Run the CEM phase only (no gradient polish) and return the best J
    found after cem_iters rounds.  Used to annotate plan_convergence plots:
    the returned value shows where CEM lands before gradient polish, which
    is the warm-start point for the gradient phase.

    Uses reduced n_samples for speed in the viz context.
    """
    device     = z0.device
    d          = agent.d
    action_dim = agent.B.shape[1]
    N          = z0.shape[0]

    with torch.no_grad():
        B = agent.B.detach()
        W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)
        ZIR   = torch.einsum('kij,nj->nki', A_stack[1:], z0)  # [N, H, d]
        ZIR_s = ZIR.unsqueeze(1).expand(N, n_samples, horizon, d)
        gammas_path = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)

        mu    = torch.zeros(N, horizon, action_dim, device=device)
        sigma = torch.ones(N, horizon, action_dim, device=device) * 2.0

        for _ in range(cem_iters):
            u_logits_s = (mu.unsqueeze(1)
                          + sigma.unsqueeze(1)
                          * torch.randn(N, n_samples, horizon, action_dim, device=device))
            u_s    = torch.tanh(u_logits_s)
            NS     = N * n_samples
            X_flat = (u_s @ B.T).reshape(NS, horizon * d)
            Z_path = (ZIR_s.reshape(NS, horizon, d)
                      + (X_flat @ W_toeplitz.T).reshape(NS, horizon, d)
                      ).reshape(N, n_samples, horizon, d)

            z0_s   = z0.unsqueeze(1).unsqueeze(2).expand(N, n_samples, 1, d)
            Z_curr = torch.cat([z0_s, Z_path[:, :, :-1, :]], dim=2)
            ZU     = torch.cat([Z_curr, u_s], dim=-1)
            disc_path = (gammas_path * agent.r_net(ZU).squeeze(-1)).sum(dim=2)

            z_H    = Z_path[:, :, -1, :]
            a_H    = agent.pi_net(z_H)
            q_H    = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
            scores = disc_path + gamma ** horizon * q_H

            elite_idx    = scores.topk(n_elites, dim=1).indices
            batch_idx    = torch.arange(N, device=device).unsqueeze(1).expand(N, n_elites)
            elite_logits = u_logits_s[batch_idx, elite_idx]
            mu    = elite_logits.mean(dim=1)
            sigma = elite_logits.std(dim=1).clamp(min=1e-3)

        # Score of the final CEM mean
        u_best = torch.tanh(mu)
        X_flat = (u_best @ B.T).reshape(N, horizon * d)
        Z_path = ZIR + (X_flat @ W_toeplitz.T).reshape(N, horizon, d)
        Z_curr = torch.cat([z0.unsqueeze(1), Z_path[:, :-1, :]], dim=1)
        ZU     = torch.cat([Z_curr, u_best], dim=-1)
        disc_path = (gammas_path.unsqueeze(0) * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
        z_H    = Z_path[:, -1, :]
        a_H    = agent.pi_net(z_H)
        q_H    = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
        score  = (disc_path + gamma ** horizon * q_H).mean().item()
    return score, mu.detach()


def _cem_grad_curve(agent, z0, mu, horizon, grad_iters, gamma):
    """
    Run gradient polish starting from CEM mean mu, recording J at each step.
    Returns curve [grad_iters+1] — one value before the first step, then one per step.
    """
    device     = z0.device
    d          = agent.d
    action_dim = agent.B.shape[1]

    with torch.no_grad():
        B = agent.B.detach()
        W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)
        ZIR = torch.einsum('kij,nj->nki', A_stack[1:], z0)
    gammas_path = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)

    u_logits = mu.clone().requires_grad_(True)
    opt = optim.Adam([u_logits], lr=0.1)
    curve = []

    for it in range(grad_iters + 1):
        with torch.no_grad():
            u      = torch.tanh(u_logits)
            X_flat = (u @ B.T).reshape(1, horizon * d)
            Z      = ZIR + (X_flat @ W_toeplitz.T).reshape(1, horizon, d)
            Z_curr = torch.cat([z0.unsqueeze(1), Z[:, :-1, :]], dim=1)
            ZU     = torch.cat([Z_curr, u], dim=-1)
            disc   = (gammas_path * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
            z_H    = Z[:, -1, :]
            a_H    = agent.pi_net(z_H)
            q_H    = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
            curve.append((disc + gamma ** horizon * q_H).mean().item())

        if it < grad_iters:
            opt.zero_grad()
            u      = torch.tanh(u_logits)
            X_flat = (u @ B.T).reshape(1, horizon * d)
            Z      = ZIR + (X_flat @ W_toeplitz.T).reshape(1, horizon, d)
            Z_curr = torch.cat([z0.unsqueeze(1), Z[:, :-1, :]], dim=1)
            ZU     = torch.cat([Z_curr, u], dim=-1)
            disc   = (gammas_path * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
            z_H    = Z[:, -1, :]
            a_H    = agent.pi_net(z_H)
            q_H    = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
            loss   = -(disc + gamma ** horizon * q_H).mean()
            (g,)   = torch.autograd.grad(loss, u_logits, only_inputs=True)
            u_logits.grad = g
            opt.step()

    return curve


# ---------------------------------------------------------------------------
# CEM instrumented runner + diagnostics
# ---------------------------------------------------------------------------

def _run_cem_instrumented(agent, state: np.ndarray, cfg):
    """
    Run CEM+grad planner on a single state, capturing per-iteration internals.

    Returns dict:
      iter_scores  [cem_iters, n_samples] — raw J for all samples each CEM round
      iter_best    [cem_iters]            — best J per round
      iter_mu0     [cem_iters+1]          — mean first-action logit each round
      iter_sigma0  [cem_iters+1]          — std  first-action logit each round
      iter_best_u  [cem_iters, H]         — best sample action logits each round
      cem_final_u  [H, action_dim]        — CEM mean before grad polish
      polished_u   [H, action_dim]        — action logits after grad polish
      J_cem        scalar                 — J at CEM mean
      J_polished   scalar                 — J after grad polish
      z0           [1, d]                 — encoded start state
    """
    device     = next(agent.parameters()).device
    d          = agent.d
    action_dim = agent.B.shape[1]
    gamma      = cfg.algo.gamma
    horizon    = cfg.planner.horizon
    ci         = cfg.planner.cem_iters
    ns         = cfg.planner.cem_samples
    ne         = cfg.planner.cem_elites
    gi         = cfg.planner.cem_grad_iters
    lr         = cfg.planner.lr

    with torch.no_grad():
        z0 = agent.encoder(
            torch.tensor(state[np.newaxis], dtype=torch.float32, device=device))
        B = agent.B.detach()
        W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)
        ZIR       = torch.einsum('kij,nj->nki', A_stack[1:], z0)          # [1, H, d]
        ZIR_s     = ZIR.unsqueeze(1).expand(1, ns, horizon, d)             # [1, ns, H, d]
        gp        = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)

        mu    = torch.zeros(1, horizon, action_dim, device=device)
        sigma = torch.ones(1, horizon, action_dim, device=device) * 2.0

        iter_scores, iter_best, iter_mu0, iter_sigma0, iter_best_u = [], [], [], [], []
        iter_mu0.append(mu[0, :, 0].cpu().numpy().copy())
        iter_sigma0.append(sigma[0, :, 0].cpu().numpy().copy())

        for _ in range(ci):
            u_logits_s = (mu.unsqueeze(1)
                          + sigma.unsqueeze(1)
                          * torch.randn(1, ns, horizon, action_dim, device=device))
            u_s = torch.tanh(u_logits_s)

            X_flat = (u_s @ B.T).reshape(ns, horizon * d)
            Z_path = (ZIR_s.reshape(ns, horizon, d)
                      + (X_flat @ W_toeplitz.T).reshape(ns, horizon, d)
                      ).reshape(1, ns, horizon, d)

            z0_s   = z0.unsqueeze(1).unsqueeze(2).expand(1, ns, 1, d)
            Z_curr = torch.cat([z0_s, Z_path[:, :, :-1, :]], dim=2)
            ZU     = torch.cat([Z_curr, u_s], dim=-1)
            disc   = (gp * agent.r_net(ZU).squeeze(-1)).sum(dim=2)         # [1, ns]
            z_H    = Z_path[:, :, -1, :]
            a_H    = agent.pi_net(z_H)
            q_H    = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
            scores = (disc + gamma ** horizon * q_H)[0]                     # [ns]

            iter_scores.append(scores.cpu().numpy().copy())
            iter_best.append(float(scores.max()))

            best_idx = int(scores.argmax())
            iter_best_u.append(u_logits_s[0, best_idx, :, 0].cpu().numpy().copy())

            elite_idx    = scores.topk(ne).indices
            elite_logits = u_logits_s[0, elite_idx]                         # [ne, H, ad]
            mu    = elite_logits.mean(dim=0, keepdim=True)
            sigma = elite_logits.std(dim=0, keepdim=True).clamp(min=1e-3)

            iter_mu0.append(mu[0, :, 0].cpu().numpy().copy())
            iter_sigma0.append(sigma[0, :, 0].cpu().numpy().copy())

        # Score the CEM mean
        def _score_u(u_tensor):
            X = (u_tensor @ B.T).reshape(1, horizon * d)
            Z = ZIR + (X @ W_toeplitz.T).reshape(1, horizon, d)
            Zc = torch.cat([z0.unsqueeze(1), Z[:, :-1, :]], dim=1)
            ZU = torch.cat([Zc, u_tensor], dim=-1)
            disc = (gp.unsqueeze(0) * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
            z_H = Z[:, -1, :]
            a_H = agent.pi_net(z_H)
            q_H = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
            return (disc + gamma ** horizon * q_H).item()

        u_cem  = torch.tanh(mu)
        J_cem  = _score_u(u_cem)
        cem_final_u = mu[0].cpu().numpy()

    # Grad polish
    u_logits = mu.clone().requires_grad_(True)
    opt = optim.Adam([u_logits], lr=lr)
    for _ in range(gi):
        opt.zero_grad()
        u = torch.tanh(u_logits)
        X = (u @ B.T).reshape(1, horizon * d)
        Z = ZIR + (X @ W_toeplitz.T).reshape(1, horizon, d)
        Zc = torch.cat([z0.unsqueeze(1), Z[:, :-1, :]], dim=1)
        ZU = torch.cat([Zc, u], dim=-1)
        disc = (gp.unsqueeze(0) * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
        z_H = Z[:, -1, :]
        a_H = agent.pi_net(z_H)
        q_H = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
        loss = -(disc + gamma ** horizon * q_H).mean()
        (g,) = torch.autograd.grad(loss, u_logits, only_inputs=True)
        u_logits.grad = g
        opt.step()

    with torch.no_grad():
        J_polished  = _score_u(torch.tanh(u_logits))
        polished_u  = u_logits.detach()[0].cpu().numpy()

    return dict(
        iter_scores  = np.array(iter_scores),   # [ci, ns]
        iter_best    = np.array(iter_best),      # [ci]
        iter_mu0     = np.array(iter_mu0),       # [ci+1, H]
        iter_sigma0  = np.array(iter_sigma0),    # [ci+1, H]
        iter_best_u  = np.array(iter_best_u),    # [ci, H]
        cem_final_u  = cem_final_u,              # [H, action_dim]
        polished_u   = polished_u,               # [H, action_dim]
        J_cem        = J_cem,
        J_polished   = J_polished,
        z0           = z0.cpu().numpy(),          # [1, d]
    )


def _execute_open_loop(start_state: np.ndarray, actions: np.ndarray):
    """
    Execute action sequence open-loop in Pendulum-v1 without replanning.
    actions: [T, 1] or [T] actual torques in [-action_scale, action_scale].
    Returns (states [T+1, 3], rewards [T]).
    """
    ev = gym.make("Pendulum-v1")
    ev.reset()
    theta = np.arctan2(float(start_state[1]), float(start_state[0]))
    ev.unwrapped.state = np.array([theta, float(start_state[2])])
    s       = start_state.copy()
    states  = [s.copy()]
    rewards = []
    for a in actions:
        s, r, term, trunc, _ = ev.step(np.atleast_1d(a).astype(np.float32))
        states.append(s.copy())
        rewards.append(r)
        if term or trunc:
            break
    ev.close()
    return np.array(states, dtype=np.float32), np.array(rewards, dtype=np.float32)


def plot_model_rollout_accuracy(agent, cfg, n_steps: int = 200):
    """
    Compare a real policy rollout in Pendulum-v1 against the imagined rollout
    obtained by applying the policy autoregressively in latent space:

        z_{t+1} = A z_t + B u_t,   u_t = tanh(π(z_t)) · scale
        ŝ_t     = decoder(z_t)

    If the two trajectories match, the Koopman dynamics are faithful and any
    planning failure is a policy / objective problem.
    If they diverge quickly, the model-reality gap is the bottleneck.

    Saves: output/viz/pendulum/model_rollout_accuracy.png
    """
    agent.eval()
    device       = next(agent.parameters()).device
    action_scale = cfg.env.action_scale
    gamma        = cfg.algo.gamma

    starts = [
        ("hanging ω=0",   np.array([-1.0,  0.0,  0.0], np.float32)),
        ("side ω=+3",     np.array([ 0.0,  1.0,  3.0], np.float32)),
        ("near-top ω=0",  np.array([np.cos(0.3), np.sin(0.3), 0.0], np.float32)),
        ("side ω=−3",     np.array([ 0.0, -1.0, -3.0], np.float32)),
    ]
    n_starts = len(starts)

    # ── collect real and imagined rollouts ─────────────────────────────────────
    real_states_all  = []   # [n_starts][T+1, 3]
    imag_states_all  = []   # [n_starts][T+1, 3]  decoded latent rollout
    real_actions_all = []   # [n_starts][T]
    imag_actions_all = []   # [n_starts][T]
    z_norm_real_all  = []   # [n_starts][T+1]  ||enc(s_t)|| for real states
    z_norm_imag_all  = []   # [n_starts][T+1]  ||z_t|| for imagined rollout
    z_latent_err_all = []   # [n_starts][T+1]  ||z_imag_t - enc(s_real_t)||

    for _, s0 in starts:
        # ---- real rollout ----
        ev = gym.make("Pendulum-v1")
        ev.reset()
        theta_s0 = np.arctan2(float(s0[1]), float(s0[0]))
        ev.unwrapped.state = np.array([theta_s0, float(s0[2])])
        s = s0.copy()
        real_states  = [s.copy()]
        real_actions = []
        for _ in range(n_steps):
            a = agent.act_policy_continuous_batch(
                s[np.newaxis], action_scale=action_scale)[0].astype(np.float32)
            s, _, term, trunc, _ = ev.step(a)
            real_states.append(s.copy())
            real_actions.append(a.copy())
            if term or trunc:
                break
        ev.close()
        T = len(real_actions)

        # ---- imagined rollout + latent tracking ----
        with torch.no_grad():
            real_s_t = torch.tensor(
                np.array(real_states, dtype=np.float32), device=device)  # [T+1, 3]
            z_real_all_t = agent.encoder(real_s_t)                        # [T+1, d]
            z_norms_real = z_real_all_t.norm(dim=1).cpu().numpy()         # [T+1]

            A = agent.A.detach()
            B = agent.B.detach()

            z = z_real_all_t[:1].clone()   # start from encoder of s0
            imag_states   = [agent.decoder(z).cpu().numpy()[0].copy()]
            imag_actions  = []
            z_norms_imag  = [z.norm().item()]
            z_lat_errs    = [0.0]   # by definition equal at t=0

            for t in range(T):
                u = agent.pi_net(z)
                a_imag = (torch.tanh(u) * action_scale).cpu().numpy()[0]
                imag_actions.append(a_imag.copy())
                z = z @ A.T + u @ B.T
                s_hat = agent.decoder(z).cpu().numpy()[0]
                imag_states.append(s_hat.copy())
                z_norms_imag.append(z.norm().item())
                # latent error vs encoder of the actual real state at this step
                z_lat_errs.append((z - z_real_all_t[t + 1:t + 2]).norm().item())

        real_states_all.append(np.array(real_states,  dtype=np.float32))
        imag_states_all.append(np.array(imag_states,  dtype=np.float32))
        real_actions_all.append(np.array(real_actions, dtype=np.float32))
        imag_actions_all.append(np.array(imag_actions, dtype=np.float32))
        z_norm_real_all.append(np.array(z_norms_real,  dtype=np.float32))
        z_norm_imag_all.append(np.array(z_norms_imag,  dtype=np.float32))
        z_latent_err_all.append(np.array(z_lat_errs,   dtype=np.float32))

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, n_starts, figsize=(4.5 * n_starts, 15), squeeze=False)
    fig.suptitle(
        "Model–Reality Gap: Real Policy Rollout vs Latent-Space Imagined Rollout\n"
        "orange = imagined (z_{t+1}=Az_t+Bu_t, decoded)   blue = actual env",
        fontsize=10,
    )

    try:
        vg = _value_grid(agent)
    except Exception:
        vg = None

    for col, (label, _) in enumerate(starts):
        rs  = real_states_all[col]   # [T+1, 3]
        im  = imag_states_all[col]   # [T+1, 3]
        ra  = real_actions_all[col]  # [T]
        ia  = imag_actions_all[col]  # [T]
        T   = len(ra)
        ts  = np.arange(T + 1)

        real_th  = np.arctan2(rs[:, 1], rs[:, 0])
        real_td  = rs[:, 2]
        imag_th  = np.arctan2(im[:, 1], im[:, 0])
        imag_td  = im[:, 2]

        # Row 0 — phase portrait
        ax = axes[0, col]
        if vg is not None:
            _, _, V_grid = vg
            ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                      aspect="auto", origin="lower", cmap="viridis", alpha=0.35)
        ax.plot(real_th, real_td, "o-", color="#4c8bc9", lw=1.5, ms=2.5,
                alpha=0.9, label="real")
        ax.plot(imag_th, imag_td, "o-", color="#ff7f0e", lw=1.5, ms=2.5,
                alpha=0.9, label="imagined")
        ax.plot(real_th[0], real_td[0], "gs", ms=8, zorder=6, label="start")
        ax.plot(real_th[-1], real_td[-1], "b^", ms=7, zorder=6)
        ax.plot(imag_th[-1], imag_td[-1], "r^", ms=7, zorder=6)
        ax.axvline(0, color="white", lw=0.8, alpha=0.5)
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(["-π", "0", "π"], fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("θ"); ax.set_ylabel("θ̇")
        if col == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.2)

        # Row 1 — angle and angular velocity time series
        ax = axes[1, col]
        ax.plot(ts, real_th, color="#4c8bc9", lw=1.5, label="θ real")
        ax.plot(ts, imag_th, color="#ff7f0e", lw=1.5, ls="--", label="θ imagined")
        ax.fill_between(ts, real_th, imag_th, alpha=0.15, color="#d62728")
        ax.set_ylabel("θ (rad)")
        ax.set_xlabel("step")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        if col == 0:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(ts, real_td, color="#2ca02c", lw=1.2, alpha=0.6, label="θ̇ real")
        ax2.plot(ts, imag_td, color="#9467bd", lw=1.2, ls="--", alpha=0.6,
                 label="θ̇ imag")
        ax2.set_ylabel("θ̇ (rad/s)", fontsize=7)
        if col == n_starts - 1:
            ax2.legend(fontsize=6, loc="upper right")

        # Row 2 — per-step decoded-state error and action comparison
        ax = axes[2, col]
        # State prediction error: use cosθ, sinθ, θ̇ directly (avoids angle-wrap)
        err = np.linalg.norm(rs[:T+1] - im[:T+1], axis=1)
        ax.plot(ts, err, color="#d62728", lw=1.5, label="||ŝ − s||")
        ax.fill_between(ts, err, alpha=0.2, color="#d62728")
        ax.set_ylabel("||ŝ − s||", color="#d62728")
        ax.set_xlabel("step")

        # Cumulative error
        ax3 = ax.twinx()
        ax3.plot(ts, np.cumsum(err) / (ts + 1), color="#8c564b", lw=1.2,
                 ls=":", label="mean err")
        ax3.set_ylabel("running mean err", fontsize=7)

        # Annotate divergence step (first time err > 0.5)
        div_steps = np.where(err > 0.5)[0]
        if len(div_steps) > 0:
            ax.axvline(div_steps[0], color="black", lw=1.2, ls="--",
                       label=f"div t={div_steps[0]}")
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Row 3 — latent norm evolution + latent-space divergence
        ax = axes[3, col]
        z_nr = z_norm_real_all[col]   # [T+1]  ||enc(s_t)||
        z_ni = z_norm_imag_all[col]   # [T+1]  ||z_t_imag||
        z_le = z_latent_err_all[col]  # [T+1]  ||z_imag - enc(s_real)||

        ax.plot(ts, z_nr, color="#4c8bc9", lw=1.5, label="‖enc(sₜ)‖  real")
        ax.plot(ts, z_ni, color="#ff7f0e", lw=1.5, ls="--",
                label="‖zₜ‖  imagined")
        ax.set_ylabel("latent norm ‖z‖")
        ax.set_xlabel("step")
        if col == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

        # Latent divergence on twin axis — this is the core model-error signal
        ax4 = ax.twinx()
        ax4.plot(ts, z_le, color="#d62728", lw=1.5, alpha=0.85,
                 label="‖zₜ_imag − enc(sₜ)‖")
        ax4.fill_between(ts, z_le, alpha=0.12, color="#d62728")
        ax4.set_ylabel("latent divergence", color="#d62728", fontsize=7)
        # Mark when latent error first exceeds 1 (normalised by z norm)
        lat_div_steps = np.where(z_le > 1.0)[0]
        if len(lat_div_steps) > 0:
            ax4.axvline(lat_div_steps[0], color="#d62728", lw=1.0, ls=":",
                        alpha=0.7)
            ax4.annotate(f"t={lat_div_steps[0]}",
                         xy=(lat_div_steps[0], z_le[lat_div_steps[0]]),
                         fontsize=6, color="#d62728", ha="left")
        if col == n_starts - 1:
            ax4.legend(fontsize=6, loc="lower right")

    fig.tight_layout()
    path = os.path.join(VIZ_DIR, "model_rollout_accuracy.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {path}")


def plot_natural_dynamics(agent, cfg, n_steps: int = 300):
    """
    Compare the natural (uncontrolled, u=0) evolution of the real Pendulum-v1
    against the imagined free-flight in learned latent space:

        real env:      step with a=0 each time
        latent model:  z_{t+1} = A z_t   (B·0 = 0)
        decoded:       ŝ_t = decoder(z_t)

    Since A ∈ O(d) the imagined latent norm is constant (sphere geodesic),
    while the real pendulum is subject to gravity.  Their divergence reveals
    whether the learnt A correctly encodes natural free-fall/swing dynamics.

    4 rows × n_starts cols:
      Row 0  — phase portrait (real vs imagined decoded), value grid background
      Row 1  — θ and θ̇ time series (real vs imagined)
      Row 2  — ‖ŝ − s‖ decoded state error + running mean
      Row 3  — ‖enc(sₜ)‖ and ‖zₜ_imag‖ norms (left) + latent divergence (right)

    Saves: output/viz/pendulum/natural_dynamics.png
    """
    agent.eval()
    device = next(agent.parameters()).device

    # Diverse starting states; θ=0 = upright (unstable), θ=π = hanging (stable)
    starts = [
        ("near-top ω=0",    np.array([np.cos(0.25),  np.sin(0.25),   0.0], np.float32)),
        ("side (π/2) ω=0",  np.array([np.cos(np.pi/2), np.sin(np.pi/2), 0.0], np.float32)),
        ("hanging ω=+4",    np.array([-1.0, 0.0,  4.0], np.float32)),
        ("¾-up (2 rad) ω=0",np.array([np.cos(2.0),  np.sin(2.0),    0.0], np.float32)),
    ]
    n_starts = len(starts)

    # ── collect real (u=0) and imagined (Az) rollouts ──────────────────────────
    real_states_all  = []
    imag_states_all  = []
    z_norm_real_all  = []
    z_norm_imag_all  = []
    z_latent_err_all = []

    for _, s0 in starts:
        # real env with zero action
        ev = gym.make("Pendulum-v1")
        ev.reset()
        ev.unwrapped.state = np.array([np.arctan2(float(s0[1]), float(s0[0])),
                                       float(s0[2])])
        s = s0.copy()
        real_states = [s.copy()]
        for _ in range(n_steps):
            s, _, term, trunc, _ = ev.step(np.array([0.0], dtype=np.float32))
            real_states.append(s.copy())
            if term or trunc:
                break
        ev.close()
        T = len(real_states) - 1

        # latent free-flight: z_{t+1} = A z_t
        with torch.no_grad():
            real_s_t = torch.tensor(
                np.array(real_states, dtype=np.float32), device=device)   # [T+1, 3]
            z_enc_all = agent.encoder(real_s_t)                            # [T+1, d]
            z_norms_real = z_enc_all.norm(dim=1).cpu().numpy()             # [T+1]

            A = agent.A.detach()
            z = z_enc_all[:1].clone()   # encode s0
            imag_states  = [agent.decoder(z).cpu().numpy()[0].copy()]
            z_norms_imag = [z.norm().item()]
            z_lat_errs   = [0.0]

            for t in range(T):
                z = z @ A.T             # free-flight: u=0
                s_hat = agent.decoder(z).cpu().numpy()[0]
                imag_states.append(s_hat.copy())
                z_norms_imag.append(z.norm().item())
                z_lat_errs.append((z - z_enc_all[t + 1:t + 2]).norm().item())

        real_states_all.append(np.array(real_states,  dtype=np.float32))
        imag_states_all.append(np.array(imag_states,  dtype=np.float32))
        z_norm_real_all.append(np.array(z_norms_real, dtype=np.float32))
        z_norm_imag_all.append(np.array(z_norms_imag, dtype=np.float32))
        z_latent_err_all.append(np.array(z_lat_errs,  dtype=np.float32))

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, n_starts, figsize=(4.5 * n_starts, 15), squeeze=False)
    fig.suptitle(
        "Natural (Uncontrolled, u=0) Dynamics: Real Env vs Latent Free-Flight (z_{t+1}=Az_t)\n"
        "blue = real env   orange = imagined (A z_t decoded)   "
        "A ∈ O(d) ⟹ ‖z‖ preserved by model",
        fontsize=10,
    )

    try:
        vg = _value_grid(agent)
    except Exception:
        vg = None

    for col, (label, _) in enumerate(starts):
        rs  = real_states_all[col]   # [T+1, 3]
        im  = imag_states_all[col]   # [T+1, 3]
        T   = len(rs) - 1
        ts  = np.arange(T + 1)

        real_th = np.arctan2(rs[:, 1], rs[:, 0])
        real_td = rs[:, 2]
        imag_th = np.arctan2(im[:, 1], im[:, 0])
        imag_td = im[:, 2]

        # Row 0 — phase portrait
        ax = axes[0, col]
        if vg is not None:
            _, _, V_grid = vg
            ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                      aspect="auto", origin="lower", cmap="viridis", alpha=0.35)
        ax.plot(real_th, real_td, "o-", color="#4c8bc9", lw=1.5, ms=2.5,
                alpha=0.9, label="real (u=0)")
        ax.plot(imag_th, imag_td, "o-", color="#ff7f0e", lw=1.5, ms=2.5,
                alpha=0.9, label="imag (Az)")
        ax.plot(real_th[0], real_td[0], "gs", ms=8, zorder=6, label="start")
        ax.plot(real_th[-1], real_td[-1], "b^", ms=7, zorder=6)
        ax.plot(imag_th[-1], imag_td[-1], "r^", ms=7, zorder=6)
        ax.axvline(0, color="white", lw=0.8, alpha=0.5)
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels(["-π", "0", "π"], fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("θ"); ax.set_ylabel("θ̇")
        if col == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.2)

        # Row 1 — angle and angular velocity time series
        ax = axes[1, col]
        ax.plot(ts, real_th, color="#4c8bc9", lw=1.5, label="θ real")
        ax.plot(ts, imag_th, color="#ff7f0e", lw=1.5, ls="--", label="θ imagined")
        ax.fill_between(ts, real_th, imag_th, alpha=0.15, color="#d62728")
        ax.set_ylabel("θ (rad)")
        ax.set_xlabel("step")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        if col == 0:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(ts, real_td, color="#2ca02c", lw=1.2, alpha=0.6, label="θ̇ real")
        ax2.plot(ts, imag_td, color="#9467bd", lw=1.2, ls="--", alpha=0.6,
                 label="θ̇ imag")
        ax2.set_ylabel("θ̇ (rad/s)", fontsize=7)
        if col == n_starts - 1:
            ax2.legend(fontsize=6, loc="upper right")

        # Row 2 — decoded state error
        ax = axes[2, col]
        err = np.linalg.norm(rs - im, axis=1)   # [T+1]
        ax.plot(ts, err, color="#d62728", lw=1.5, label="‖ŝ − s‖")
        ax.fill_between(ts, err, alpha=0.2, color="#d62728")
        ax.set_ylabel("‖ŝ − s‖", color="#d62728")
        ax.set_xlabel("step")
        ax3 = ax.twinx()
        ax3.plot(ts, np.cumsum(err) / (ts + 1), color="#8c564b", lw=1.2,
                 ls=":", label="running mean")
        ax3.set_ylabel("running mean", fontsize=7)
        div_steps = np.where(err > 0.5)[0]
        if len(div_steps) > 0:
            ax.axvline(div_steps[0], color="black", lw=1.2, ls="--",
                       label=f"div t={div_steps[0]}")
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Row 3 — latent norms + latent divergence
        ax = axes[3, col]
        z_nr = z_norm_real_all[col]
        z_ni = z_norm_imag_all[col]
        z_le = z_latent_err_all[col]

        ax.plot(ts, z_nr, color="#4c8bc9", lw=1.5, label="‖enc(sₜ)‖  real")
        ax.plot(ts, z_ni, color="#ff7f0e", lw=1.5, ls="--",
                label="‖zₜ‖  imagined")
        ax.set_ylabel("latent norm ‖z‖")
        ax.set_xlabel("step")
        if col == 0:
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)

        ax4 = ax.twinx()
        ax4.plot(ts, z_le, color="#d62728", lw=1.5, alpha=0.85,
                 label="‖z_imag − enc(s)‖")
        ax4.fill_between(ts, z_le, alpha=0.12, color="#d62728")
        ax4.set_ylabel("latent divergence", color="#d62728", fontsize=7)
        lat_div_steps = np.where(z_le > 1.0)[0]
        if len(lat_div_steps) > 0:
            ax4.axvline(lat_div_steps[0], color="#d62728", lw=1.0, ls=":", alpha=0.7)
            ax4.annotate(f"t={lat_div_steps[0]}",
                         xy=(lat_div_steps[0], z_le[lat_div_steps[0]]),
                         fontsize=6, color="#d62728", ha="left")
        if col == n_starts - 1:
            ax4.legend(fontsize=6, loc="lower right")

    fig.tight_layout()
    path = os.path.join(VIZ_DIR, "natural_dynamics.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {path}")


def plot_cem_diagnostics(agent, cfg):
    """
    Four diagnostic figures checking whether the CEM hybrid planner works correctly.

    Figure 1 — cem_score_evolution.png
      Per-CEM-iteration score distributions, best score trajectory, and mu/sigma
      evolution for 3 starting states.  Reveals whether CEM is finding better
      trajectories or whether the J landscape is flat.

    Figure 2 — cem_trajectory_evolution.png
      Best-sample decoded trajectory at each CEM iteration from the hanging state.
      Shows whether the zero-order search is finding committed swing-up actions.

    Figure 3 — cem_open_loop.png
      Planned latent trajectory (decoded) vs actual open-loop execution vs
      closed-loop CEM MPC.  Diagnoses the model-reality gap.

    Figure 4 — cem_reward_landscape.png
      J(u[0]) landscape from hanging with the rest fixed, decomposed into
      disc_path and γᴴQ.  Plus r_net predicted reward vs actual reward scatter.
    """
    agent.eval()
    device       = next(agent.parameters()).device
    gamma        = cfg.algo.gamma
    horizon      = cfg.planner.horizon
    action_scale = cfg.env.action_scale
    reward_scale = cfg.algo.reward_scale
    ci           = cfg.planner.cem_iters
    d            = agent.d
    action_dim   = agent.B.shape[1]

    starts = [
        ("hanging\nω=0",  np.array([-1.0,  0.0,  0.0], np.float32)),
        ("side\nω=+3",    np.array([ 0.0,  1.0,  3.0], np.float32)),
        ("side\nω=−3",    np.array([ 0.0, -1.0, -3.0], np.float32)),
    ]

    # Collect CEM diagnostics for all 3 starting states
    diags = {}
    for label, s0 in starts:
        diags[label] = _run_cem_instrumented(agent, s0, cfg)

    # ── Figure 1: Score distribution + mu/sigma evolution ─────────────────────
    fig1, axes1 = plt.subplots(4, len(starts), figsize=(5 * len(starts), 14), squeeze=False)
    fig1.suptitle(
        f"CEM Internals — Score Evolution, mu/sigma  "
        f"(H={horizon}  {ci}×{cfg.planner.cem_samples} samples  γ={gamma})",
        fontsize=11,
    )

    cmap_iters = plt.cm.viridis(np.linspace(0.15, 0.95, ci))
    iter_x     = np.arange(ci + 1)

    for col, (label, s0) in enumerate(starts):
        dg = diags[label]

        # Row 0 — score histograms, one curve per CEM iteration
        ax = axes1[0, col]
        all_scores = dg["iter_scores"]   # [ci, ns]
        score_min  = all_scores.min()
        score_max  = all_scores.max()
        bins = np.linspace(score_min, score_max, 40)
        for i in range(ci):
            counts, edges = np.histogram(all_scores[i], bins=bins, density=True)
            ax.plot((edges[:-1] + edges[1:]) / 2, counts,
                    color=cmap_iters[i], lw=1.2, alpha=0.85)
        # Colorbar-style legend
        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=plt.Normalize(1, ci))
        sm.set_array([])
        fig1.colorbar(sm, ax=ax, label="CEM iter", fraction=0.046, pad=0.04)
        ax.set_title(label.replace("\n", " "), fontsize=9)
        ax.set_xlabel("J score")
        if col == 0:
            ax.set_ylabel("density")
        ax.grid(alpha=0.3)

        # Row 1 — best/mean score per iteration
        ax = axes1[1, col]
        ax.plot(np.arange(1, ci + 1), dg["iter_best"],
                "o-", color="#2ca02c", lw=2, ms=5, label="best sample")
        ax.plot(np.arange(1, ci + 1), all_scores.mean(axis=1),
                "s--", color="#4c8bc9", lw=1.5, ms=4, label="mean sample")
        ax.axhline(dg["J_cem"],      color="#ff7f0e", ls=":",  lw=1.5,
                   label=f"CEM mean  J={dg['J_cem']:.3f}")
        ax.axhline(dg["J_polished"], color="#d62728", ls="--", lw=1.5,
                   label=f"After grad  J={dg['J_polished']:.3f}")
        ax.set_xlabel("CEM iteration")
        if col == 0:
            ax.set_ylabel("J (value-MPC)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Row 2 — mu[0] (first-step mean action logit → actual torque)
        ax = axes1[2, col]
        mu0_actual = np.tanh(dg["iter_mu0"][:, 0]) * action_scale   # [ci+1]
        ax.plot(iter_x, mu0_actual, "o-", color="#9467bd", lw=2, ms=5)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_ylim(-action_scale * 1.2, action_scale * 1.2)
        ax.set_xlabel("CEM iteration")
        if col == 0:
            ax.set_ylabel("tanh(μ₁) × scale\n(first-step torque)")
        ax.grid(alpha=0.3)

        # Row 3 — sigma[0] (first-step std of action logit)
        ax = axes1[3, col]
        ax.plot(iter_x, dg["iter_sigma0"][:, 0], "o-", color="#e377c2", lw=2, ms=5)
        ax.set_xlabel("CEM iteration")
        if col == 0:
            ax.set_ylabel("σ₁ (first-step logit std)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)

    fig1.tight_layout()
    path1 = os.path.join(VIZ_DIR, "cem_score_evolution.png")
    fig1.savefig(path1, dpi=110, bbox_inches="tight")
    plt.close(fig1)
    print(f"  [viz] {path1}")

    # ── Figure 2: Best-sample decoded trajectory per CEM iteration ─────────────
    hang_label = starts[0][0]
    hang_s0    = starts[0][1]
    dg_hang    = diags[hang_label]

    n_show    = min(ci, 6)
    show_iters = np.round(np.linspace(0, ci - 1, n_show)).astype(int)

    with torch.no_grad():
        B          = agent.B.detach()
        W_toeplitz, _, A_stack = agent.get_toeplitz_cache(horizon, gamma)
        z0_t       = torch.tensor(dg_hang["z0"], dtype=torch.float32, device=device)
        ZIR_hang   = torch.einsum('kij,nj->nki', A_stack[1:], z0_t)  # [1, H, d]

    def _decode_u(u_logits_1d):
        """u_logits_1d: [H] → (thetas, thetadots, actions_actual)"""
        with torch.no_grad():
            u_t = torch.tanh(
                torch.tensor(u_logits_1d[:, np.newaxis], dtype=torch.float32, device=device)
            ).unsqueeze(0)   # [1, H, 1]
            X     = (u_t @ B.T).reshape(1, horizon * d)
            Z     = ZIR_hang + (X @ W_toeplitz.T).reshape(1, horizon, d)
            all_z = torch.cat([z0_t.unsqueeze(1), Z], dim=1).squeeze(0)  # [H+1, d]
            dec   = agent.decoder(all_z).cpu().numpy()
        th  = np.arctan2(dec[:, 1], dec[:, 0])
        td  = dec[:, 2]
        act = (np.tanh(u_logits_1d) * action_scale)
        return th, td, act

    try:
        _vg = _value_grid(agent)
    except Exception:
        _vg = None

    fig2, axes2 = plt.subplots(2, n_show, figsize=(3.5 * n_show, 7), squeeze=False)
    fig2.suptitle(
        f"CEM: Best-Sample Trajectory Evolution from Hanging  "
        f"({ci} CEM iters + {cfg.planner.cem_grad_iters} grad polish)",
        fontsize=10,
    )

    for col, it in enumerate(show_iters):
        u_best = dg_hang["iter_best_u"][it]   # [H]

        th, td, acts = _decode_u(u_best)

        # Phase portrait
        ax = axes2[0, col]
        if _vg is not None:
            _, _, V_grid = _vg
            ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                      aspect="auto", origin="lower", cmap="viridis", alpha=0.4)
        ax.plot(th, td, "o-", lw=1.4, ms=3, color="#ff7f0e")
        ax.plot(th[0],  td[0],  "go", ms=7, zorder=5)
        ax.plot(th[-1], td[-1], "r*", ms=9, zorder=5)
        ax.axvline(0, color="white", lw=0.8, alpha=0.5)
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"], fontsize=7)
        ax.set_title(f"CEM iter {it+1}\nJ={dg_hang['iter_best'][it]:.3f}", fontsize=9)
        if col == 0:
            ax.set_ylabel("θ̇", fontsize=8)

        # Action bars
        ax = axes2[1, col]
        bar_cols = ["#e377c2" if a >= 0 else "#17becf" for a in acts]
        ax.bar(np.arange(horizon), acts, color=bar_cols, alpha=0.85, width=0.7)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-action_scale * 1.2, action_scale * 1.2)
        ax.set_xlabel("step", fontsize=7)
        if col == 0:
            ax.set_ylabel("torque", fontsize=8)

    # Final polished plan
    if n_show > 0:
        th_p, td_p, acts_p = _decode_u(dg_hang["polished_u"][:, 0])
        ax = axes2[0, -1]
        ax.set_title(
            f"After grad polish\nJ={dg_hang['J_polished']:.3f}", fontsize=9)
        # Overlay polished trajectory in a different colour
        if _vg is not None:
            _, _, V_grid = _vg
            ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                      aspect="auto", origin="lower", cmap="viridis", alpha=0.4)
        ax.plot(th_p, td_p, "o-", lw=1.4, ms=3, color="#d62728")
        ax.plot(th_p[0],  td_p[0],  "go", ms=7, zorder=5)
        ax.plot(th_p[-1], td_p[-1], "r*", ms=9, zorder=5)
        ax.axvline(0, color="white", lw=0.8, alpha=0.5)
        ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"], fontsize=7)
        ax = axes2[1, -1]
        bar_cols = ["#e377c2" if a >= 0 else "#17becf" for a in acts_p]
        ax.bar(np.arange(horizon), acts_p, color=bar_cols, alpha=0.85, width=0.7)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-action_scale * 1.2, action_scale * 1.2)
        ax.set_xlabel("step", fontsize=7)

    fig2.tight_layout()
    path2 = os.path.join(VIZ_DIR, "cem_trajectory_evolution.png")
    fig2.savefig(path2, dpi=110, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [viz] {path2}")

    # ── Figure 3: Open-loop plan vs actual (model-reality gap) ─────────────────
    cem_planner = CEMPlannerWarmStart(
        agent,
        horizon       = horizon,
        cem_iters     = cfg.planner.cem_iters,
        n_samples     = cfg.planner.cem_samples,
        n_elites      = cfg.planner.cem_elites,
        grad_iters    = cfg.planner.cem_grad_iters,
        lr            = cfg.planner.lr,
        gamma         = gamma,
        action_scale  = action_scale,
        objective     = "value",
    )

    fig3, axes3 = plt.subplots(3, len(starts), figsize=(5 * len(starts), 12), squeeze=False)
    fig3.suptitle(
        "Model–Reality Gap: CEM Planned (decoded) vs Open-Loop Actual vs Closed-Loop MPC",
        fontsize=11,
    )

    for col, (label, s0) in enumerate(starts):
        dg = diags[label]

        # Decoded plan for this starting state
        with torch.no_grad():
            z0_c = torch.tensor(dg["z0"], dtype=torch.float32, device=device)
            ZIR_c = torch.einsum('kij,nj->nki', A_stack[1:], z0_c)
            u_t   = torch.tanh(
                torch.tensor(dg["polished_u"], dtype=torch.float32, device=device)
            ).unsqueeze(0)  # [1, H, action_dim]
            X_c   = (u_t @ B.T).reshape(1, horizon * d)
            Z_c   = ZIR_c + (X_c @ W_toeplitz.T).reshape(1, horizon, d)
            all_z = torch.cat([z0_c.unsqueeze(1), Z_c], dim=1).squeeze(0)
            dec   = agent.decoder(all_z).cpu().numpy()
        th_plan  = np.arctan2(dec[:, 1], dec[:, 0])
        td_plan  = dec[:, 2]
        acts_plan = (np.tanh(dg["polished_u"][:, 0]) * action_scale)

        # Open-loop actual execution
        states_ol, rewards_ol = _execute_open_loop(s0, acts_plan[:, np.newaxis]
                                                   if acts_plan.ndim == 1 else acts_plan)
        th_ol, td_ol = _to_theta_thetadot(states_ol)

        # Closed-loop MPC
        cem_planner.reset()
        ev_cl = gym.make("Pendulum-v1")
        ev_cl.reset()
        theta_s0 = np.arctan2(float(s0[1]), float(s0[0]))
        ev_cl.unwrapped.state = np.array([theta_s0, float(s0[2])])
        s_cl = s0.copy()
        states_cl, rets_cl = [s_cl.copy()], 0.0
        for _ in range(horizon * 4):   # run 4 planning horizons
            a_cl = cem_planner(s_cl)
            s_cl, r_cl, term, trunc, _ = ev_cl.step(np.atleast_1d(a_cl).astype(np.float32))
            states_cl.append(s_cl.copy()); rets_cl += r_cl
            if term or trunc:
                break
        ev_cl.close()
        states_cl = np.array(states_cl, dtype=np.float32)
        th_cl, td_cl = _to_theta_thetadot(states_cl)

        # Predicted r_net rewards along open-loop trajectory
        with torch.no_grad():
            s_ol_t = torch.tensor(states_ol[:-1], dtype=torch.float32, device=device)
            z_ol   = agent.encoder(s_ol_t)
            a_ol_t = torch.tensor(acts_plan[:len(rewards_ol), np.newaxis],
                                  dtype=torch.float32, device=device)
            r_pred = agent.r_net(torch.cat([z_ol, a_ol_t], dim=-1)).squeeze(-1).cpu().numpy()
        r_actual_scaled = np.array(rewards_ol) / reward_scale

        # Phase portrait: planned (decoded)
        for row, (th, td, col_h, lbl) in enumerate([
            (th_plan, td_plan, "#ff7f0e", f"planned (decoded)  H={horizon}"),
            (th_ol[:horizon+1], td_ol[:horizon+1], "#2ca02c",
             f"open-loop actual  ret={rewards_ol.sum():.0f}"),
            (th_cl, td_cl, "#d62728",
             f"closed-loop MPC  ret={rets_cl:.0f}"),
        ]):
            ax = axes3[row, col]
            if _vg is not None:
                _, _, V_grid = _vg
                ax.imshow(V_grid, extent=[-np.pi, np.pi, -8, 8],
                          aspect="auto", origin="lower", cmap="viridis", alpha=0.35)
            ax.plot(th, td, "o-", lw=1.5, ms=3, color=col_h)
            ax.plot(th[0],  td[0],  "wo", ms=7, zorder=5)
            ax.plot(th[-1], td[-1], "w*", ms=9, zorder=5)
            ax.axvline(0, color="white", lw=0.8, alpha=0.5)
            ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-8, 8)
            ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels(["-π", "0", "π"], fontsize=7)
            ax.set_xlabel("θ"); ax.set_ylabel("θ̇")
            if row == 0:
                ax.set_title(label.replace("\n", " "), fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{lbl}\nθ̇", fontsize=8)
            ax.grid(alpha=0.2)

        # Annotate open-loop row with r_net vs actual reward
        # (r_net prediction quality is the primary open-loop diagnostic)
        T = len(rewards_ol)
        axes3[1, col].set_title(
            f"open-loop actual  ret={rewards_ol.sum():.0f}\n"
            f"r_net MAE={(np.abs(r_pred[:T] - r_actual_scaled[:T])).mean():.3f}",
            fontsize=8,
        )

    fig3.tight_layout()
    path3 = os.path.join(VIZ_DIR, "cem_open_loop.png")
    fig3.savefig(path3, dpi=110, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [viz] {path3}")

    # ── Figure 4: J landscape + r_net accuracy ──────────────────────────────────
    fig4, axes4 = plt.subplots(2, 2, figsize=(12, 9))
    fig4.suptitle("CEM: J Landscape & r_net / Q Accuracy", fontsize=11)

    # [0,0] J(u[0]) landscape — sweep first-action logit, fix rest to 0
    ax = axes4[0, 0]
    u0_sweep = np.linspace(-3.0, 3.0, 80, dtype=np.float32)   # logit space
    for (lbl, s0), col_h in zip(starts, ["#ff7f0e", "#2ca02c", "#d62728"]):
        with torch.no_grad():
            z0_sw = agent.encoder(
                torch.tensor(s0[np.newaxis], dtype=torch.float32, device=device))
            ZIR_sw = torch.einsum('kij,nj->nki', A_stack[1:], z0_sw)
            gp     = gamma ** torch.arange(horizon, device=device, dtype=torch.float32)
            J_vals = []
            for u0_val in u0_sweep:
                u_lgt = torch.zeros(1, horizon, action_dim, device=device)
                u_lgt[0, 0, 0] = float(u0_val)
                u_act = torch.tanh(u_lgt)
                X   = (u_act @ B.T).reshape(1, horizon * d)
                Z   = ZIR_sw + (X @ W_toeplitz.T).reshape(1, horizon, d)
                Zc  = torch.cat([z0_sw.unsqueeze(1), Z[:, :-1, :]], dim=1)
                ZU  = torch.cat([Zc, u_act], dim=-1)
                disc = (gp.unsqueeze(0) * agent.r_net(ZU).squeeze(-1)).sum(dim=1)
                z_H  = Z[:, -1, :]
                a_H  = agent.pi_net(z_H)
                q_H  = agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)
                J_vals.append((disc + gamma ** horizon * q_H).item())
        ax.plot(np.tanh(u0_sweep) * action_scale, J_vals,
                color=col_h, lw=2, label=lbl.replace("\n", " "))
    ax.set_xlabel("u[0] actual torque (fixed rest=0)")
    ax.set_ylabel("J = Σγᵗr_net + γᴴQ")
    ax.set_title("J landscape vs first action (rest zeros)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # [0,1] Decompose J: disc_path vs gamma^H*Q for hanging sweep
    ax = axes4[0, 1]
    with torch.no_grad():
        z0_h = agent.encoder(
            torch.tensor(starts[0][1][np.newaxis], dtype=torch.float32, device=device))
        ZIR_h = torch.einsum('kij,nj->nki', A_stack[1:], z0_h)
        path_vals, q_vals = [], []
        for u0_val in u0_sweep:
            u_lgt = torch.zeros(1, horizon, action_dim, device=device)
            u_lgt[0, 0, 0] = float(u0_val)
            u_act = torch.tanh(u_lgt)
            X   = (u_act @ B.T).reshape(1, horizon * d)
            Z   = ZIR_h + (X @ W_toeplitz.T).reshape(1, horizon, d)
            Zc  = torch.cat([z0_h.unsqueeze(1), Z[:, :-1, :]], dim=1)
            ZU  = torch.cat([Zc, u_act], dim=-1)
            disc = (gp.unsqueeze(0) * agent.r_net(ZU).squeeze(-1)).sum(dim=1).item()
            z_H  = Z[:, -1, :]
            a_H  = agent.pi_net(z_H)
            q_H  = (gamma ** horizon * agent.q_net(torch.cat([z_H, a_H], -1)).squeeze(-1)).item()
            path_vals.append(disc)
            q_vals.append(q_H)
    u0_actual = np.tanh(u0_sweep) * action_scale
    ax.plot(u0_actual, path_vals, color="#2ca02c", lw=2, label="Σγᵗ r_net(z_t, u_t)")
    ax.plot(u0_actual, q_vals,    color="#9467bd", lw=2, label=f"γᴴ Q(z_H, π(z_H))")
    ax.plot(u0_actual, np.add(path_vals, q_vals), color="#ff7f0e", lw=2.5, ls="--", label="J total")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("u[0] actual torque"); ax.set_ylabel("value")
    ax.set_title("J decomposed (hanging state)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # [1,0] r_net predicted vs actual reward on random transitions
    ax = axes4[1, 0]
    ev_rnet = gym.make("Pendulum-v1")
    r_preds_all, r_actual_all = [], []
    for _ in range(8):   # 8 random episodes
        s_rnet, _ = ev_rnet.reset()
        for _ in range(25):
            a_rnet = ev_rnet.action_space.sample()
            s_next, r_true, term, trunc, _ = ev_rnet.step(a_rnet)
            with torch.no_grad():
                z_rnet = agent.encoder(
                    torch.tensor(s_rnet[np.newaxis], dtype=torch.float32, device=device))
                a_t    = torch.tensor(a_rnet[np.newaxis], dtype=torch.float32, device=device)
                r_hat  = agent.r_net(torch.cat([z_rnet, a_t], dim=-1)).item()
            r_preds_all.append(r_hat)
            r_actual_all.append(r_true / reward_scale)
            s_rnet = s_next
            if term or trunc:
                break
    ev_rnet.close()
    r_preds_all  = np.array(r_preds_all)
    r_actual_all = np.array(r_actual_all)
    corr = np.corrcoef(r_actual_all, r_preds_all)[0, 1]
    ax.scatter(r_actual_all, r_preds_all, s=12, alpha=0.5, color="#4c8bc9")
    lo = min(r_actual_all.min(), r_preds_all.min())
    hi = max(r_actual_all.max(), r_preds_all.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="ideal")
    ax.set_xlabel("actual r / reward_scale")
    ax.set_ylabel("r_net prediction")
    ax.set_title(f"r_net accuracy  (Pearson r={corr:.3f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # [1,1] Q vs actual MC return from diverse starting states
    ax = axes4[1, 1]
    mc_starts = [
        np.array([-1.0,  0.0,  0.0], np.float32),   # hang
        np.array([ 0.0,  1.0,  3.0], np.float32),   # side+
        np.array([ 0.0, -1.0, -3.0], np.float32),   # side-
        np.array([np.cos(0.4), np.sin(0.4), 0.0], np.float32),  # near top
        np.array([-1.0,  0.0,  2.0], np.float32),   # hang+vel
    ]
    for mc_s0 in mc_starts:
        # Q prediction
        with torch.no_grad():
            z_mc = agent.encoder(
                torch.tensor(mc_s0[np.newaxis], dtype=torch.float32, device=device))
            a_mc = agent.pi_net(z_mc)
            q_mc = agent.q_net(torch.cat([z_mc, a_mc], -1)).item()
        # Monte-Carlo return from this state via policy
        ev_mc = gym.make("Pendulum-v1")
        ev_mc.reset()
        th_mc = np.arctan2(float(mc_s0[1]), float(mc_s0[0]))
        ev_mc.unwrapped.state = np.array([th_mc, float(mc_s0[2])])
        s_mc = mc_s0.copy()
        mc_ret, disc_factor = 0.0, 1.0
        for _ in range(200):
            a_mc_env = agent.act_policy_continuous_batch(
                s_mc[np.newaxis], action_scale=action_scale)[0]
            s_mc, r_mc, term, trunc, _ = ev_mc.step(a_mc_env.astype(np.float32))
            mc_ret     += disc_factor * r_mc / reward_scale
            disc_factor *= gamma
            if term or trunc:
                break
        ev_mc.close()
        ax.scatter([mc_ret], [q_mc], s=60, zorder=5)
        ax.annotate(
            f"θ={np.arctan2(mc_s0[1],mc_s0[0]):.1f}",
            (mc_ret, q_mc), fontsize=6, ha="left", va="bottom")
    lo_q = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi_q = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo_q, hi_q], [lo_q, hi_q], "k--", lw=1.2, label="ideal")
    ax.set_xlabel("MC return (γ-discounted, / reward_scale)")
    ax.set_ylabel("Q(z, π(z))")
    ax.set_title("Q accuracy vs MC return")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig4.tight_layout()
    path4 = os.path.join(VIZ_DIR, "cem_reward_landscape.png")
    fig4.savefig(path4, dpi=110, bbox_inches="tight")
    plt.close(fig4)
    print(f"  [viz] {path4}")


def plot_plan_convergence(agent, cfg):
    """
    2-row × 3-col grid:
    - Top row:    value-MPC objective J vs Adam iteration (one col per starting state)
    - Bottom row: value-free Σ_t ‖z_t − z_goal‖² vs Adam iteration (same cols)

    Shows how quickly each planner converges and whether they plateau, oscillate,
    or diverge from their respective basins.
    """
    agent.eval()
    device = next(agent.parameters()).device
    horizon    = cfg.planner.horizon
    plan_iters = cfg.planner.plan_iters
    gamma      = cfg.algo.gamma

    with torch.no_grad():
        z_goal = agent.encoder(
            torch.tensor(PENDULUM_GOAL[np.newaxis], dtype=torch.float32, device=device)
        )  # [1, d]

    starts = [
        ("hanging  ω=0",  np.array([-1.0,  0.0,  0.0], np.float32)),
        ("side  ω=+3",    np.array([ 0.0,  1.0,  3.0], np.float32)),
        ("side  ω=−3",    np.array([ 0.0, -1.0, -3.0], np.float32)),
    ]
    colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    iters  = np.arange(plan_iters + 1)

    cem_iters   = cfg.planner.cem_iters
    cem_samples = cfg.planner.cem_samples
    cem_elites  = cfg.planner.cem_elites

    fig, axes = plt.subplots(2, len(starts), figsize=(5 * len(starts), 8))
    fig.suptitle(
        f"Plan Convergence — Value-MPC (grad) vs Value-Free (grad) vs CEM warm-start  "
        f"(H={horizon}  iters={plan_iters}  γ={gamma})",
        fontsize=11,
    )

    for i, (label, s0) in enumerate(starts):
        with torch.no_grad():
            z0 = agent.encoder(
                torch.tensor(s0[np.newaxis], dtype=torch.float32, device=device)
            )

        try:
            val_curve, vfree_curve, _, _ = _plan_objectives(
                agent, z0, z_goal, horizon, plan_iters, gamma)
            cem_score, cem_mu = _cem_best_score(
                agent, z0, horizon, cem_iters, cem_samples, cem_elites, gamma)
            cem_grad_curve = _cem_grad_curve(
                agent, z0, cem_mu, horizon, plan_iters, gamma)
        except Exception as e:
            for row in range(2):
                axes[row, i].text(0.5, 0.5, str(e), ha="center", va="center",
                                  transform=axes[row, i].transAxes)
            continue

        ax = axes[0, i]
        ax.plot(iters, val_curve,      color=colors[i], lw=2,   label="grad (biased-init)")
        ax.plot(iters, cem_grad_curve, color="#9467bd",  lw=2, ls="-.",
                label=f"CEM→grad  J₀={cem_score:.3f}")
        ax.axhline(cem_score, color="#9467bd", lw=1, ls=":",
                   alpha=0.5, label="_nolegend_")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Adam iteration")
        if i == 0:
            ax.set_ylabel("J = Σγᵗr_net + γᴴQ  [value-MPC]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[1, i]
        ax.plot(iters, vfree_curve, color=colors[i], lw=2, ls="--")
        ax.set_xlabel("Adam iteration")
        if i == 0:
            ax.set_ylabel("Σ_t ‖z_t − z_goal‖²  [value-free]")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(VIZ_DIR, "plan_convergence.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {path}")


def _default_pendulum_cfg():
    """Minimal cfg for standalone use of viz functions (e.g., loading a checkpoint)."""
    return Config(
        env=EnvConfig(state_dim=3, n_actions=ACTION_DIM, action_scale=ACTION_SCALE),
        planner=PlannerConfig(horizon=PLAN_HORIZON, plan_iters=PLAN_ITERS),
    )


# ---------------------------------------------------------------------------
# Koopman latent space analysis
# ---------------------------------------------------------------------------

def plot_koopman_latent_analysis(agent, buf, cfg):
    """
    Four-panel Koopman latent space analysis.

    Layout (2×4 GridSpec):
      [0, 0:2] PCA trajectory + Q(z,0) contour  — the 'killer plot'
      [0, 2]   Koopman flow field (Az − z) in PC1-PC2 space
      [0, 3]   Eigenvalues of A on the complex plane
      [1, 0-3] Latent dimension heatmaps z_0 … z_3 over (θ, θ̇)
    """
    agent.eval()
    device     = next(agent.parameters()).device
    d          = agent.d
    action_dim = agent.B.shape[1]
    action_scale = cfg.env.action_scale

    # ── Encode buffer samples for PCA ────────────────────────────────────
    n = min(buf.size, 4096)
    idx = np.random.choice(buf.size, n, replace=False)
    s_np = buf.states[idx]
    with torch.no_grad():
        Z = agent.encoder(
            torch.tensor(s_np, dtype=torch.float32, device=device)
        ).cpu().numpy()                                          # [N, d]

    # PCA via SVD on centered Z — preserves linear geometry exactly
    Z_mean = Z.mean(0)
    Zc = Z - Z_mean
    _, S_svd, Vt = np.linalg.svd(Zc, full_matrices=False)      # Vt: [d, d]
    explained  = S_svd ** 2 / (S_svd ** 2).sum()
    PC         = Vt[:2]                                          # [2, d]
    Z_2d       = Zc @ PC.T                                       # [N, 2]

    # ── Collect a real policy episode for trajectory overlay ─────────────
    ep_states = []
    ev = gym.make("Pendulum-v1")
    ev.reset()
    ev.unwrapped.state = np.array([-np.pi, 0.0])
    s_ep = np.array([np.cos(-np.pi), np.sin(-np.pi), 0.0], np.float32)
    for _ in range(200):
        ep_states.append(s_ep.copy())
        with torch.no_grad():
            a = agent.act_policy_continuous_batch(
                s_ep[np.newaxis], action_scale)[0]
        s_ep, _, term, trunc, _ = ev.step(np.atleast_1d(a).astype(np.float32))
        if term or trunc:
            break
    ev.close()
    ep_states = np.array(ep_states, dtype=np.float32)
    with torch.no_grad():
        Z_ep = agent.encoder(
            torch.tensor(ep_states, dtype=torch.float32, device=device)
        ).cpu().numpy()
    Z_ep_2d = (Z_ep - Z_mean) @ PC.T
    ep_theta = np.arctan2(ep_states[:, 1], ep_states[:, 0])

    # ── PC1-PC2 grid for contour + vector field ───────────────────────────
    n_grid  = 40
    pc1_lim = float(np.percentile(np.abs(Z_2d[:, 0]), 95)) * 1.3
    pc2_lim = float(np.percentile(np.abs(Z_2d[:, 1]), 95)) * 1.3
    g1 = np.linspace(-pc1_lim, pc1_lim, n_grid)
    g2 = np.linspace(-pc2_lim, pc2_lim, n_grid)
    GG1, GG2 = np.meshgrid(g1, g2)
    G_2d  = np.stack([GG1.ravel(), GG2.ravel()], axis=1)        # [M, 2]
    G_full = (G_2d @ PC + Z_mean).astype(np.float32)             # [M, d]

    # Q(z, 0) contour
    with torch.no_grad():
        G_t    = torch.tensor(G_full, device=device)
        a_zero = torch.zeros(len(G_t), action_dim, device=device)
        q_grid = agent.q_net(torch.cat([G_t, a_zero], -1)).squeeze(-1).cpu().numpy()
    Q_map = q_grid.reshape(n_grid, n_grid)

    # Unforced Koopman flow: Δz = Az − z, projected to PC space
    A_np       = agent.A.detach().cpu().numpy()                  # [d, d]
    delta_full = G_full @ A_np.T - G_full                        # [M, d]
    delta_2d   = delta_full @ PC.T                               # [M, 2]
    U_q = delta_2d[:, 0].reshape(n_grid, n_grid)
    V_q = delta_2d[:, 1].reshape(n_grid, n_grid)

    # ── Eigenvalues of A ──────────────────────────────────────────────────
    with torch.no_grad():
        eigs   = torch.linalg.eig(agent.A).eigenvalues.cpu()
    eig_re = eigs.real.numpy()
    eig_im = eigs.imag.numpy()

    # ── Latent dimension heatmaps over (θ, θ̇) ────────────────────────────
    n_th, n_td = 60, 50
    th_h  = np.linspace(-np.pi, np.pi, n_th, dtype=np.float32)
    td_h  = np.linspace(-8.0,   8.0,   n_td, dtype=np.float32)
    grid_s = np.stack([
        np.tile(np.cos(th_h), n_td),
        np.tile(np.sin(th_h), n_td),
        np.repeat(td_h, n_th),
    ], axis=1).astype(np.float32)
    with torch.no_grad():
        z_heat = agent.encoder(
            torch.tensor(grid_s, device=device)
        ).cpu().numpy()                                          # [n_td*n_th, d]

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle(
        f"Koopman Latent Space Analysis  (d={d}  PC1={explained[0]*100:.1f}%"
        f"  PC2={explained[1]*100:.1f}%)",
        fontsize=13,
    )

    # [0, 0:2] PCA trajectory + Q(z,0) contour
    ax = fig.add_subplot(gs[0, :2])
    cf = ax.contourf(GG1, GG2, Q_map, levels=25, cmap="plasma", alpha=0.75)
    plt.colorbar(cf, ax=ax, label="Q(z, 0)", fraction=0.046, pad=0.04)
    sc = ax.scatter(Z_ep_2d[:, 0], Z_ep_2d[:, 1],
                    c=ep_theta, cmap="hsv", s=14, zorder=3, alpha=0.9,
                    vmin=-np.pi, vmax=np.pi)
    plt.colorbar(sc, ax=ax, label="θ (rad)", fraction=0.046, pad=0.04)
    ax.plot(Z_ep_2d[0,  0], Z_ep_2d[0,  1], "go", ms=9, zorder=4, label="start (hang)")
    ax.plot(Z_ep_2d[-1, 0], Z_ep_2d[-1, 1], "r*", ms=11, zorder=4, label="end")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title("Policy Episode Trajectory + Q(z, 0) Contour")
    ax.legend(fontsize=8)

    # [0, 2] Koopman flow field
    ax2 = fig.add_subplot(gs[0, 2])
    speed = np.sqrt(U_q**2 + V_q**2) + 1e-9
    try:
        ax2.streamplot(g1, g2, U_q, V_q, color=np.log1p(speed),
                       cmap="cool", linewidth=0.9, density=1.3, arrowsize=0.9)
    except Exception:
        ax2.quiver(GG1[::4, ::4], GG2[::4, ::4],
                   U_q[::4, ::4], V_q[::4, ::4], alpha=0.7)
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")
    ax2.set_title("Koopman Flow  Az − z  (unforced)")
    ax2.set_xlim(-pc1_lim, pc1_lim); ax2.set_ylim(-pc2_lim, pc2_lim)

    # [0, 3] Eigenvalues of A
    ax3 = fig.add_subplot(gs[0, 3])
    circ = np.linspace(0, 2 * np.pi, 300)
    ax3.plot(np.cos(circ), np.sin(circ), "k--", lw=0.8, alpha=0.4, label="unit circle")
    sc3 = ax3.scatter(eig_re, eig_im, c=np.abs(eig_im),
                      cmap="viridis", s=35, zorder=3)
    plt.colorbar(sc3, ax=ax3, label="|Im(λ)|", fraction=0.046, pad=0.04)
    ax3.set_xlabel("Re(λ)"); ax3.set_ylabel("Im(λ)")
    ax3.set_title(f"Eigenvalues of A\n‖AᵀA−I‖²={agent.ortho_error():.1e}")
    ax3.set_aspect("equal"); ax3.grid(alpha=0.3); ax3.legend(fontsize=8)

    # [1, 0-3] Latent dimension heatmaps z_0 … z_3
    for i in range(4):
        ax_h = fig.add_subplot(gs[1, i])
        heat = z_heat[:, i].reshape(n_td, n_th)
        vmax = np.abs(heat).max()
        im = ax_h.imshow(heat, extent=[-np.pi, np.pi, -8, 8],
                         aspect="auto", origin="lower",
                         cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
        ax_h.set_title(f"z_{i}  (latent dim {i})", fontsize=9)
        ax_h.set_xlabel("θ (rad)"); ax_h.set_ylabel("θ̇")
        ax_h.set_xticks([-np.pi, 0, np.pi])
        ax_h.set_xticklabels(["-π", "0", "π"])

    path = os.path.join(VIZ_DIR, "koopman_latent_analysis.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential planner for data collection (default: Toeplitz GEMM)")
    parser.add_argument("--frozen_b", action="store_true",
                        help="Detach B in sequential planner")
    parser.add_argument("--ou_noise", action="store_true",
                        help="Use Ornstein-Uhlenbeck noise instead of i.i.d. Gaussian")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=70_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg      = make_pendulum_cfg(args)
    live_png = os.path.join(VIZ_DIR, f"pendulum_live_{cfg.run_name}.png")

    def _make_vfree_fn(agent):
        """Factory: builds the value-free eval fn once the trained agent is available."""
        dev = next(agent.parameters()).device
        with torch.no_grad():
            z_goal = agent.encoder(
                torch.tensor(PENDULUM_GOAL[np.newaxis], dtype=torch.float32, device=dev)
            )
        return lambda ss: _value_free_act_batch(
            agent, ss, z_goal,
            cfg.planner.horizon, cfg.planner.plan_iters, cfg.algo.gamma, ACTION_SCALE)

    def _make_cem_value_fn(agent):
        """Factory: CEM value-based with receding-horizon warm-start."""
        return CEMPlannerWarmStart(
            agent,
            horizon=cfg.planner.horizon,
            cem_iters=cfg.planner.cem_iters,
            n_samples=cfg.planner.cem_samples,
            n_elites=cfg.planner.cem_elites,
            grad_iters=cfg.planner.cem_grad_iters,
            lr=cfg.planner.lr,
            gamma=cfg.algo.gamma,
            action_scale=ACTION_SCALE,
            objective="value",
        )

    def _make_cem_vfree_fn(agent):
        """Factory: CEM value-free with receding-horizon warm-start."""
        dev = next(agent.parameters()).device
        with torch.no_grad():
            z_goal = agent.encoder(
                torch.tensor(PENDULUM_GOAL[np.newaxis], dtype=torch.float32, device=dev)
            )
        return CEMPlannerWarmStart(
            agent,
            horizon=cfg.planner.horizon,
            cem_iters=cfg.planner.cem_iters,
            n_samples=cfg.planner.cem_samples,
            n_elites=cfg.planner.cem_elites,
            grad_iters=cfg.planner.cem_grad_iters,
            lr=cfg.planner.lr,
            gamma=cfg.algo.gamma,
            action_scale=ACTION_SCALE,
            objective="value_free",
            z_goal=z_goal,
        )

    def _make_toeplitz_warm_fn(agent):
        """Factory: Toeplitz MPC with receding-horizon warm-start."""
        return ToeplitzPlannerWarmStart(
            agent,
            horizon=cfg.planner.horizon,
            plan_iters=cfg.planner.plan_iters,
            lr=cfg.planner.lr,
            gamma=cfg.algo.gamma,
            action_scale=ACTION_SCALE,
            objective="value",
        )

    result = train_continuous(
        cfg, "Pendulum-v1", device,
        on_viz=lambda **kw: _save_live_plot(**kw, path=live_png),
        extra_eval_fns=[
            ("grad value-free      ", _make_vfree_fn),
            ("toeplitz warm-start  ", _make_toeplitz_warm_fn),
            ("CEM value-based      ", _make_cem_value_fn),
            ("CEM value-free       ", _make_cem_vfree_fn),
        ],
    )
    agent = result["agent"]
    buf   = result["buf"]
    plot_final_summary(agent, result["episode_returns"], buf, cfg=cfg)
    plot_plan_evolution(agent, cfg)
    plot_plan_convergence(agent, cfg)
    plot_model_rollout_accuracy(agent, cfg)
    plot_cem_diagnostics(agent, cfg)
    plot_koopman_latent_analysis(agent, buf, cfg)
    plot_policy_vs_planner(agent, cfg)
