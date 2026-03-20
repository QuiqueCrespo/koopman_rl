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

# ---------------------------------------------------------------------------
# Pendulum-specific constants
# ---------------------------------------------------------------------------
ACTION_DIM       = 1
ACTION_SCALE     = 2.0
PLAN_HORIZON     = 5
PLAN_ITERS       = 15
VIZ_PLAN_HORIZON = 20   # longer horizon for viz plan rollouts (offline)
VIZ_PLAN_ITERS   = 50   # more iters for viz (no speed pressure)
VIZ_DIR          = "output/viz/pendulum"

os.makedirs(VIZ_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def make_pendulum_cfg(args) -> Config:
    """Build Config from parsed argparse args."""
    sequential = args.sequential
    frozen_b   = args.frozen_b
    cumulative = args.cumulative
    ou_noise   = args.ou_noise
    seed       = args.seed

    planner_tag = "seq" if sequential else "toe"
    if sequential and frozen_b:
        planner_tag = "seq_frozb"
    if not sequential and cumulative:
        planner_tag = "toe_cumul"
    noise_tag = "_ou" if ou_noise else ""
    run_name  = f"pendulum_{planner_tag}{noise_tag}_s{seed}"

    return Config(
        env=EnvConfig(
            state_dim=3,
            n_actions=ACTION_DIM,
            action_scale=ACTION_SCALE,
            continuous=True,
        ),
        model=ModelConfig(
            d=32,
            lr=3e-4,
            ema_tau=0.005,
            ortho_a=True,
        ),
        buffer=BufferConfig(
            capacity=100_000,
            batch_size=256,
        ),
        algo=AlgoConfig(
            gamma=0.99,
            lambda_koop=1.0,
            lambda_recon=1.0,
            lambda_v=1.0,
            koop_lr_scale=1.0,
            reward_scale=10.0,
            n_envs=10,
        ),
        train=TrainConfig(
            n_steps=args.steps,
            warmup=10_000,
            noise_start=1.0,
            noise_end=0.1,
            noise_decay=15_000,
            viz_every=5_000,
            viz_dir=VIZ_DIR,
            ckpt_dir="output/checkpoints/pendulum",
            planner_type="sequential" if sequential else "toeplitz",
            cumulative=cumulative,
            frozen_b=frozen_b,
            ou_noise=ou_noise,
        ),
        planner=PlannerConfig(
            horizon=PLAN_HORIZON,
            plan_iters=PLAN_ITERS,
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
    Evaluate V_ψ(enc(s)) over a regular (θ, θ̇) grid.
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
        a = agent.act_plan_continuous(s, horizon=cfg.planner.horizon,
                                      plan_iters=cfg.planner.plan_iters,
                                      action_scale=cfg.env.action_scale)
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
        a = agent.act_plan_continuous(s, horizon=cfg.planner.horizon,
                                      plan_iters=cfg.planner.plan_iters,
                                      action_scale=cfg.env.action_scale)
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

def _save_live_plot(episode_returns, koop_log, v_log, step, ortho_err,
                    agent=None, buf=None, path="pendulum_toeplitz_live.png"):
    """
    2×3 live dashboard saved to a fixed path every viz_every steps.
    Called via the on_viz callback from train_continuous.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Pendulum Koopman (ortho_a=True, SVD) — step {step:,}"
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
    if v_log:
        ax.semilogy(v_log,    color="#17becf", lw=1.5, label="L_v")
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


def _default_pendulum_cfg():
    """Minimal cfg for standalone use of viz functions (e.g., loading a checkpoint)."""
    return Config(
        env=EnvConfig(state_dim=3, n_actions=ACTION_DIM, action_scale=ACTION_SCALE),
        planner=PlannerConfig(horizon=PLAN_HORIZON, plan_iters=PLAN_ITERS),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cumulative", action="store_true",
                        help="Use cumulative value objective during training")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential planner for data collection (default: Toeplitz GEMM)")
    parser.add_argument("--frozen_b", action="store_true",
                        help="Detach B in sequential planner")
    parser.add_argument("--ou_noise", action="store_true",
                        help="Use Ornstein-Uhlenbeck noise instead of i.i.d. Gaussian")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    cfg      = make_pendulum_cfg(args)
    live_png = os.path.join(VIZ_DIR, f"pendulum_live_{cfg.run_name}.png")

    result = train_continuous(
        cfg, "Pendulum-v1", device,
        on_viz=lambda **kw: _save_live_plot(**kw, path=live_png),
    )
    plot_final_summary(result["agent"], result["episode_returns"], result["buf"], cfg=cfg)
