"""
Visualisation utilities for the gravity basin Koopman-RL agent.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from env import (GravityBasin, N_ACTIONS, GOAL_X, GOAL_Y,
                 MAX_EP_STEPS, ACTION_NAMES, ACTION_COLORS, DELTA)


def _goal_patch(alpha: float = 0.4, **kw) -> patches.Rectangle:
    return patches.Rectangle(
        (GOAL_X, GOAL_Y), 1 - GOAL_X, 1 - GOAL_Y,
        color="lime", alpha=alpha, **kw,
    )


@torch.no_grad()
def _value_grid(agent, res: int = 80):
    xs = np.linspace(-1, 1, res)
    ys = np.linspace(-1, 1, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)
    V   = agent.v_net(agent.encode(torch.from_numpy(pts)))
    return XX, YY, V.numpy().reshape(res, res)


@torch.no_grad()
def _policy_grid(agent, res: int = 16):
    xs = np.linspace(-1, 1, res)
    ys = np.linspace(-1, 1, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)
    z   = agent.encode(torch.from_numpy(pts))
    A   = torch.stack([agent.v_net(z @ K.T) for K in agent.K], dim=1).argmax(dim=-1)
    return XX, YY, A.numpy().reshape(res, res)


def plot_results(history: dict) -> None:
    agent       = history["agent"]
    koop_losses = history["koop_losses"]
    q_losses    = history["q_losses"]
    ep_returns  = history["episode_returns"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def smooth(x, w=10):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    # ----------------------------------------------------------------
    # 1. Training losses (log scale, smoothed)
    # ----------------------------------------------------------------
    ax = axes[0, 0]
    ax.semilogy(smooth(koop_losses),
                label=r"$\mathcal{L}_{Koop}$  $\|K_a z_t - z_{t+1}\|^2$",
                color="steelblue")
    ax.semilogy(smooth(q_losses),
                label=r"$\mathcal{L}_V$  $\|V_\psi - V_{\rm target}\|^2$",
                color="tomato")
    ax.set_xlabel("Log interval")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # 2. Episode returns (rolling mean)
    # ----------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(ep_returns, alpha=0.25, color="gray", linewidth=0.6, label="Raw return")
    w2 = min(50, len(ep_returns))
    if len(ep_returns) >= w2:
        roll = np.convolve(ep_returns, np.ones(w2) / w2, mode="valid")
        ax.plot(np.arange(w2 - 1, len(ep_returns)), roll,
                color="royalblue", linewidth=1.5, label=f"Rolling mean ({w2} ep)")
    ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Episode Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # 3. Value map V(x,y)
    # ----------------------------------------------------------------
    ax = axes[0, 2]
    XX, YY, V = _value_grid(agent, res=100)
    im = ax.pcolormesh(XX, YY, V, cmap="plasma", shading="auto")
    plt.colorbar(im, ax=ax, label=r"$V_\psi(f_\theta(s))$")
    ax.add_patch(_goal_patch(label="Goal zone"))
    ax.set_title(r"Learned Value Map $V(x,y)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")

    # ----------------------------------------------------------------
    # 4. Policy map (greedy action arrows over value heatmap)
    # ----------------------------------------------------------------
    ax = axes[1, 0]
    XX_v, YY_v, V_v = _value_grid(agent, res=32)
    ax.pcolormesh(XX_v, YY_v, V_v, cmap="plasma", shading="auto", alpha=0.6)
    XX_p, YY_p, A = _policy_grid(agent, res=20)
    step_x   = XX_p[0, 1] - XX_p[0, 0]
    ARROW_DX = DELTA[:, 0] * step_x * 3
    ARROW_DY = DELTA[:, 1] * step_x * 3
    ax.quiver(XX_p.ravel(), YY_p.ravel(),
              ARROW_DX[A.ravel()], ARROW_DY[A.ravel()],
              color="white", alpha=0.8, scale=1, scale_units="xy",
              width=0.004, headwidth=5)
    ax.add_patch(_goal_patch())
    ax.set_title("Policy Map (arrows = greedy action)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # ----------------------------------------------------------------
    # 5. Sample trajectories
    # ----------------------------------------------------------------
    ax = axes[1, 1]
    XX_v2, YY_v2, V_v2 = _value_grid(agent, res=40)
    ax.pcolormesh(XX_v2, YY_v2, V_v2, cmap="plasma", shading="auto", alpha=0.5)
    np.random.seed(99)
    for _ in range(10):
        state = GravityBasin.reset()
        traj  = [state.copy()]
        for _ in range(MAX_EP_STEPS):
            a = agent.act(state, epsilon=0.0)
            state, _, done = GravityBasin.step(state, a)
            traj.append(state.copy())
            if done:
                break
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0, alpha=0.85)
        ax.plot(traj[0, 0], traj[0, 1], "o", markersize=4, color="white")
    ax.add_patch(_goal_patch())
    ax.set_title("10 Greedy Trajectories (random starts)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # ----------------------------------------------------------------
    # 6. Koopman dynamics in PCA-projected latent space
    # ----------------------------------------------------------------
    ax = axes[1, 2]
    with torch.no_grad():
        test_pts = torch.tensor(
            np.array([[x, y] for x in np.linspace(-0.9, 0.9, 12)
                             for y in np.linspace(-0.9, 0.9, 12)],
                     dtype=np.float32))
        Z = agent.encode(test_pts)

    Z_np = Z.numpy()
    Z_c  = Z_np - Z_np.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Z_c, full_matrices=False)
    proj = Z_c @ Vt[:2].T

    ax.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=12, zorder=2, label=r"$z_s$")
    for a in range(N_ACTIONS):
        with torch.no_grad():
            Z_next = Z @ agent.K[a].T
        Z_next_np = Z_next.numpy()
        proj_n    = (Z_next_np - Z_np.mean(axis=0)) @ Vt[:2].T
        ax.quiver(proj[:, 0], proj[:, 1],
                  proj_n[:, 0] - proj[:, 0],
                  proj_n[:, 1] - proj[:, 1],
                  color=ACTION_COLORS[a], alpha=0.5,
                  scale=1, scale_units="xy", width=0.003, headwidth=5,
                  label=ACTION_NAMES[a])

    ax.legend(fontsize=8, loc="best")
    ax.set_title("Koopman Dynamics in Latent Space (PCA-2D)\n"
                 r"Arrows: $K_a z \to$ predicted $z_{t+1}$")
    ax.set_xlabel(r"PC$_1$")
    ax.set_ylabel(r"PC$_2$")

    plt.tight_layout()
    plt.savefig("gravity_basin_results.png", dpi=150)
    print("\nSaved → gravity_basin_results.png")
    plt.close()
