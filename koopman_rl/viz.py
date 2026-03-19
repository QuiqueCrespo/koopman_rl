"""
Visualisation utilities — consolidated from plot.py and koopman_rl_directed.py.

Public API:
  goal_patch(cfg)                       → matplotlib Patch
  value_grid(agent, cfg, res=80)        → (XX, YY, V)
  visualize_graph(graph_data, step, cfg) → saves sheaf_graph_live.png
  plot_live(step, agent, losses, returns, cfg) → saves koopman_rl_live.png
  plot_results(history, cfg)            → saves koopman_rl_directed_results.png
  plot_planner_comparison(results)      → saves planner_comparison.png
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from koopman_rl.env import (
    GravityBasin, GOAL_X, GOAL_Y, MAX_EP_STEPS, N_ACTIONS,
    ACTION_NAMES, ACTION_COLORS, DELTA,
)
from koopman_rl.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_goal(cfg):
    """Return (goal_x, goal_y) from cfg if provided, else module-level defaults."""
    if cfg is None:
        return GOAL_X, GOAL_Y
    return cfg.env.goal_x, cfg.env.goal_y


def goal_patch(cfg=None, alpha: float = 0.4, **kw) -> patches.Rectangle:
    gx, gy = _cfg_goal(cfg)
    return patches.Rectangle(
        (gx, gy), 1 - gx, 1 - gy,
        color="lime", alpha=alpha, **kw,
    )

# Legacy name used by plot.py callers
_goal_patch = goal_patch


def _smooth(x, w=10):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


@torch.no_grad()
def value_grid(agent, cfg=None, res: int = 80):
    xs = np.linspace(-1, 1, res)
    ys = np.linspace(-1, 1, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1).astype(np.float32)
    device = next(agent.parameters()).device
    t   = torch.from_numpy(pts).to(device)
    V   = agent.v_net(agent.encode(t))
    return XX, YY, V.cpu().numpy().reshape(res, res)

# Legacy name
_value_grid = value_grid


# ---------------------------------------------------------------------------
# Graph topology visualisation
# ---------------------------------------------------------------------------

def visualize_graph(graph_data: dict, step: int, cfg=None) -> None:
    """
    1×2 diagnostic figure → sheaf_graph_live.png.
    Left:  graph topology — action-colored directed arrows, bisim bridges.
    Right: diffused values — nodes by V_diff, gold stars at reward nodes.
    """
    from matplotlib.collections import LineCollection
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    positions  = graph_data["positions"]
    src_nodes  = graph_data["src_nodes"]
    dst_nodes  = graph_data["dst_nodes"]
    actions    = graph_data["actions"]
    rewards    = graph_data["rewards"]
    bisim_src  = graph_data["bisim_src"]
    bisim_dst  = graph_data["bisim_dst"]
    bisim_dist = graph_data["bisim_dist"]
    R_v        = graph_data["R_v"]
    V_diff     = graph_data["V_diff"]

    k_diffuse      = cfg.algo.k_diffuse      if cfg else 50
    bisim_penalty  = cfg.algo.bisim_penalty_scale if cfg else 1.0

    N       = positions.shape[0]
    E_bisim = bisim_src.shape[0]
    v_lo    = 0.0
    v_hi    = max(V_diff.max(), 1e-2)

    temp_segs = np.stack([positions[src_nodes], positions[dst_nodes]], axis=1)
    if E_bisim > 0:
        bisim_segs = np.stack([positions[bisim_src], positions[bisim_dst]], axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left — Graph Topology
    if E_bisim > 0:
        bisim_lc = LineCollection(bisim_segs, colors="gray", linewidths=0.4,
                                  linestyles="dashed", alpha=0.15, zorder=1)
        ax1.add_collection(bisim_lc)

    edge_rgba = np.array([(*mcolors.to_rgb(ACTION_COLORS[int(a)]), 0.5) for a in actions])
    temp_lc   = LineCollection(temp_segs, colors=edge_rgba, linewidths=0.8, zorder=2)
    ax1.add_collection(temp_lc)

    ax1.quiver(positions[src_nodes, 0], positions[src_nodes, 1],
               positions[dst_nodes, 0] - positions[src_nodes, 0],
               positions[dst_nodes, 1] - positions[src_nodes, 1],
               color=[ACTION_COLORS[int(a)] for a in actions],
               alpha=0.5, scale=1, scale_units="xy",
               width=0.003, headwidth=5, headlength=5, zorder=3)

    ax1.scatter(positions[:, 0], positions[:, 1],
                c=V_diff, cmap="plasma", s=15, zorder=4, vmin=v_lo, vmax=v_hi)

    rew_mask = rewards > 0
    if rew_mask.any():
        rsp = positions[src_nodes[rew_mask]]
        rdp = positions[dst_nodes[rew_mask]]
        ax1.quiver(rsp[:, 0], rsp[:, 1],
                   rdp[:, 0] - rsp[:, 0], rdp[:, 1] - rsp[:, 1],
                   color="gold", alpha=0.9, scale=1, scale_units="xy",
                   width=0.005, headwidth=7, headlength=7, zorder=5)

    ax1.add_patch(goal_patch(cfg))
    ax1.set_xlim(-1.05, 1.05); ax1.set_ylim(-1.05, 1.05); ax1.set_aspect("equal")
    action_patches = [mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_NAMES[a])
                      for a in range(N_ACTIONS)]
    bisim_line = plt.Line2D([0], [0], color="gray", linestyle="dashed",
                            linewidth=0.8, label=f"bisim ({E_bisim})")
    ax1.legend(handles=action_patches + [bisim_line], fontsize=7, loc="lower left", ncol=2)
    ax1.set_title(f"Graph topology  |  step {step}  |  {N} nodes  {E_bisim} bisim edges")

    # Right — Directed Values
    v_src_norm = np.clip((V_diff[src_nodes] - v_lo) / (v_hi - v_lo + 1e-6), 0, 1)
    temp_lc2   = LineCollection(temp_segs, colors=plt.cm.plasma(v_src_norm),
                                linewidths=0.8, alpha=0.4, zorder=1)
    ax2.add_collection(temp_lc2)

    if E_bisim > 0:
        bisim_lc2 = LineCollection(bisim_segs, colors="gray", linewidths=0.3,
                                   linestyles="dashed", alpha=0.1, zorder=1)
        ax2.add_collection(bisim_lc2)

    sc2 = ax2.scatter(positions[:, 0], positions[:, 1],
                      c=V_diff, cmap="plasma", s=20, zorder=4, vmin=v_lo, vmax=v_hi)
    plt.colorbar(sc2, ax=ax2, label="V_diff")

    rew_node_mask = R_v > 1e-4
    if rew_node_mask.any():
        sizes = np.clip(R_v[rew_node_mask] * 200, 30, 300)
        ax2.scatter(positions[rew_node_mask, 0], positions[rew_node_mask, 1],
                    marker="*", c="gold", s=sizes, zorder=6,
                    edgecolors="orange", linewidths=0.5, label="R_v > 0")
        ax2.legend(fontsize=8, loc="lower left")

    ax2.add_patch(goal_patch(cfg))
    ax2.set_xlim(-1.05, 1.05); ax2.set_ylim(-1.05, 1.05); ax2.set_aspect("equal")
    stats = (f"V_diff: μ={V_diff.mean():.3f}  max={V_diff.max():.3f}\n"
             f"bisim dist: μ={bisim_dist.mean():.3f}  min={bisim_dist.min():.3f}"
             if len(bisim_dist) > 0 else "bisim dist: n/a (disabled)")
    ax2.text(0.02, 0.98, stats, transform=ax2.transAxes, fontsize=8,
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax2.set_title(f"Directed values  |  K={k_diffuse} Bellman steps  "
                  f"|  penalty={bisim_penalty}×dist")

    plt.tight_layout()
    run = cfg.run_name if cfg else "run"
    plt.savefig(f"{run}_graph_live.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Live training plot
# ---------------------------------------------------------------------------

def plot_live(
    step:            int,
    agent,
    koop_losses:     list,
    v_losses:        list,
    bisim_losses:    list,
    episode_returns: list,
    graph_v_diff=None,
    cfg=None,
) -> None:
    cpu    = torch.device("cpu")
    device = next(agent.parameters()).device
    agent.to(cpu)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Koopman-RL v2 (Directed VI)  —  step {step:,}", fontsize=13)

    ax = axes[0, 0]
    if koop_losses:
        ax.semilogy(_smooth(koop_losses), label=r"$\mathcal{L}_{Koop}$",  color="steelblue")
        ax.semilogy(_smooth(v_losses),    label=r"$\mathcal{L}_{V}$",     color="tomato")
        ax.semilogy(_smooth(bisim_losses),label=r"$\mathcal{L}_{recon}$", color="purple")
    ax.set_xlabel("Log interval"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Losses"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if graph_v_diff is not None:
        vd = graph_v_diff.cpu().numpy()
        ax.hist(vd, bins=50, color="tomato", alpha=0.7, density=True)
        ax.axvline(vd.mean(), color="darkred", linestyle="--",
                   label=f"μ={vd.mean():.3f}  max={vd.max():.3f}")
        ax.set_xlabel("Directed V"); ax.set_ylabel("Density")
        ax.set_title("Directed Value Distribution"); ax.legend(fontsize=9)
    else:
        if episode_returns:
            succ = [1.0 if r > 0 else 0.0 for r in episode_returns]
            w = min(50, len(succ))
            ep_x = np.arange(len(succ))
            # Raw returns as faint scatter
            ax.scatter(ep_x, episode_returns, s=2, color="gray", alpha=0.3, zorder=1)
            # Rolling return
            if len(episode_returns) >= w:
                roll_ret = np.convolve(episode_returns, np.ones(w) / w, mode="valid")
                ax.plot(np.arange(w - 1, len(succ)), roll_ret,
                        color="tomato", linewidth=1.2, label=f"Return (rolling {w})", zorder=2)
            ax.set_ylabel("Return", color="tomato")
            ax.tick_params(axis="y", labelcolor="tomato")
            # Success rate on twin axis
            ax2 = ax.twinx()
            if len(succ) >= w:
                roll_sr = np.convolve(succ, np.ones(w) / w, mode="valid") * 100
                ax2.plot(np.arange(w - 1, len(succ)), roll_sr,
                         color="royalblue", linewidth=1.5, label=f"Success % (rolling {w})", zorder=3)
            ax2.axhline(100, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
            ax2.set_ylabel("Success rate (%)", color="royalblue")
            ax2.tick_params(axis="y", labelcolor="royalblue")
            ax2.set_ylim(0, 108)
            n_succ = int(sum(succ))
            # Combined legend
            lines1, labs1 = ax.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")
            ax.set_title(f"Returns & Success Rate  ({n_succ}/{len(succ)} ep)")
            ax.set_xlabel("Episode")
        else:
            ax.set_title("Returns & Success Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    try:
        XX, YY, V = value_grid(agent, cfg, res=80)
        im = ax.pcolormesh(XX, YY, V, cmap="plasma", shading="auto")
        plt.colorbar(im, ax=ax, label=r"$V_\psi(f_\theta(s))$")
        ax.add_patch(goal_patch(cfg, label="Goal zone"))
        ax.legend(loc="lower left", fontsize=8)
    except Exception:
        ax.text(0.5, 0.5, "Value grid unavailable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title(r"Learned Value Map $V(x,y)$")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")

    ax = axes[1, 1]
    n_a = cfg.env.n_actions if cfg else N_ACTIONS
    try:
        with torch.no_grad():
            test_pts = torch.tensor(
                np.array([[x, y] for x in np.linspace(-0.9, 0.9, 10)
                                 for y in np.linspace(-0.9, 0.9, 10)],
                         dtype=np.float32))
            Z = agent.encode(test_pts)
        Z_np = Z.numpy()
        Z_c  = Z_np - Z_np.mean(axis=0, keepdims=True)
        _, s_vals, Vt = np.linalg.svd(Z_c, full_matrices=False)
        var_exp = (s_vals[:2] ** 2).sum() / (s_vals ** 2).sum() * 100
        proj = Z_c @ Vt[:2].T
        ax.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=10, zorder=2)
        for a in range(n_a):
            with torch.no_grad():
                Z_next = agent.dyn_step(Z, agent.B[:, a]).numpy()
            pn = (Z_next - Z_np.mean(axis=0)) @ Vt[:2].T
            ax.quiver(proj[:, 0], proj[:, 1],
                      pn[:, 0] - proj[:, 0], pn[:, 1] - proj[:, 1],
                      color=ACTION_COLORS[a], alpha=0.5,
                      scale=1, scale_units="xy", width=0.004, headwidth=5,
                      label=ACTION_NAMES[a])
        ax.legend(fontsize=7, loc="best")
    except Exception:
        ax.text(0.5, 0.5, "Dynamics plot unavailable", ha="center", va="center",
                transform=ax.transAxes)
        var_exp = 0.0
    ax.set_title(f"Linear Dynamics (PCA-2D, {var_exp:.0f}% var)")
    ax.set_xlabel("PC$_1$"); ax.set_ylabel("PC$_2$")

    run = cfg.run_name if cfg else "run"
    plt.tight_layout()
    plt.savefig(f"{run}_live.png", dpi=120)
    plt.close(fig)
    agent.to(device)


# ---------------------------------------------------------------------------
# Final results plot
# ---------------------------------------------------------------------------

def plot_results(history: dict, cfg=None) -> None:
    agent        = history["agent"]
    koop_losses  = history["koop_losses"]
    v_losses     = history["v_losses"]
    bisim_losses = history["bisim_losses"]
    ep_returns   = history["episode_returns"]

    agent.to(torch.device("cpu"))
    n_a = cfg.env.n_actions if cfg else N_ACTIONS

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    ax = axes[0, 0]
    ax.semilogy(_smooth(koop_losses), label=r"$\mathcal{L}_{Koop}$", color="steelblue")
    ax.semilogy(_smooth(v_losses),    label=r"$\mathcal{L}_{V}$ (directed VI)", color="tomato")
    ax.semilogy(_smooth(bisim_losses),label=r"$\mathcal{L}_{bisim}$", color="purple")
    ax.set_xlabel("Log interval"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Losses"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    succ = [1.0 if r > 0 else 0.0 for r in ep_returns]
    w2 = min(50, len(succ))
    ep_x = np.arange(len(succ))
    ax.scatter(ep_x, ep_returns, s=2, color="gray", alpha=0.3, zorder=1)
    if len(ep_returns) >= w2:
        roll_ret = np.convolve(ep_returns, np.ones(w2) / w2, mode="valid")
        ax.plot(np.arange(w2 - 1, len(succ)), roll_ret,
                color="tomato", linewidth=1.2, label=f"Return (rolling {w2})", zorder=2)
    ax.set_ylabel("Return", color="tomato")
    ax.tick_params(axis="y", labelcolor="tomato")
    ax2 = axes[0, 1].twinx()
    if len(succ) >= w2:
        roll_sr = np.convolve(succ, np.ones(w2) / w2, mode="valid") * 100
        ax2.plot(np.arange(w2 - 1, len(succ)), roll_sr,
                 color="royalblue", linewidth=1.5, label=f"Success % (rolling {w2})", zorder=3)
    ax2.axhline(100, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_ylabel("Success rate (%)", color="royalblue")
    ax2.tick_params(axis="y", labelcolor="royalblue")
    ax2.set_ylim(0, 108)
    n_succ = int(sum(succ))
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")
    ax.set_xlabel("Episode")
    ax.set_title(f"Returns & Success Rate  ({n_succ}/{len(succ)} ep)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    XX, YY, V = value_grid(agent, cfg, res=100)
    im = ax.pcolormesh(XX, YY, V, cmap="plasma", shading="auto")
    plt.colorbar(im, ax=ax, label=r"$V_\psi(f_\theta(s))$")
    ax.add_patch(goal_patch(cfg, label="Goal zone"))
    ax.set_title(r"Learned Value Map $V(x,y)$")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="lower left", fontsize=8); ax.set_aspect("equal")

    ax = axes[1, 1]
    with torch.no_grad():
        test_pts = torch.tensor(
            np.array([[x, y] for x in np.linspace(-0.9, 0.9, 12)
                             for y in np.linspace(-0.9, 0.9, 12)],
                     dtype=np.float32))
        Z = agent.encode(test_pts)
    Z_np = Z.numpy()
    Z_c  = Z_np - Z_np.mean(axis=0, keepdims=True)
    _, s_vals, Vt = np.linalg.svd(Z_c, full_matrices=False)
    var_exp = (s_vals[:2] ** 2).sum() / (s_vals ** 2).sum() * 100
    proj = Z_c @ Vt[:2].T
    ax.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=12, zorder=2, label=r"$z_s$")
    for a in range(n_a):
        with torch.no_grad():
            Z_next = agent.dyn_step(Z, agent.B[:, a])
        proj_n = (Z_next.numpy() - Z_np.mean(axis=0)) @ Vt[:2].T
        ax.quiver(proj[:, 0], proj[:, 1],
                  proj_n[:, 0] - proj[:, 0], proj_n[:, 1] - proj[:, 1],
                  color=ACTION_COLORS[a], alpha=0.5,
                  scale=1, scale_units="xy", width=0.003, headwidth=5,
                  label=ACTION_NAMES[a])
    ax.legend(fontsize=8, loc="best")
    ax.set_title(f"Linear Dynamics (PCA-2D, {var_exp:.0f}% var)\n"
                 r"Arrows: $Az + Be_a \to$ predicted $z_{t+1}$")
    ax.set_xlabel("PC$_1$"); ax.set_ylabel("PC$_2$")

    run = cfg.run_name if cfg else "run"
    plt.tight_layout()
    plt.savefig(f"{run}_results.png", dpi=150)
    print(f"\nSaved -> {run}_results.png")
    plt.close()


# ---------------------------------------------------------------------------
# Planner comparison
# ---------------------------------------------------------------------------

def plot_planner_comparison(results: dict) -> None:
    """Bar chart comparing planners by success rate."""
    modes      = list(results.keys())
    sr         = [sum(v > 0 for v in results[m]) / len(results[m]) * 100 for m in modes]

    labels = ["Greedy\nact()", "Softmax\n(terminal)",
              "Softmax\n(cumulative)", "Random Shooting\n(K=200)", "Beam Search\n(W=8)"]
    labels = labels[:len(modes)]   # trim if fewer planners
    colors = ["steelblue", "tomato", "seagreen", "orange", "purple"][:len(modes)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, sr, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, sr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title("Greedy vs Hierarchical Latent Planner\n(ε=0)", fontsize=13)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("planner_comparison.png", dpi=130)
    print("\nSaved → planner_comparison.png")
    plt.close()


# ---------------------------------------------------------------------------
# Ablation comparison (moved from ablation_compare.py)
# ---------------------------------------------------------------------------

def plot_ablation_comparison(results: list, out: str = "ablation_comparison.png") -> None:
    """Bar + learning curve figure for a list of ablation result dicts."""
    import matplotlib.patches as mpatches

    # Sort: BASE first, then alphabetically
    results = sorted(results, key=lambda r: (r["run_name"] != "BASE", r["run_name"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(6, len(results) * 0.45 + 2)))
    fig.suptitle("Ablation Study — Koopman-RL Directed VI", fontsize=14, fontweight="bold")

    names      = [r["run_name"] for r in results]
    succ_final = [r["succ_final"] for r in results]
    base_final = next((r["succ_final"] for r in results if r["run_name"] == "BASE"), None)

    def bar_color(val, base):
        if base is None: return "steelblue"
        if val >= base:  return "#4caf50"
        if val >= base - 2: return "#ff9800"
        return "#f44336"

    colors    = [bar_color(v, base_final) for v in succ_final]
    colors[0] = "#1565c0"

    y_pos = np.arange(len(names))
    bars  = ax1.barh(y_pos, succ_final, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, succ_final):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", ha="left", fontsize=9, fontweight="bold")
    if base_final is not None:
        ax1.axvline(base_final, color="#1565c0", linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.set_yticks(y_pos); ax1.set_yticklabels(names, fontsize=9)
    ax1.set_xlabel("succ / 20  (last 20 training episodes)")
    ax1.set_title("Final Success (step 100k)")
    ax1.set_xlim(0, 22)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    legend_patches = [
        mpatches.Patch(color="#4caf50", label="≥ baseline"),
        mpatches.Patch(color="#ff9800", label="within 2"),
        mpatches.Patch(color="#f44336", label="> 2 below"),
        plt.Line2D([0], [0], color="#1565c0", linestyle="--", linewidth=1.2,
                   label=f"BASE = {base_final}/20"),
    ]
    ax1.legend(handles=legend_patches, fontsize=8, loc="lower right")

    def rolling_succ(ep_list, window=20):
        out = []
        for i in range(len(ep_list)):
            chunk = ep_list[max(0, i - window): i + 1]
            out.append(sum(r > 0 for r in chunk) / window)
        return np.array(out)

    cmap = plt.get_cmap("tab20")
    for idx, r in enumerate(results):
        ep = r.get("episode_returns", [])
        if not ep:
            continue
        rs  = rolling_succ(ep) * 20
        lw  = 2.0 if r["run_name"] == "BASE" else 0.9
        col = "#1565c0" if r["run_name"] == "BASE" else cmap(idx / len(results))
        ax2.plot(np.arange(len(rs)), rs, lw=lw, color=col, alpha=0.85, label=r["run_name"])

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rolling succ / 20")
    ax2.set_title("Learning Curves (rolling 20-ep success)")
    ax2.set_ylim(0, 21)
    ax2.axhline(20, color="gray", linestyle=":", linewidth=0.8)
    ax2.legend(fontsize=7, loc="upper left", ncol=2)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()
