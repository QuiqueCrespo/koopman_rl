"""
Gravity Orbit — re-export shim for koopman_global.py.

Drop-in replacement for gravity_basin.py that swaps in GravityOrbit
and provides orbit-specific plot helpers and evaluate function.
Everything else (model, buffer, losses) is unchanged.
"""

# ---- orbit environment ---------------------------------------------------
from env_orbit import (
    GravityOrbit,
    TARGET_RADIUS, SIGMA_R, ORBIT_MAX_EP_STEPS,
)
# expose MAX_EP_STEPS under the standard name so koopman_global.py
# doesn't need a special case
MAX_EP_STEPS = ORBIT_MAX_EP_STEPS

# ---- shared env constants (actions, state dim, etc.) --------------------
from env import N_ACTIONS, STATE_DIM, DELTA, ACTION_NAMES, ACTION_COLORS

# ---- model, buffer, losses — unchanged ----------------------------------
from model import Encoder, ValueNetwork, QNetwork, KoopmanGradientPlanner, TargetNetwork, D, LR, EMA_TAU
from buffer import ReplayBuffer, BUFFER_SIZE, B, T_CHUNK
from losses import (
    compute_contrastive_loss, compute_isometric_loss,
    compute_bisimulation_loss,
    GAMMA, LAMBDA_KOOP, LAMBDA_BISIM, LAMBDA_ISOMETRIC, TAU_START, TAU_END,
)
from train import WARMUP, N_STEPS, EPS_START, EPS_END, EPS_DECAY, LOG_EVERY

# ---- plot helpers --------------------------------------------------------
from plot import _value_grid, _policy_grid

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _orbit_ring(ax, label: str = None, color: str = "lime",
                alpha: float = 0.7, linewidth: float = 1.5):
    """
    Draw the target orbital radius ring on ax.
    Replaces _goal_patch() — call as _orbit_ring(ax) instead of
    ax.add_patch(_goal_patch()).
    """
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(
        TARGET_RADIUS * np.cos(theta),
        TARGET_RADIUS * np.sin(theta),
        color=color, alpha=alpha, linewidth=linewidth,
        linestyle="--", label=label,
    )


def evaluate(agent, n_episodes: int = 20) -> tuple[float, float]:
    """
    Run n_episodes greedy rollouts and return (mean_return, max_return).
    Episode length = ORBIT_MAX_EP_STEPS; no early termination.
    """
    import random
    returns = []
    for _ in range(n_episodes):
        state  = GravityOrbit.reset()
        ep_ret = 0.0
        for _ in range(MAX_EP_STEPS):
            a = agent.act(state, epsilon=0.0)
            state, r, _ = GravityOrbit.step(state, a)
            ep_ret += r
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.max(returns))
