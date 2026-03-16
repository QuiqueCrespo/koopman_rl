"""
2D Gravity Orbit — environment.

The agent must maintain a circular orbit at TARGET_RADIUS around the origin
while fighting cubic gravity that pulls toward (0, 0).

Dynamics (identical to GravityBasin):
    x_{t+1} = clip(x_t + Δx - 0.05·x_t³, -1, 1)

Reward: Gaussian centred on the target orbital radius:
    r_t = exp(-0.5 · ((||s_t|| - TARGET_RADIUS) / SIGMA_R)²)

    r = 1.0  when exactly on orbit
    r → 0    at the gravity-trap origin (||s|| = 0)
    r → 0    far outside orbit

Done: never — episodes end only at the step limit (ORBIT_MAX_EP_STEPS).
"""

import numpy as np

# -- shared with GravityBasin (imported from env.py by other modules) -------
from env import N_ACTIONS, STATE_DIM, DELTA, ACTION_NAMES, ACTION_COLORS

# -- orbit-specific constants -----------------------------------------------
TARGET_RADIUS      = 0.6    # desired orbital radius
SIGMA_R            = 0.05   # reward Gaussian width (±1σ ≈ ±0.05 from orbit)
ORBIT_MAX_EP_STEPS = 500    # longer episodes — no early termination


class GravityOrbit:
    """
    2D gravity environment where the task is to orbit at TARGET_RADIUS.

    Same cubic-gravity dynamics as GravityBasin; different reward and
    termination: reward is dense (Gaussian in radial distance), done=False
    always so every episode runs for exactly ORBIT_MAX_EP_STEPS steps.
    """

    @staticmethod
    def step(state: np.ndarray, action: int) -> tuple[np.ndarray, float, bool]:
        dx, dy = DELTA[action]
        x, y   = state
        xn = float(np.clip(x + dx - 0.05 * x**3, -1.0, 1.0))
        yn = float(np.clip(y + dy - 0.05 * y**3, -1.0, 1.0))
        ns   = np.array([xn, yn], dtype=np.float32)
        dist = float(np.sqrt(xn**2 + yn**2))
        reward = float(np.exp(-0.5 * ((dist - TARGET_RADIUS) / SIGMA_R) ** 2))
        return ns, reward, False   # done is always False

    @staticmethod
    def reset() -> np.ndarray:
        """Uniform random start anywhere in the state space."""
        return np.random.uniform(-1.0, 1.0, 2).astype(np.float32)
