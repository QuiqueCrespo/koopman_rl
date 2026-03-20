"""Ornstein-Uhlenbeck exploration noise."""

import numpy as np


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
