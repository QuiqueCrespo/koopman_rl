"""
2D Gravity Basin environment.

Module-level constants mirror the defaults in EnvConfig for backward compatibility.
GravityBasin has instance methods that use values set at __init__ time, so it can
be configured via make_env(cfg) while GravityBasin() with no args matches legacy behaviour.
"""

import numpy as np

from sheaf_rl.config import EnvConfig

# Module-level constants (backward compat — match EnvConfig defaults)
N_ACTIONS    = 4
STATE_DIM    = 2
GOAL_X       = 0.8
GOAL_Y       = 0.8
MAX_EP_STEPS = 200
STEP_PENALTY = -0.01

# Action vectors [dx, dy] for Up, Down, Left, Right
DELTA = np.array([[0.0,  0.1],
                  [0.0, -0.1],
                  [-0.1, 0.0],
                  [ 0.1, 0.0]], dtype=np.float32)

ACTION_NAMES  = ["Up", "Down", "Left", "Right"]
ACTION_COLORS = ["blue", "red", "green", "orange"]


def make_env(cfg: EnvConfig) -> "GravityBasin":
    """Factory: returns a GravityBasin configured from EnvConfig."""
    return GravityBasin(cfg)


class GravityBasin:
    """
    2D plane with cubic gravity toward the origin.

    Dynamics (per axis):
        x_{t+1} = clip(x_t + Δx - 0.05·x_t³, -1, 1)

    Goal: x > goal_x AND y > goal_y  (top-right corner)
    Reward: +1.0 on reaching goal, step_penalty otherwise.

    Can be used in two ways:
        env = GravityBasin()               # uses module-level defaults
        env = GravityBasin(cfg.env)        # uses EnvConfig values
        next_s, r, done = env.step(s, a)
    """

    def __init__(self, cfg: EnvConfig = None):
        cfg = cfg or EnvConfig()
        self.goal_x       = cfg.goal_x
        self.goal_y       = cfg.goal_y
        self.max_ep_steps = cfg.max_ep_steps
        self.step_penalty = cfg.step_penalty
        self.n_actions    = cfg.n_actions

    def step(self, state: np.ndarray, action: int) -> tuple:
        """Stateless step — takes state explicitly so env can be used from visualisation."""
        dx, dy = DELTA[action]
        x, y   = state
        xn = float(np.clip(x + dx - 0.05 * x**3, -1.0, 1.0))
        yn = float(np.clip(y + dy - 0.05 * y**3, -1.0, 1.0))
        ns     = np.array([xn, yn], dtype=np.float32)
        done   = bool(xn > self.goal_x and yn > self.goal_y)
        reward = 1.0 if done else self.step_penalty
        return ns, reward, done

    def reset(self) -> np.ndarray:
        """Uniform random start outside the goal zone."""
        while True:
            s = np.random.uniform(-1.0, 1.0, 2).astype(np.float32)
            if not (s[0] > self.goal_x and s[1] > self.goal_y):
                return s
