"""
koopman_rl — Koopman Reinforcement Learning package.
"""

__version__ = "0.1.0"

from koopman_rl.config import (Config, EnvConfig, ModelConfig, BufferConfig,
                              AlgoConfig, TrainConfig, PlannerConfig)
from koopman_rl.env import GravityBasin, make_env
from koopman_rl.model import (Encoder, ValueNetwork, QNetwork,
                             KoopmanGradientPlanner, TargetNetwork)
from koopman_rl.buffer import ReplayBuffer
from koopman_rl.algorithms import train, evaluate, directed_value_iteration, build_and_propagate

# Backward-compatibility alias
KoopmanAgent = KoopmanGradientPlanner

__all__ = [
    "Config", "EnvConfig", "ModelConfig", "BufferConfig", "AlgoConfig",
    "TrainConfig", "PlannerConfig",
    "GravityBasin", "make_env",
    "Encoder", "ValueNetwork", "QNetwork",
    "KoopmanGradientPlanner", "KoopmanAgent", "TargetNetwork",
    "ReplayBuffer",
    "train", "evaluate", "directed_value_iteration", "build_and_propagate",
]
