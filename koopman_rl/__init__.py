"""
koopman_rl — Koopman Reinforcement Learning package.
"""

__version__ = "0.1.0"

from koopman_rl.config import (Config, EnvConfig, ModelConfig, BufferConfig,
                              AlgoConfig, TrainConfig, PlannerConfig)
from koopman_rl.env import GravityBasin, make_env
from koopman_rl.model import (Encoder, ValueNetwork, QNetwork,
                             KoopmanGradientPlanner, TargetNetwork)
from koopman_rl.buffer import ReplayBuffer, ContinuousReplayBuffer
from koopman_rl.noise import OUNoise
from koopman_rl.algorithms import train, evaluate, directed_value_iteration, build_and_propagate

# Backward-compatibility alias
KoopmanGradientPlanner = KoopmanGradientPlanner

__all__ = [
    "Config", "EnvConfig", "ModelConfig", "BufferConfig", "AlgoConfig",
    "TrainConfig", "PlannerConfig",
    "GravityBasin", "make_env",
    "Encoder", "ValueNetwork", "QNetwork",
    "KoopmanGradientPlanner", "KoopmanGradientPlanner", "TargetNetwork",
    "ReplayBuffer", "ContinuousReplayBuffer",
    "OUNoise",
    "train", "evaluate", "directed_value_iteration", "build_and_propagate",
]
