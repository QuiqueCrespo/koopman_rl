"""
sheaf_rl — Latent Affine Sheaf Reinforcement Learning package.
"""

__version__ = "0.1.0"

from sheaf_rl.config import (Config, EnvConfig, ModelConfig, BufferConfig,
                              AlgoConfig, TrainConfig, PlannerConfig)
from sheaf_rl.env import GravityBasin, make_env
from sheaf_rl.model import (Encoder, ValueNetwork, QNetwork,
                             KoopmanGradientPlanner, TargetNetwork)
from sheaf_rl.buffer import ReplayBuffer
from sheaf_rl.algorithms import train, evaluate, directed_value_iteration, build_and_propagate

# Backward-compatibility alias
SheafAgent = KoopmanGradientPlanner

__all__ = [
    "Config", "EnvConfig", "ModelConfig", "BufferConfig", "AlgoConfig",
    "TrainConfig", "PlannerConfig",
    "GravityBasin", "make_env",
    "Encoder", "ValueNetwork", "QNetwork",
    "KoopmanGradientPlanner", "SheafAgent", "TargetNetwork",
    "ReplayBuffer",
    "train", "evaluate", "directed_value_iteration", "build_and_propagate",
]
