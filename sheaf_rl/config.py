"""
Centralised hyperparameter config for sheaf_rl.

All scattered module-level constants are consolidated here as typed dataclasses.
Serialisable to/from dict and JSON with no extra dependencies.
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class EnvConfig:
    state_dim:    int   = 2
    n_actions:    int   = 4
    goal_x:       float = 0.8
    goal_y:       float = 0.8
    max_ep_steps: int   = 200
    step_penalty: float = -0.01


@dataclass
class ModelConfig:
    d:       int   = 32       # latent dimension
    lr:      float = 3e-4
    ema_tau: float = 0.005


@dataclass
class BufferConfig:
    capacity:   int = 100_000
    batch_size: int = 256


@dataclass
class AlgoConfig:
    gamma:               float = 0.95
    lambda_koop:         float = 1.0
    lambda_v:            float = 0.5
    lambda_recon:        float = 1.0
    koop_lr_scale:       float = 0.5
    # Graph
    n_chunks:            int   = 32
    t_chunk:             int   = 16
    k_bisim_nn:          int   = 5
    bisim_penalty_scale: float = 1.0
    force_goal:          bool  = True
    graph_rebuild:       int   = 500
    k_diffuse:           int   = 50
    stratified:          bool  = True
    # Ablation switches
    no_graph:            bool  = False
    td_plus_vi:          bool  = False
    fix_a:               bool  = False
    no_normalize:        bool  = False


@dataclass
class TrainConfig:
    n_steps:    int   = 100_000
    warmup:     int   = 3_000
    eps_start:  float = 1.0
    eps_end:    float = 0.05
    eps_decay:  int   = 40_000
    log_every:  int   = 2_000
    plot_every: int   = 2_000


@dataclass
class Config:
    env:      EnvConfig    = field(default_factory=EnvConfig)
    model:    ModelConfig  = field(default_factory=ModelConfig)
    buffer:   BufferConfig = field(default_factory=BufferConfig)
    algo:     AlgoConfig   = field(default_factory=AlgoConfig)
    train:    TrainConfig  = field(default_factory=TrainConfig)
    run_name: str          = "unnamed"
    seed:     int          = 42
    device:   str          = "auto"   # "auto" | "cpu" | "cuda" | "mps"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Create Config from nested dict (e.g., loaded from JSON)."""
        return cls(
            env=EnvConfig(**d.get("env", {})),
            model=ModelConfig(**d.get("model", {})),
            buffer=BufferConfig(**d.get("buffer", {})),
            algo=AlgoConfig(**d.get("algo", {})),
            train=TrainConfig(**d.get("train", {})),
            run_name=d.get("run_name", "unnamed"),
            seed=d.get("seed", 42),
            device=d.get("device", "auto"),
        )

    @classmethod
    def from_ablation_dict(cls, overrides: dict) -> "Config":
        """Compatibility shim: create Config from flat ablation override dict.
        Maps old uppercase keys (K_BISIM_NN, NO_GRAPH, ...) to nested dataclass fields."""
        MAPPING = {
            "K_BISIM_NN":          ("algo", "k_bisim_nn"),
            "BISIM_PENALTY_SCALE": ("algo", "bisim_penalty_scale"),
            "K_DIFFUSE":           ("algo", "k_diffuse"),
            "LAMBDA_RECON":        ("algo", "lambda_recon"),
            "LAMBDA_V":            ("algo", "lambda_v"),
            "N_CHUNKS":            ("algo", "n_chunks"),
            "T_CHUNK":             ("algo", "t_chunk"),
            "FORCE_GOAL":          ("algo", "force_goal"),
            "STRATIFIED":          ("algo", "stratified"),
            "NO_GRAPH":            ("algo", "no_graph"),
            "TD_PLUS_VI":          ("algo", "td_plus_vi"),
            "FIX_A":               ("algo", "fix_a"),
            "NO_NORMALIZE":        ("algo", "no_normalize"),
            "D":                   ("model", "d"),
        }
        cfg = cls()
        for k, v in overrides.items():
            if k in MAPPING:
                group, attr = MAPPING[k]
                setattr(getattr(cfg, group), attr, v)
        return cfg
