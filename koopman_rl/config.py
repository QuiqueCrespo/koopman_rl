"""
Centralised hyperparameter config for koopman_rl.

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
    action_scale: float = 1.0    # multiply tanh output → real action range
    continuous:   bool  = False  # continuous action space (vs discrete)
    obs_type:     str   = "state"  # "state" | "pixels"
    img_size:     int   = 64       # pixel H = W (square frames assumed)
    img_channels: int   = 3


@dataclass
class ModelConfig:
    d:        int   = 32       # latent dimension
    lr:       float = 3e-4
    ema_tau:  float = 0.005
    ortho_a:  bool  = False    # constrain A ∈ O(d) via SVD Procrustes (CUDA) / soft penalty (MPS/CPU)
    tanh_out: bool  = False    # tanh final encoder layer instead of L2 norm


@dataclass
class BufferConfig:
    capacity:   int = 100_000
    batch_size: int = 512


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
    no_graph:            bool  = True
    td_plus_vi:          bool  = False
    fix_a:               bool  = False
    no_normalize:        bool  = False
    lambda_ortho:        float = 1.0   # weight of soft ||AᵀA − I||²_F penalty (ortho_a=True only)
    noise_z_std:         float = 0.0   # Gaussian noise std injected into z before A·z+Bu (0 = disabled)
    utd_ratio:           int   = 1     # gradient steps per env step (update-to-data ratio)
    lambda_sigreg:       float = 0.0   # weight of SIGReg (Sketch Isotropic Gaussian) loss (0 = disabled)
    use_target_encoder:  bool  = True  # use EMA target encoder for z_dst (False → online encoder, rely on SIGReg)
    # Continuous control
    reward_scale:        float = 1.0   # divide rewards before TD
    n_envs:              int   = 1     # parallel envs for vectorised collection


@dataclass
class PlannerConfig:
    horizon:    int   = 10
    plan_iters: int   = 20
    lr:         float = 0.1
    tau:        float = 1.0    # Gumbel temperature (lower → more discrete / sharper)
    n_samples:  int   = 200    # random shooting
    beam_width: int   = 8      # beam search
    # CEM-Gradient hybrid (plan_cem_gradient_batch)
    cem_iters:      int = 20    # CEM refitting rounds
    cem_samples:    int = 1000  # trajectories sampled per CEM round
    cem_elites:     int = 30   # top-K kept for Gaussian refit
    cem_grad_iters: int = 20   # Adam steps after CEM warm-start


@dataclass
class TrainConfig:
    n_steps:    int   = 100_000
    warmup:     int   = 20_000
    eps_start:  float = 1.0
    eps_end:    float = 0.05
    eps_decay:  int   = 40_000
    log_every:  int   = 2_000
    plot_every: int   = 2_000
    # Continuous control
    noise_start:  float = 1.0
    noise_end:    float = 0.05
    noise_decay:  int   = 40_000
    viz_every:    int   = 5_000
    viz_dir:      str   = "output/viz"
    ckpt_dir:     str   = "output/checkpoints"
    planner_type: str   = "policy"     # data collection always uses policy; benchmark: "policy" | "toeplitz" | "sequential"
    frozen_b:     bool  = False        # detach B in sequential planner
    ou_noise:     bool  = False        # Ornstein-Uhlenbeck vs i.i.d. Gaussian


@dataclass
class Config:
    env:      EnvConfig     = field(default_factory=EnvConfig)
    model:    ModelConfig   = field(default_factory=ModelConfig)
    buffer:   BufferConfig  = field(default_factory=BufferConfig)
    algo:     AlgoConfig    = field(default_factory=AlgoConfig)
    train:    TrainConfig   = field(default_factory=TrainConfig)
    planner:  PlannerConfig = field(default_factory=PlannerConfig)
    run_name: str           = "unnamed"
    seed:     int           = 42
    device:   str           = "auto"   # "auto" | "cpu" | "cuda" | "mps"

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
            planner=PlannerConfig(**d.get("planner", {})),
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
            "LAMBDA_ORTHO":        ("algo", "lambda_ortho"),
            "D":                   ("model", "d"),
            "ORTHO_A":             ("model", "ortho_a"),
        }
        cfg = cls()
        for k, v in overrides.items():
            if k in MAPPING:
                group, attr = MAPPING[k]
                setattr(getattr(cfg, group), attr, v)
        return cfg
