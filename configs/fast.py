"""Fast config — small graph + 20k steps for quick iteration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sheaf_rl.config import Config, TrainConfig, AlgoConfig

cfg = Config(
    run_name="fast",
    train=TrainConfig(
        n_steps=20_000,
        warmup=1_000,
        log_every=500,
        plot_every=500,
    ),
    algo=AlgoConfig(
        n_chunks=16,
        k_diffuse=10,
        graph_rebuild=200,
    ),
)
