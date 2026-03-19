"""ortho_a + tanh_out: A ∈ O(d), encoder bounded by tanh (no L2 norm)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from koopman_rl.config import Config, TrainConfig, ModelConfig

cfg = Config(
    run_name="ortho_a_tanh",
    model=ModelConfig(ortho_a=True, tanh_out=True),
    train=TrainConfig(
        n_steps=50_000,
        warmup=3_000,
        log_every=2_000,
        plot_every=2_000,
    ),
)
