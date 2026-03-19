"""ortho_a + tanh encoder (bounded hypercube), 100k steps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from koopman_rl.config import Config, TrainConfig, ModelConfig

cfg = Config(
    run_name="ortho_tanh",
    model=ModelConfig(ortho_a=True, tanh_out=True),
    train=TrainConfig(n_steps=100_000, warmup=20_000, log_every=5_000, plot_every=5_000,
                      eps_end=0.15, eps_decay=70_000),
)
