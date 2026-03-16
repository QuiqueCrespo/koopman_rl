"""Base config — mirrors current default hyperparameters."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sheaf_rl.config import Config

cfg = Config(run_name="base")
