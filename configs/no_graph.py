"""Pure TD baseline — graph VI disabled."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sheaf_rl.config import Config, AlgoConfig

cfg = Config(
    run_name="no_graph",
    algo=AlgoConfig(no_graph=True),
)
