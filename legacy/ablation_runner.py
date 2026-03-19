"""
ablation_runner.py — thin shim.
Logic has moved to scripts/ablation_run.py.

All CLI flags are identical; this file just forwards to the new script.
"""

import subprocess
import sys
from pathlib import Path

scripts_dir = Path(__file__).parent / "scripts"
result = subprocess.run(
    [sys.executable, str(scripts_dir / "ablation_run.py")] + sys.argv[1:],
    check=False,
)
sys.exit(result.returncode)
