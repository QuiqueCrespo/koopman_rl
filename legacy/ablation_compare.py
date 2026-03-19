"""
ablation_compare.py — thin shim.
Logic has moved to scripts/ablation_compare.py.

Loads from both results/ablation/ (new) and ablation_results/ (legacy).
"""

import subprocess
import sys
from pathlib import Path

scripts_dir = Path(__file__).parent / "scripts"

# Pass through any args; default results-dir tries legacy path
extra = sys.argv[1:]
if not any("--results-dir" in a for a in extra):
    extra = ["--results-dir", "ablation_results"] + extra

result = subprocess.run(
    [sys.executable, str(scripts_dir / "ablation_compare.py")] + extra,
    check=False,
)
sys.exit(result.returncode)
