"""
Load all results/ablation/*.json and produce comparison figure + markdown table.

Usage:
  python scripts/ablation_compare.py
  python scripts/ablation_compare.py --results-dir ablation_results  # legacy path
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sheaf_rl.viz import plot_ablation_comparison

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="results/ablation",
                    help="Directory containing *.json result files")
parser.add_argument("--out", default="ablation_comparison.png",
                    help="Output figure path")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.results_dir}/*.json"))
# Fallback: try legacy ablation_results/
if not files:
    files = sorted(glob.glob("ablation_results/*.json"))
if not files:
    print(f"No results found in {args.results_dir}/  — run scripts/ablation_run.py first.")
    raise SystemExit(1)

results = []
for f in files:
    with open(f) as fh:
        results.append(json.load(fh))

results.sort(key=lambda r: (r["run_name"] != "BASE", r["run_name"]))

# Markdown table
print(f"\n{'run_name':<25} {'succ_final':>10} {'succ_50k':>8} {'greedy_sr':>10} "
      f"{'L_koop':>8} {'L_recon':>8}  cfg")
print("-" * 95)
for r in results:
    cfg_str = str(r.get("cfg", {})) if r.get("cfg") else "(baseline)"
    print(f"{r['run_name']:<25} {r['succ_final']:>10} {r['succ_50k']:>8} "
          f"{r['greedy_sr']*100:>9.1f}% "
          f"{r.get('koop_final') or 0:>8.4f} {r.get('recon_final') or 0:>8.4f}  {cfg_str}")

plot_ablation_comparison(results, out=args.out)
