"""
Train Sheaf-RL.

Usage:
  python scripts/train.py
  python scripts/train.py --config configs/fast.py
  python scripts/train.py --config configs/base.py --seed 0 --device cpu
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

# Add repo root to path so sheaf_rl package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from sheaf_rl.config import Config
from sheaf_rl.algorithms import train, evaluate
from sheaf_rl.viz import plot_results

parser = argparse.ArgumentParser(description="Train Sheaf-RL")
parser.add_argument("--config", default="configs/base.py",
                    help="Path to config file (default: configs/base.py)")
parser.add_argument("--seed",   type=int,  default=None, help="Override config seed")
parser.add_argument("--device", default=None,            help="Override config device")
args = parser.parse_args()

# Load config from file
spec = importlib.util.spec_from_file_location("cfg_module", args.config)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg: Config = mod.cfg

if args.seed   is not None: cfg.seed   = args.seed
if args.device is not None: cfg.device = args.device

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

print(f"\nRun: {cfg.run_name}  seed={cfg.seed}  device={cfg.device}")

history = train(cfg)
agent   = history["agent"].to(torch.device("cpu"))

sr, ms = evaluate(agent, cfg, n_episodes=100)
print(f"\nFinal evaluation (100 episodes, greedy):")
print(f"  Success rate : {sr*100:.1f}%")
print(f"  Mean steps   : {ms:.1f}  (successful episodes only)")

ep     = history["episode_returns"]
n_succ = sum(r > 0 for r in ep)
print(f"\n  Training episodes    : {len(ep)}")
print(f"  Training successes   : {n_succ}  ({100*n_succ/max(len(ep),1):.1f}%)")

plot_results(history, cfg)
