"""
Ablation study runner — uses typed Config instead of raw dicts.

Usage examples:
  python scripts/ablation_run.py --run-name BASE
  python scripts/ablation_run.py --run-name A1_no_graph     --no-graph
  python scripts/ablation_run.py --run-name B1_no_bisim     --k-bisim 0
  python scripts/ablation_run.py --run-name C1_no_recon     --lambda-recon 0.0
  python scripts/ablation_run.py --run-name H2_k10          --k-diffuse 10
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from sheaf_rl.config import Config
from sheaf_rl.algorithms import train, evaluate

parser = argparse.ArgumentParser(description="Sheaf-RL ablation runner")
parser.add_argument("--run-name",      required=True,        help="Identifier for this run")
parser.add_argument("--seed",          type=int, default=42)
parser.add_argument("--results-dir",   default="results/ablation", help="Output directory")

# Ablation flags
parser.add_argument("--no-graph",      action="store_true",  help="Disable graph VI → pure TD")
parser.add_argument("--td-plus-vi",    action="store_true",  help="L_v = L_v_local + L_v_global")
parser.add_argument("--k-bisim",       type=int,   default=None, help="k_bisim_nn (0=disabled)")
parser.add_argument("--bisim-penalty", type=float, default=None, help="bisim_penalty_scale")
parser.add_argument("--k-diffuse",     type=int,   default=None, help="k_diffuse steps")
parser.add_argument("--lambda-recon",  type=float, default=None, help="lambda_recon (0=disabled)")
parser.add_argument("--lambda-v",      type=float, default=None, help="lambda_v")
parser.add_argument("--fix-A",         action="store_true",  help="Freeze A=I (not trained)")
parser.add_argument("--no-stratified", action="store_true",  help="Uniform random chunk sampling")
parser.add_argument("--no-force-goal", action="store_true",  help="Disable forced goal chunk")
parser.add_argument("--n-chunks",      type=int,   default=None, help="n_chunks (graph size)")
parser.add_argument("--t-chunk",       type=int,   default=None, help="t_chunk (chunk length)")
parser.add_argument("--latent-dim",    type=int,   default=None, help="Latent dimension d")
parser.add_argument("--no-normalize",  action="store_true",  help="Disable F.normalize in encoder")
parser.add_argument("--eval-planner",  action="store_true",  help="Run planner comparison after training")
parser.add_argument("--plan-iters",    type=int, default=10,  help="Adam steps per planner action")
parser.add_argument("--plan-horizon",  type=int, default=10,  help="Planner rollout horizon")

args = parser.parse_args()

# Build flat overrides dict (same key names as old ablation_runner.py)
overrides = {}
if args.no_graph:                    overrides["NO_GRAPH"]            = True
if args.td_plus_vi:                  overrides["TD_PLUS_VI"]          = True
if args.k_bisim       is not None:   overrides["K_BISIM_NN"]          = args.k_bisim
if args.bisim_penalty is not None:   overrides["BISIM_PENALTY_SCALE"] = args.bisim_penalty
if args.k_diffuse     is not None:   overrides["K_DIFFUSE"]           = args.k_diffuse
if args.lambda_recon  is not None:   overrides["LAMBDA_RECON"]        = args.lambda_recon
if args.lambda_v      is not None:   overrides["LAMBDA_V"]            = args.lambda_v
if args.fix_A:                       overrides["FIX_A"]               = True
if args.no_stratified:               overrides["STRATIFIED"]          = False
if args.no_force_goal:               overrides["FORCE_GOAL"]          = False
if args.n_chunks      is not None:   overrides["N_CHUNKS"]            = args.n_chunks
if args.t_chunk       is not None:   overrides["T_CHUNK"]             = args.t_chunk
if args.latent_dim    is not None:   overrides["D"]                   = args.latent_dim
if args.no_normalize:                overrides["NO_NORMALIZE"]        = True

cfg = Config.from_ablation_dict(overrides)
cfg.seed     = args.seed
cfg.run_name = args.run_name

print(f"\n{'='*60}")
print(f"  Ablation run: {args.run_name}")
print(f"  Config overrides: {overrides if overrides else '(baseline)'}")
print(f"{'='*60}\n")

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)

history = train(cfg)
agent   = history["agent"].to(torch.device("cpu"))

sr, ms = evaluate(agent, cfg, n_episodes=100)
print(f"\nGreedy evaluation (100 ep): success={sr*100:.1f}%  mean_steps={ms:.1f}")

ep = history["episode_returns"]

def succ_window(ep_list, ep_idx, window=20):
    chunk = ep_list[max(0, ep_idx - window): ep_idx]
    return sum(r > 0 for r in chunk) if chunk else 0

ep_50k = min(50_000 // 60, len(ep))

summary = {
    "run_name":        args.run_name,
    "cfg":             overrides,
    "seed":            args.seed,
    "succ_final":      succ_window(ep, len(ep)),
    "succ_50k":        succ_window(ep, ep_50k),
    "greedy_sr":       round(sr, 4),
    "greedy_ms":       round(ms, 1) if ms == ms else None,
    "koop_final":      round(history["koop_losses"][-1],  6) if history["koop_losses"]  else None,
    "v_loss_final":    round(history["v_losses"][-1],     6) if history["v_losses"]     else None,
    "recon_final":     round(history["bisim_losses"][-1], 6) if history["bisim_losses"] else None,
    "episode_returns": ep,
}

os.makedirs(args.results_dir, exist_ok=True)
out_path = f"{args.results_dir}/{args.run_name}.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSaved → {out_path}")
print(f"  succ_final={summary['succ_final']}/20  "
      f"succ_50k={summary['succ_50k']}/20  "
      f"greedy={sr*100:.1f}%")

if args.eval_planner:
    from sheaf_rl.algorithms import evaluate_planner
    from sheaf_rl.viz import plot_planner_comparison

    model_path = f"{args.results_dir}/{args.run_name}_model.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"\nSaved model → {model_path}")

    print(f"\n{'='*50}")
    print(f"  Planner comparison  (horizon={args.plan_horizon}, iters={args.plan_iters})")
    print(f"{'='*50}")
    plan_results = evaluate_planner(agent, cfg, n_episodes=50,
                                    horizon=args.plan_horizon,
                                    plan_iters=args.plan_iters)
    for mode, vals in plan_results.items():
        succ = sum(v > 0 for v in vals)
        print(f"  {mode:22s}: {succ}/50 successes  "
              f"mean_return={np.mean(vals):.3f}")
    summary["planner"] = {
        m: {"succ": sum(v > 0 for v in vals), "mean_return": round(float(np.mean(vals)), 4)}
        for m, vals in plan_results.items()
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    plot_planner_comparison(plan_results)
