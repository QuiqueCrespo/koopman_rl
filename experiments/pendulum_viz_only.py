"""
Load a saved pendulum checkpoint and regenerate all visualizations.
Usage: python experiments/pendulum_viz_only.py [--ckpt path/to/ckpt.pt]
"""
import sys, argparse, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.pendulum_kgp import (
    make_pendulum_cfg, plot_final_summary, plot_plan_evolution,
    plot_plan_convergence, plot_policy_vs_planner, plot_koopman_latent_analysis,
    plot_cem_diagnostics, plot_model_rollout_accuracy, plot_natural_dynamics,
)
from koopman_rl.model import KoopmanGradientPlanner, TargetNetwork
from koopman_rl.buffer import ContinuousReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="output/checkpoints/pendulum/kgp_pendulum_policy_s0_best.pt")
parser.add_argument("--device", default="auto")
args_raw = parser.parse_args()

# Build a minimal args namespace that make_pendulum_cfg expects
import argparse as _ap
args = _ap.Namespace(sequential=False, frozen_b=False, ou_noise=False,
                     seed=0, steps=50000, device=args_raw.device)
cfg = make_pendulum_cfg(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
         if args_raw.device == "auto" else torch.device(args_raw.device)

ckpt = torch.load(args_raw.ckpt, map_location=device, weights_only=False)

agent = KoopmanGradientPlanner.from_cfg(cfg, device=device)
agent.load_state_dict(ckpt["agent_state_dict"])
agent.eval()

target = TargetNetwork(agent)
target.encoder.load_state_dict(ckpt["target_encoder"])
target.q_net.load_state_dict(ckpt["target_q_net"])
target.pi_net.load_state_dict(ckpt["target_pi_net"])
target.encoder.to(device).eval()
target.q_net.to(device).eval()
target.pi_net.to(device).eval()

episode_returns = ckpt.get("episode_returns", [])

# Dummy buffer (not needed for most plots)
buf = ContinuousReplayBuffer(1000, cfg.env.state_dim, cfg.env.n_actions)

print(f"Loaded: {args_raw.ckpt}")
print(f"device={device}  d={cfg.model.d}  H={cfg.planner.horizon}  iters={cfg.planner.plan_iters}")

plot_final_summary(agent, episode_returns, buf, cfg=cfg)
plot_plan_evolution(agent, cfg)
plot_plan_convergence(agent, cfg)
plot_model_rollout_accuracy(agent, cfg)
plot_natural_dynamics(agent, cfg)
plot_cem_diagnostics(agent, cfg)
plot_policy_vs_planner(agent, cfg)
