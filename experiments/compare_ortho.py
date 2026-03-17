"""
SVD Procrustes orthogonalisation — training validation.

Runs ortho_a=True (SVD) and reports:
  - Steps-per-second (throughput)
  - ||AᵀA − I||²_F  over training  (orthogonality error)
  - det(A) to confirm O(d) coverage
  - Greedy success rate at end

Usage:
  python experiments/compare_ortho.py
  python experiments/compare_ortho.py --steps 60000 --seed 0
"""

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from sheaf_rl.config import Config, ModelConfig, AlgoConfig, TrainConfig
from sheaf_rl.model import KoopmanGradientPlanner, TargetNetwork
from sheaf_rl.env import GravityBasin
from sheaf_rl.buffer import ReplayBuffer
from sheaf_rl.algorithms import _resolve_device, evaluate
from sheaf_rl.viz import plot_live


# ---------------------------------------------------------------------------
# Minimal train loop — same as algorithms.train() but also tracks
# ||AᵀA−I||² every log_every steps and returns it alongside losses.
# ---------------------------------------------------------------------------

def run_experiment(cfg: Config, label: str) -> dict:
    device = _resolve_device(cfg.device)
    a, t, m = cfg.algo, cfg.train, cfg.model

    env    = GravityBasin(cfg.env)
    buf    = ReplayBuffer.from_cfg(cfg)
    agent  = KoopmanGradientPlanner.from_cfg(cfg, device=device)
    if a.no_normalize:
        agent.encoder.no_normalize = True
    target = TargetNetwork(agent)

    neural_params = (list(agent.encoder.parameters()) +
                     list(agent.v_net.parameters()) +
                     list(agent.decoder.parameters()))
    koop_params   = agent.koop_parameters()
    opt = optim.Adam([
        {"params": neural_params, "lr": m.lr},
        {"params": koop_params,   "lr": m.lr * a.koop_lr_scale},
    ])
    agent.to(device); target.encoder.to(device); target.v_net.to(device)

    GAMMA        = a.gamma
    LAMBDA_KOOP  = a.lambda_koop
    LAMBDA_RECON = a.lambda_recon
    LAMBDA_V     = a.lambda_v
    LAMBDA_ORTHO = a.lambda_ortho
    BATCH        = cfg.buffer.batch_size
    min_buf      = 2 * a.n_chunks * a.t_chunk + BATCH

    ep_returns, ortho_errs       = [], []
    recent_koop, recent_v, recent_recon = [], [], []
    koop_log, v_log, recon_log          = [], [], []

    state     = env.reset()
    ep_return = 0.0
    ep_steps  = 0
    t0        = time.time()

    print(f"\n{'='*60}")
    print(f"  {label}  |  device={device}  |  hard_ortho={agent._use_hard_ortho}")
    print(f"{'='*60}")

    for step in range(1, t.n_steps + 1):
        eps    = max(t.eps_end, t.eps_start - (t.eps_start - t.eps_end) * step / t.eps_decay)
        action = agent.act(state, epsilon=(eps if step > t.warmup else 1.0))
        ns, reward, done = env.step(state, action)
        buf.push(state, action, reward, ns, done)
        ep_return += reward; ep_steps += 1

        if done or ep_steps >= env.max_ep_steps:
            ep_returns.append(ep_return)
            ep_return, ep_steps = 0.0, 0
            state = env.reset()
        else:
            state = ns

        if step <= t.warmup or not buf.ready(min_buf):
            continue

        batch = buf.sample_transitions(BATCH)
        s_b  = torch.from_numpy(batch["states"]).to(device)
        ns_b = torch.from_numpy(batch["next_s"]).to(device)
        a_b  = torch.from_numpy(batch["actions"]).long().to(device)
        r_b  = torch.from_numpy(batch["rewards"]).to(device)
        d_b  = torch.from_numpy(batch["dones"]).to(device)

        z_src = agent.encode(s_b)
        with torch.no_grad():
            z_dst = target.encoder(ns_b)

        # Koopman loss
        z_pred = agent.dyn_step(z_src, agent.B.T[a_b])
        L_koop = ((z_pred - z_dst.detach()).pow(2)
                   .sum(dim=-1).mul(1.0 - d_b)).mean()
        # Reconstruction
        L_recon = (agent.decoder(z_src) - s_b).pow(2).mean()
        # TD value loss
        with torch.no_grad():
            z_An  = z_dst @ agent.A.detach().T
            B_c   = agent.B.detach().T
            raw   = z_An.unsqueeze(1) + B_c.unsqueeze(0)
            Bs, nA, d_ = raw.shape
            z_na  = raw if agent._ortho_a else \
                    nn.functional.normalize(raw.reshape(Bs*nA, d_), dim=-1).reshape(Bs, nA, d_)
            v_all = agent.v_net(z_na.reshape(Bs*nA, d_)).reshape(Bs, nA)
            best  = v_all.argmax(dim=1)
            v_tgt = target.v_net(z_na.reshape(Bs*nA, d_)).reshape(Bs, nA)
            V_nxt = v_tgt.gather(1, best.unsqueeze(1)).squeeze(1)
            y_td  = r_b + GAMMA * V_nxt * (1.0 - d_b)
        L_v = (agent.v_net(z_src) - y_td).pow(2).mean()

        # Soft ortho penalty only when hard constraint is NOT active (MPS/CPU)
        L_ortho = (agent.ortho_penalty()
                   if (m.ortho_a and not agent._use_hard_ortho)
                   else torch.tensor(0.0, device=device))

        loss = LAMBDA_KOOP*L_koop + LAMBDA_V*L_v + LAMBDA_RECON*L_recon + LAMBDA_ORTHO*L_ortho
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 10.0)
        opt.step()
        target.update(agent, tau=m.ema_tau)

        recent_koop.append(L_koop.item())
        recent_v.append(L_v.item())
        recent_recon.append(L_recon.item())

        if step % t.log_every == 0:
            elapsed = time.time() - t0; t0 = time.time()
            sps     = t.log_every / elapsed
            recent20 = ep_returns[-20:] if ep_returns else []
            succ     = sum(r > 0 for r in recent20)

            err = agent.ortho_error()   # device-safe helper
            ortho_errs.append(err)
            koop_log.append(np.mean(recent_koop))
            v_log.append(np.mean(recent_v))
            recon_log.append(np.mean(recent_recon))
            recent_koop.clear(); recent_v.clear(); recent_recon.clear()

            print(f"  step {step:6d}  ε={eps:.3f}  L_koop={koop_log[-1]:.4f}"
                  f"  L_v={v_log[-1]:.4f}  succ/20={succ}"
                  f"  ||AᵀA-I||²={err:.2e}  sps={sps:.0f}", flush=True)

        if step % t.plot_every == 0:
            plot_live(step, agent, koop_log, v_log, recon_log, ep_returns,
                      graph_v_diff=None, cfg=cfg)

    agent.to(torch.device("cpu"))
    det_a = torch.linalg.det(agent.A.detach()).item()
    err   = agent.ortho_error()
    sr, ms = evaluate(agent, cfg, n_episodes=200)
    print(f"\n  Greedy eval (200 ep): {sr*100:.1f}%  mean_steps={ms:.1f}"
          f"  det(A)={det_a:+.4f}  ||AᵀA-I||²={err:.1e}")

    return {
        "label":       label,
        "ep_returns":  ep_returns,
        "ortho_errs":  ortho_errs,
        "koop_log":    koop_log,
        "v_log":       v_log,
        "sr":          sr,
        "ms":          ms,
    }


# ---------------------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------------------

def _rolling(x, w=20):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_comparison(results: list, out: str = "ortho_comparison.png") -> None:
    colors = ["royalblue", "tomato", "seagreen"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cayley vs Differentiable-SVD Orthogonalisation", fontsize=13)

    # Panel 1: rolling success rate
    ax = axes[0]
    for r, c in zip(results, colors):
        succ = [1.0 if v > 0 else 0.0 for v in r["ep_returns"]]
        w = min(50, len(succ))
        if len(succ) >= w:
            roll = _rolling(succ, w) * 100
            ax.plot(np.arange(w - 1, len(succ)), roll, color=c,
                    linewidth=1.5, label=f"{r['label']}  ({r['sr']*100:.0f}% final)")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("Success rate (%)"); ax.set_ylim(0, 108)
    ax.set_title("Learning Curves"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: ||AᵀA − I||²_F over log intervals
    ax = axes[1]
    for r, c in zip(results, colors):
        ax.semilogy(r["ortho_errs"], color=c, linewidth=1.5, label=r["label"])
    ax.set_xlabel("Log interval"); ax.set_ylabel(r"$\|A^\top A - I\|_F^2$")
    ax.set_title("Orthogonality Error"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 3: Koopman loss
    ax = axes[2]
    for r, c in zip(results, colors):
        ax.semilogy(r["koop_log"], color=c, linewidth=1.5, label=r["label"])
    ax.set_xlabel("Log interval"); ax.set_ylabel("L_koop")
    ax.set_title("Koopman Loss"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int,   default=60_000)
    parser.add_argument("--warmup", type=int,   default=10_000)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    cfg = Config(
        run_name="ortho_svd",
        model=ModelConfig(ortho_a=True),
        algo=AlgoConfig(no_normalize=True),
        train=TrainConfig(
            n_steps=args.steps,
            warmup=args.warmup,
            log_every=5_000,
            plot_every=5_000,
            eps_end=0.15,
            eps_decay=int(args.steps * 0.7),
        ),
    )
    if args.device:
        cfg.device = args.device

    result = run_experiment(cfg, "SVD Procrustes")
    plot_comparison([result])
