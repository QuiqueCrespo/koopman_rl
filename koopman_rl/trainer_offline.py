"""
Offline Koopman trainer — no environment, no rewards.

Trains the dynamics A, B and visual encoder via the Koopman consistency
loss:  L_koop = ‖A z_t + B u_t − z̄_{t+1}‖²
where z̄_{t+1} comes from an EMA target encoder (prevents collapse).

No R_φ, no Q, no actor — world model only.
"""

import os
import time
import copy
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from koopman_rl.model import KoopmanGradientPlanner, TargetNetwork
from koopman_rl.config import Config


def train_offline(
    agent:   KoopmanGradientPlanner,
    target:  TargetNetwork,
    loader:  DataLoader,
    cfg:     Config,
    device:  str = "cpu",
    ckpt_path: str = "output/checkpoints/reacher/kgp_reacher.pt",
) -> None:
    """
    Offline training loop.

    Args:
        agent:      model being trained
        target:     EMA copy of agent (inference-only)
        loader:     DataLoader yielding (obs_t, obs_{t+1}, action_t)
        cfg:        experiment config
        device:     torch device string
        ckpt_path:  where to save the checkpoint
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── Unpack config ─────────────────────────────────────────────────────────
    d             = cfg.model.d
    lr            = cfg.model.lr
    ema_tau       = cfg.model.ema_tau
    koop_lr_scale = cfg.algo.koop_lr_scale
    lambda_koop   = cfg.algo.lambda_koop
    lambda_ortho  = cfg.algo.lambda_ortho
    action_scale  = cfg.env.action_scale
    n_steps       = cfg.train.n_steps
    log_every     = cfg.train.log_every
    batch_size    = cfg.buffer.batch_size

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    # ── Optimiser — encoder + A + B only ─────────────────────────────────────
    koop_params = agent.koop_parameters()   # A, B
    opt = optim.Adam([
        {"params": list(agent.encoder.parameters()), "lr": lr},
        {"params": koop_params,                       "lr": lr * koop_lr_scale},
    ])

    agent.to(device)
    target.encoder.to(device)
    target.encoder.eval()
    agent.train()

    # Pre-allocated noise buffer (avoids repeated cuRAND allocs on GPU)
    _z_noise = (torch.empty(batch_size, d, device=device)
                if cfg.algo.noise_z_std > 0 else None)

    # Infinite iterator over the offline dataset
    def _inf(dl):
        while True:
            yield from dl
    data_iter = _inf(loader)

    print("=" * 60)
    print(f"  Offline Koopman training — {n_steps:,} steps")
    print(f"  device={device}  d={d}  batch={batch_size}")
    print(f"  lr={lr}  ema_tau={ema_tau}  noise_z_std={cfg.algo.noise_z_std}")
    print("=" * 60)

    t0 = time.time()
    loss_acc = 0.0

    for step in range(1, n_steps + 1):
        obs_t, obs_t1, a = next(data_iter)
        obs_t  = obs_t.to(device)
        obs_t1 = obs_t1.to(device)
        a      = a.to(device)

        z_src  = agent.encoder(obs_t)
        a_norm = a / action_scale

        with torch.no_grad():
            z_dst = target.encoder(obs_t1)

        # Koopman consistency loss
        z_src_dyn = z_src
        if _z_noise is not None:
            z_src_dyn = z_src + _z_noise.normal_(0, cfg.algo.noise_z_std)
        z_pred = agent.dyn_step(z_src_dyn, a_norm @ agent.B.T)
        L_koop = (z_pred - z_dst.detach()).pow(2).mean()

        # Soft ortho penalty (CPU/MPS; zero on CUDA with hard SVD)
        L_ortho = (agent.ortho_penalty()
                   if (agent._ortho_a and not agent._use_hard_ortho)
                   else torch.tensor(0.0, device=device))

        loss = lambda_koop * L_koop + lambda_ortho * L_ortho

        opt.zero_grad()
        loss.backward()
        opt.step()

        # EMA target update
        tau = ema_tau
        with torch.no_grad():
            for p, tp in zip(agent.encoder.parameters(),
                             target.encoder.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)

        loss_acc += loss.item()

        if step % log_every == 0:
            elapsed = time.time() - t0
            sps = log_every / elapsed
            print(f"  step {step:>8,}  L_koop={L_koop.item():.4f}"
                  f"  L_ortho={L_ortho.item():.4f}"
                  f"  loss={loss_acc/log_every:.4f}"
                  f"  {sps:.1f} sps")
            loss_acc = 0.0
            t0 = time.time()

    # Save checkpoint
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "target_encoder":   target.encoder.state_dict(),
        "config":           cfg.to_dict(),
    }, ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")
