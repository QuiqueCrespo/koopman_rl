"""
Offline Koopman world model on LEWM Reacher (pixel observations).

Data setup (run once):
    pip install huggingface_hub h5py zstandard
    python -c "
    from huggingface_hub import hf_hub_download
    hf_hub_download('quentinll/lewm-reacher', 'reacher.tar.zst', local_dir='data/')
    "
    cd data && tar --zstd -xf reacher.tar.zst

Usage:
    python experiments/reacher_kgp.py --data data/reacher.h5
    python experiments/reacher_kgp.py --data data/reacher.h5 --steps 50000 --device cuda
"""

import argparse
import copy

import torch
from torch.utils.data import DataLoader

from koopman_rl.config import Config, EnvConfig, ModelConfig, AlgoConfig, BufferConfig, TrainConfig
from koopman_rl.model import KoopmanGradientPlanner, TargetNetwork
from koopman_rl.offline_dataset import LEWMDataset
from koopman_rl.trainer_offline import train_offline


def make_reacher_cfg(n_steps: int = 200_000) -> Config:
    return Config(
        env=EnvConfig(
            state_dim=3,          # unused for pixel obs; kept for decoder shape
            n_actions=2,          # Reacher: 2D continuous torque
            action_scale=1.0,
            continuous=True,
            obs_type="pixels",
            img_size=64,
            img_channels=3,
        ),
        model=ModelConfig(
            d=64,
            lr=3e-4,
            ema_tau=0.005,
            ortho_a=True,
        ),
        algo=AlgoConfig(
            gamma=0.99,
            lambda_koop=1.0,
            lambda_ortho=1.0,
            koop_lr_scale=0.5,
            noise_z_std=0.05,
            no_graph=True,
        ),
        buffer=BufferConfig(
            batch_size=256,
        ),
        train=TrainConfig(
            n_steps=n_steps,
            log_every=1_000,
            warmup=0,            # offline: no warmup needed
            ckpt_dir="output/checkpoints/reacher",
        ),
        run_name="reacher_kgp",
        seed=0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to reacher.h5")
    parser.add_argument("--steps",  type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device

    cfg = make_reacher_cfg(n_steps=args.steps)
    cfg.seed = args.seed

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = LEWMDataset(args.data)
    loader  = DataLoader(
        dataset,
        batch_size=cfg.buffer.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device != "cpu"),
        drop_last=True,
    )
    print(f"Dataset: {len(dataset):,} transitions  "
          f"obs={tuple(dataset.obs.shape[1:])}  "
          f"action_dim={dataset.acts.shape[1]}")

    # ── Model ────────────────────────────────────────────────────────────────
    agent  = KoopmanGradientPlanner.from_cfg(cfg, device=device)
    target = TargetNetwork(agent)

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent parameters: {n_params:,}")

    # ── Train ────────────────────────────────────────────────────────────────
    ckpt_path = f"output/checkpoints/reacher/kgp_reacher_s{cfg.seed}.pt"
    train_offline(agent, target, loader, cfg,
                  device=device, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
