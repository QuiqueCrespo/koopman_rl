# Koopman-RL

Reinforcement learning with **linear latent dynamics** enforced by the Koopman operator.
A learned encoder maps states to a latent space where a single matrix `A ∈ O(d)` governs all dynamics.
Planning reduces to optimising over a sequence of action vectors, with the entire horizon computed as one dense matrix multiply.

---

## Overview

The key insight is that if `A` is orthogonal and dynamics are linear (`z' = Az + Bu`), an H-step latent trajectory is a closed-form affine function of the action sequence:

```
Z = ZIR + W_toeplitz @ X
```

where `ZIR` (zero-input response) captures the free evolution `A^k z_0`, and `W_toeplitz` is a precomputed lower-triangular block matrix of `A` powers. This turns H sequential rollouts into **one cuBLAS GEMM**, making gradient-based MPC fast and numerically clean.

For continuous control, the model uses a **DDPG-style actor-critic** on top of the Koopman latent space: a direct policy `π(z)` for fast data collection, a reward predictor `r_net(z, a)` trained by direct regression, and a Q-network `Q(z, a)` with Bellman TD. MPC planners (sequential and Toeplitz) use `Q(z_H, π(z_H))` as their terminal value, replacing the previous `V(z_H)` which was blind to the action-energy penalty.

---

## Features

| Feature | Detail |
|---|---|
| Latent dynamics | `z' = Az + Bu`, `A ∈ O(d)` (exact on CUDA via SVD Procrustes) |
| Planner variants | Random shooting, beam search, Gumbel-Softmax, sequential MPC, Block-Toeplitz GEMM |
| Continuous actor-critic | `r_net(z,a)` + `Q(z,a)` + `π(z)` — DDPG-style on Koopman latent space |
| Data collection | Direct policy `π(z)` + decaying Gaussian/OU noise (O(d) vs O(H·d·iters) for MPC) |
| MPC terminal value | `Q(z_H, π(z_H))` — action-aware; Toeplitz planner adds r_net discounted path costs |
| Exploration | Ornstein-Uhlenbeck noise or i.i.d. Gaussian |
| Checkpointing | Best-model save by rolling return (peak, not final) |

---

## Installation

```bash
git clone <repo>
cd koopman_rl
pip install torch gymnasium matplotlib numpy
```

No other dependencies required.

---

## Quick Start

### Pendulum-v1 (continuous control)

```bash
# Default: policy data collection + Toeplitz/sequential benchmark at the end
python experiments/pendulum_kgp.py --steps 30000 --seed 0

# With Ornstein-Uhlenbeck exploration noise
python experiments/pendulum_kgp.py --ou_noise --seed 0

# Run longer for a fully converged agent
python experiments/pendulum_kgp.py --steps 100000 --seed 0
```

A three-variant **benchmark runs automatically at the end of training**:

```
================================================================
  Planner benchmark — 20 episodes each
================================================================
  direct policy           mean=  -XXX  std=  XX  wall=0.1s
  sequential MPC (R+Q)    mean=  -XXX  std=  XX  wall=0.6s
  toeplitz MPC (r_net+Q)  mean=  -XXX  std=  XX  wall=0.6s
```

Results and live plots are saved to `output/viz/pendulum/`.
The best checkpoint (by rolling return over last 20 episodes) is saved alongside the final one.

### GravityBasin (discrete, 2D navigation)

```bash
python scripts/train.py
```

---

## Key Results

### Pendulum-v1 — Toeplitz GEMM (pre-actor-critic upgrade, 40k steps, OU noise)

| Planner | Seeds solved (ret > −300) | Best ret/20 |
|---|---|---|
| Toeplitz GEMM | 4/5 | −180 |
| Sequential MPC | 1/1 (baseline) | −369 |

The current actor-critic upgrade (r_net + Q + π) addresses the structural weakness of the old `V(z)` head, which could not see the action-energy penalty accumulated along the path. See the Technical Document for details.

### GravityBasin: Best Config Results

Config: `ortho_raw` (`ortho_a=True`, `no_normalize=True`). 40k steps.

- 100% greedy success rate
- Mean episode length on success: 20.6 steps

---

## Repository Structure

```
koopman_rl/
├── koopman_rl/
│   ├── model.py              # KoopmanGradientPlanner, Encoder, r_net/Q/π heads, TargetNetwork
│   ├── planner.py            # All MPC planner variants (discrete + continuous)
│   ├── trainer_continuous.py # Continuous training loop (actor-critic)
│   ├── config.py             # Typed dataclass config hierarchy
│   ├── checkpoint.py         # Save/load helpers
│   ├── env.py                # GravityBasin environment
│   ├── buffer.py             # ContinuousReplayBuffer
│   └── noise.py              # OUNoise
├── experiments/
│   ├── pendulum_kgp.py       # Pendulum-v1 continuous control experiment
│   └── ...
├── scripts/
│   └── train.py              # GravityBasin training entry point
└── docs/
    └── TECHNICAL.md          # Theory and implementation deep-dive
```

---

## Configuration

The `pendulum_kgp.py` experiment exposes these flags:

| Flag | Default | Description |
|---|---|---|
| `--ou_noise` | off | Ornstein-Uhlenbeck exploration noise |
| `--frozen_b` | off | Detach B from comp. graph in benchmark MPC |
| `--sequential` | off | Legacy: used for benchmark labelling only |
| `--seed N` | 0 | RNG seed for reproducibility |
| `--steps N` | 30000 | Total environment steps |
| `--device` | auto | `cpu` / `cuda` / `mps` |

Core hyperparameters (set via `Config` in `make_pendulum_cfg`):

```python
d            = 32        # latent dimension
gamma        = 0.99
lr           = 3e-4
warmup       = 5_000     # random steps before policy kicks in
horizon      = 5         # MPC planning horizon (benchmark only)
plan_iters   = 10        # Adam steps per MPC call (benchmark only)
```

---

## Technical Document

See [`docs/TECHNICAL.md`](docs/TECHNICAL.md) for:
- Koopman operator theory
- SVD Procrustes orthogonal constraint derivation
- Block-Toeplitz GEMM derivation and complexity analysis
- Actor-critic architecture: r_net, Q-network, policy π
- Training losses and optimizer structure
- Full experimental results
