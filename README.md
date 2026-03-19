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

where `ZIR` (zero-input response) captures the free evolution `A^k z_0`, and `W_toeplitz` is a precomputed lower-triangular block matrix of `A` powers.  This turns H sequential rollouts into **one cuBLAS GEMM**, making gradient-based MPC fast and numerically clean.

---

## Features

| Feature | Detail |
|---|---|
| Latent dynamics | `z' = Az + Bu`, `A ∈ O(d)` (exact on CUDA via SVD Procrustes) |
| Planner variants | Random shooting, beam search, Gumbel-Softmax, sequential MPC, Block-Toeplitz GEMM |
| Continuous actions | tanh-squashed logits, float64 precision inside planner |
| Exploration | Ornstein-Uhlenbeck noise or i.i.d. Gaussian |
| Training signal | Double-DQN-style TD + optional directed value-iteration graph |
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
# Sequential MPC planner (most stable)
python experiments/pendulum_kgp.py --sequential --seed 0

# Block-Toeplitz GEMM planner
python experiments/pendulum_kgp.py --seed 0

# With Ornstein-Uhlenbeck exploration noise
python experiments/pendulum_kgp.py --ou_noise --seed 0

# Cumulative discounted objective (better at eval, noisier during training)
python experiments/pendulum_kgp.py --cumulative --seed 0
```

Results and live plots are saved to `./viz_pendulum/`.
The best checkpoint (by rolling return over last 20 episodes) is saved alongside the final one.

### GravityBasin (discrete, 2D navigation)

```bash
python scripts/train.py
```

---

## Key Results

### Pendulum-v1 — 5-seed comparison (40k steps, OU noise, 10k warmup)

| Planner | Seeds solved (ret > −300) | Best ret/20 |
|---|---|---|
| Toeplitz GEMM | 4/5 | −180 |
| Sequential MPC | 1/1 (baseline) | −369 |

Toeplitz planner requires **float64 precision** inside the planning loop for numerical stability (see Technical Document).

---

## Repository Structure

```
koopman_rl/
├── koopman_rl/
│   ├── model.py          # KoopmanGradientPlanner, Encoder, ValueNetwork, TargetNetwork
│   ├── planner.py        # All MPC planner variants (discrete + continuous)
│   ├── algorithms.py     # Training loop, directed value iteration
│   ├── config.py         # Typed dataclass config hierarchy
│   ├── env.py            # GravityBasin environment
│   ├── buffer.py         # Replay buffer
│   └── losses.py         # Loss utilities
├── experiments/
│   ├── pendulum_kgp.py   # Pendulum-v1 continuous control experiment
│   └── ...
├── configs/
│   ├── ortho_raw.py      # Best config: ortho_a=True, no encoder normalisation
│   └── ...
├── scripts/
│   └── train.py          # GravityBasin training entry point
└── docs/
    └── TECHNICAL.md      # Theory and implementation deep-dive
```

---

## Configuration

The `pendulum_kgp.py` experiment exposes these flags:

| Flag | Default | Description |
|---|---|---|
| `--sequential` | off | Use sequential MPC instead of Toeplitz |
| `--frozen_b` | off | Detach B from comp. graph during planning |
| `--ou_noise` | off | Ornstein-Uhlenbeck exploration noise |
| `--cumulative` | off | Discounted sum objective (vs terminal-only) |
| `--seed N` | 0 | RNG seed for reproducibility |

Core hyperparameters (top of file):

```python
D            = 32        # latent dimension
GAMMA        = 0.99
LR           = 3e-4
WARMUP       = 10_000    # random steps before planning begins
HORIZON      = 5         # MPC planning horizon
PLAN_ITERS   = 20        # Adam steps per planning call
```

---

## Technical Document

See [`docs/TECHNICAL.md`](docs/TECHNICAL.md) for:
- Koopman operator theory
- SVD Procrustes orthogonal constraint derivation
- Block-Toeplitz GEMM derivation and complexity analysis
- Training losses and optimizer structure
- Numerical stability findings (float64 in planner)
- Full experimental results
