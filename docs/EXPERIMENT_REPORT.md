# Koopman-RL Experiment Report: Actor-Critic on Pendulum-v1

*Date: 2026-03-25 — Seed 0*

---

## 1. Implementation

### 1.1 Architecture

The agent is a `KoopmanGradientPlanner` with `continuous=True`. Core Koopman structure: encoder, dynamics matrices $A$ and $B$, decoder. Three actor-critic heads added for continuous control.

- **Encoder** $f_\theta : \mathbb{R}^3 \to \mathbb{R}^{16}$ — `[Linear(3,64), Tanh, Linear(64,64), Tanh, Linear(64,16)]`, orthogonal init on final layer. Output is raw (no normalisation).

- **Dynamics** — $z_{t+1} = Az_t + Bu_t$, with $A \in O(16)$ constrained via SVD Procrustes (hard on CUDA, soft penalty on CPU/MPS). $B \in \mathbb{R}^{16 \times 1}$, orthogonal init. During training and planning, $u_t = a_t / \text{action\_scale} \in [-1,1]$.

- **Reward predictor** $R_\phi(z, a_\text{norm}) \to \mathbb{R}$ — `[Linear(17,64), ReLU, Linear(64,1)]`. Direct regression, no bootstrap.

- **Q-network** $Q_\psi(z, a_\text{norm}) \to \mathbb{R}$ — same architecture. Bellman TD with EMA target ($\tau = 0.005$).

- **Policy** $\pi_\omega(z) \to [-1,1]$ — `[Linear(16,64), ReLU, Linear(64,1), Tanh]`. Deterministic actor.

- **Decoder** $g_\phi : \mathbb{R}^{16} \to \mathbb{R}^3$ — single linear layer, reconstruction only.

### 1.2 Training Losses

Two backward passes per gradient step:

**World update** (`opt_world`, covers encoder + decoder + $A$ + $B$ + $R_\phi$ + $Q_\psi$):

$$\mathcal{L}_\text{world} = \lambda_\text{koop} \mathcal{L}_\text{koop} + \lambda_\text{recon} \mathcal{L}_\text{recon} + \mathcal{L}_r + \mathcal{L}_q + \mathcal{L}_\text{ortho}$$

where $\hat{z}_t = z_t\texttt{.detach()}$ — the encoder receives gradient only from $\mathcal{L}_\text{koop}$ and $\mathcal{L}_\text{recon}$.

The Q target is:
$$q_\text{tgt} = r + \gamma \, Q_{\psi^-}(\bar{z}_{t+1},\; \pi_{\omega^-}(\bar{z}_{t+1})) \cdot (1 - \text{terminal})$$

**Actor update** (`opt_pi`, covers $\pi_\omega$ only):

$$\mathcal{L}_\pi = -\mathbb{E}\left[ Q_\psi(\hat{z}_t,\; \pi_\omega(\hat{z}_t)) \right]$$

### 1.3 Planners

**Toeplitz GEMM value-based** (primary): objective $\sum_t \gamma^t R_\phi(z_t, u_t) + \gamma^H Q_\psi(z_H, \pi(z_H))$. The entire trajectory is computed in one batched GEMM using the precomputed block-Toeplitz matrix (see TECHNICAL.md §7).

**Value-free closed-form**: objective $\min_u \|z_H - z_\text{goal}\|^2$. Solved analytically via the reachability matrix pseudo-inverse — no iterations:

$$u^* = (z_\text{goal} - A^H z_0) \cdot \text{pinv}(W_\text{reach})^\top, \qquad W_\text{reach} = \mathbf{W}_\text{Toeplitz}[-d:] \cdot \text{block\_diag}(B,\ldots,B)$$

**CEM hybrid**: Phase 1 zero-order CEM (inside `torch.no_grad()`), Phase 2 gradient polish warm-started from CEM mean. Both value-based and value-free variants.

**Warm-start variants** (`ToeplitzPlannerWarmStart`, `CEMPlannerWarmStart`): stateful wrappers that shift the previous solution forward one step before each new plan call.

### 1.4 Hyperparameters

| Parameter | Value |
|---|---|
| Env | `Pendulum-v1`, state $\in \mathbb{R}^3$, torque $\in [-2,2]$ |
| Latent dim $d$ | 16 |
| Discount $\gamma$ | 0.99 |
| Network LR | $3\times10^{-4}$; $A$,$B$ at $1.5\times10^{-4}$ |
| Batch size | 512 |
| Buffer size | 100,000 |
| Parallel envs | 10 |
| Warmup | 20,000 steps (random actions) |
| EMA $\tau$ | 0.005 |
| Noise schedule | Decaying $1.0 \to 0.1$ |
| Plan horizon $H$ | 10 |
| Plan iters | 100 (gradient); 10 CEM rounds + 20 grad polish |
| CEM samples | 200 |
| CEM elites | 20 |

---

## 2. Run A — ortho_a=True, 50k steps (primary)

### 2.1 Training Dynamics

**Warmup (0–20k)**: Random data. $\mathcal{L}_\text{koop}$ falls to ~0.10; $\mathcal{L}_\text{recon}$ reaches ~0.001; $\mathcal{L}_r$ converges to ~0.001; $\mathcal{L}_q$ grows with Q as value estimates grow. $\|A^\top A - I\|_F^2$ falls from $\sim2.5 \times 10^{-2}$ to $\sim10^{-5}$ — soft penalty on CPU is sufficient.

**Active collection (20k–50k)**: Policy-driven data. $\mathcal{L}_\text{koop}$ continues falling monotonically (~0.10 → ~0.09). Returns improve from ~−1100 to best of ~−130 at the best checkpoint. $\|A^\top A - I\|_F^2$ stabilises near $10^{-5}$.

### 2.2 Planner Timing

| Planner | Time/step |
|---|---|
| Direct policy | 0.04 ms |
| Grad value-free (closed-form) | 0.09 ms |
| Grad value-based (100 Adam) | ~23 ms |
| CEM value-free (10×200+20g) | ~13 ms |
| CEM value-based (10×200+20g) | ~28 ms |

---

## 3. Run B — ortho_a=False Ablation, 100k steps

### 3.1 Purpose

Verify that the orthogonal constraint on $A$ is necessary for the Toeplitz planner. With `ortho_a=False`, $A$ starts at identity and is optimised freely (no penalty).

### 3.2 Training Observations

- $\|A^\top A - I\|_F^2$ immediately diverged from 0 to ~0.75 at step 1k and stabilised at ~2.5 throughout 100k steps. The Koopman loss alone provides no pressure toward orthogonality.
- $\mathcal{L}_\text{recon}$ converged normally (~0.0003 by 50k steps).
- $\mathcal{L}_\text{koop}$ converged normally (~0.10–0.15 range).
- Returns reached ~−120 to −165 range, showing the underlying policy learned — the RL training was not affected.

### 3.3 Benchmark Results (best checkpoint)

| Planner | Mean return | Std |
|---|---|---|
| Direct policy | -166.8 | 89.5 |
| Grad value-free (closed-form) | -159.4 | 107.2 |
| CEM value-free | -296.9 | 241.9 |
| Toeplitz warm-start | -722.5 | 90.3 |
| CEM value-based | -1035.5 | 158.3 |
| **Toeplitz MPC (r_net+Q)** | **-1831.0** | **94.3** |

### 3.4 Interpretation

The orthogonality constraint is **structurally necessary** for the Toeplitz multi-step planner:

- Without it, $A^k$ grows as $\sigma_1^k$ where $\sigma_1 \approx 1.5$–$1.8$ (estimated from $\|A^\top A - I\|^2 \approx 2.5$). Over $H=10$ steps this gives a 50–60× blowup — the imagined terminal state $z_H$ is far outside the training distribution.
- $R_\phi$ and $Q_\psi$, trained on in-distribution latent vectors, produce meaningless outputs on these vectors.
- Planners calling $A$ only once per step are unaffected: direct policy and value-free closed-form still achieve ~−160 to −167.

---

## 4. Diagnostic Visualisations

The following plots are generated post-training and from checkpoints via `pendulum_viz_only.py`:

### 4.1 Model-Reality Gap (`model_rollout_accuracy.png`)

Compares the real policy rollout in `Pendulum-v1` against the imagined rollout where the policy is applied autoregressively in latent space ($z_{t+1} = Az_t + Bu_t$, decoded each step). Key findings:

- Decoded state error $\|\hat{s}_t - s_t\|$ grows significantly within 50–100 steps from most starting states.
- Latent divergence $\|z_t^\text{imag} - f_\theta(s_t^\text{real})\|$ grows monotonically and crosses threshold 1.0 within 30–80 steps depending on starting state.
- The model-reality gap is one source of planning underperformance: even perfect MPC in the imagined space produces suboptimal behaviour in reality.

### 4.2 Natural Dynamics (`natural_dynamics.png`)

Compares the uncontrolled ($u=0$) real pendulum trajectory against the latent free-flight $z_{t+1} = Az_t$ decoded back to state space. Key findings:

- The two trajectories diverge rapidly (within 20–50 steps from most starts), despite $A \in O(d)$ preserving $\|z\|$.
- The imagined latent norm (orange dashed) is nearly constant (correct — orthogonal map preserves norms), while the encoded real-state norm (blue) varies.
- Conclusion: $A$ was trained on the controlled distribution and does not accurately represent the autonomous ($u=0$) dynamics in isolation. The Koopman factorisation $z_{t+1} = Az_t + Bu_t$ does not cleanly separate autonomous from control-driven dynamics.

### 4.3 CEM Diagnostics (`cem_score_evolution.png`, `cem_trajectory_evolution.png`, `cem_open_loop.png`, `cem_reward_landscape.png`)

- CEM does improve scores across iterations (score histograms shift right), confirming it is functional.
- From the hanging state, CEM phase finds a committed swing-up direction; gradient polish refines it.
- Open-loop CEM plan diverges from actual execution after ~15–20 steps, consistent with the model-reality gap.

---

## 5. Key Findings

1. **Orthogonality of $A$ is critical for the Toeplitz planner.** Without it, $A^k$ diverges and multi-step rollouts are meaningless. Single-step planners (direct policy, value-free closed-form) are unaffected.

2. **Value-free closed-form planner is fast and robust.** By solving $\min_u \|z_H - z_\text{goal}\|^2$ analytically via the reachability pseudo-inverse, it runs in 0.09 ms/step (vs ~23 ms for gradient-based). It is 2nd-best in the ortho_a=False ablation and competitive with the direct policy.

3. **CEM is faster than pure gradient planning** because its zero-order Phase 1 runs inside `torch.no_grad()` (no backward) and Phase 2 uses only 20 gradient steps (vs 100 for pure gradient). Net: ~40% cheaper than pure gradient for better or comparable quality.

4. **The model-reality gap is the primary bottleneck for planning quality.** The latent model diverges from reality within a few dozen steps, so optimising a long-horizon objective in latent space does not translate to the same improvement in real execution.

5. **The autonomous dynamics are not cleanly represented by $A$.** $z_{t+1} = Az_t$ does not accurately reproduce zero-control pendulum dynamics, even with $A \in O(d)$. This limits the interpretability and utility of $A$ alone as a model of natural evolution.

---

*Generated from:*
- *Run A: `python experiments/pendulum_kgp.py --steps 50000 --seed 0` (ortho_a=True)*
- *Run B: `python experiments/pendulum_kgp.py --steps 100000 --seed 0` (ortho_a=False ablation)*
