# Koopman-RL Experiment Report: Actor-Critic on Pendulum-v1

*Date: 2026-03-22 — Seed 0, 50,000 steps*

---

## 1. Implementation

### 1.1 Architecture

The agent is a `KoopmanGradientPlanner` with `continuous=True`. The core Koopman structure — encoder, dynamics matrices $A$ and $B$, decoder — is unchanged from the discrete case. Three new heads are added:

- **Encoder** $f_\theta : \mathbb{R}^3 \to \mathbb{R}^{32}$ — `[Linear(3,64), Tanh, Linear(64,64), Tanh, Linear(64,32)]`, orthogonal init on final layer. Output is raw (no normalisation), required for linear superposition in the Toeplitz planner.

- **Dynamics** — $z_{t+1} = Az_t + Bu_t$, with $A \in O(32)$ constrained via soft penalty on CPU/MPS. $B \in \mathbb{R}^{32 \times 1}$, orthogonal init. During training, $u_t = a_t / \text{action\_scale}$ — normalised to $[-1,1]$ before multiplication by $B$, so $B$ is trained in normalised action units. The planner also operates in normalised units, making the two consistent.

- **Reward predictor** $R_\phi(z, a_\text{norm}) \to \mathbb{R}$ — `[Linear(33,64), ReLU, Linear(64,1)]`. Direct regression against observed reward. No bootstrap, no target network.

- **Q-network** $Q_\psi(z, a_\text{norm}) \to \mathbb{R}$ — same architecture as $R_\phi$. Bellman TD with EMA target.

- **Policy** $\pi_\omega(z) \to [-1,1]$ — `[Linear(32,64), ReLU, Linear(64,1), Tanh]`. Deterministic actor, DDPG-style.

- **Decoder** $g_\phi : \mathbb{R}^{32} \to \mathbb{R}^3$ — single linear layer, reconstruction only.

### 1.2 Training Losses

Two backward passes per gradient step:

**World update** (`opt_world`, covers encoder + decoder + $A$ + $B$ + $R_\phi$ + $Q_\psi$):

$$\mathcal{L}_\text{world} = \lambda_\text{koop} \underbrace{\| Az_t + Bu_t - \bar{z}_{t+1} \|^2 \cdot (1-d)}_{\mathcal{L}_\text{koop}} + \lambda_\text{recon} \underbrace{\| g_\phi(z_t) - s_t \|^2}_{\mathcal{L}_\text{recon}} + \underbrace{\| R_\phi(\hat{z}_t, u_t) - r/c \|^2}_{\mathcal{L}_r} + \underbrace{\| Q_\psi(\hat{z}_t, u_t) - q_\text{tgt} \|^2}_{\mathcal{L}_q} + \mathcal{L}_\text{ortho}$$

where $\hat{z}_t = z_t\texttt{.detach()}$ — the encoder receives gradient only from $\mathcal{L}_\text{koop}$ and $\mathcal{L}_\text{recon}$.

The Q target is:
$$q_\text{tgt} = r/c + \gamma \, Q_{\psi^-}(\bar{z}_{t+1},\; \pi_{\omega^-}(\bar{z}_{t+1})) \cdot (1 - \text{terminal})$$

using `terminal` (true episode end) not `done` (which includes truncation), so truncated episodes correctly bootstrap.

**Actor update** (`opt_pi`, covers $\pi_\omega$ only):

$$\mathcal{L}_\pi = -\mathbb{E}\left[ Q_\psi(\hat{z}_t,\; \pi_\omega(\hat{z}_t)) \right]$$

$Q_\psi$ is treated as a fixed scoring function during this backward pass — its parameters receive no gradient from $\mathcal{L}_\pi$.

### 1.3 Planner

The Toeplitz GEMM planner is used for data collection (H=20, iters=50) and for benchmarking. The planning objective is:

$$\max_{u_0,\ldots,u_{H-1}} \sum_{t=0}^{H-1} \gamma^t R_\phi(z_t, u_t) + \gamma^H Q_\psi(z_H, \pi_\omega(z_H))$$

where $u_t = \tanh(\ell_t)$ with logits $\ell \in \mathbb{R}^{H \times 1}$, and the entire trajectory $z_1, \ldots, z_H$ is computed in one batched GEMM:

$$Z = Z_\text{IR} + (X_\text{flat} \cdot W_\text{Toeplitz}^\top).$$

The path rewards use **current** states $z_0, \ldots, z_{H-1}$ paired with their corresponding actions — $r_\text{net}(z_t, u_t)$ — matching the training convention. Logits are initialised at zero at each call, and optimised for `plan_iters` Adam steps (`lr=0.1`). The first action $\tanh(\ell_0) \cdot \text{action\_scale}$ is taken.

$W_\text{Toeplitz}$ and $A^0, \ldots, A^H$ are cached per (horizon, $\gamma$) key and invalidated after each `opt_world.step()`.

### 1.4 Hyperparameters

| Parameter | Value |
|---|---|
| Env | `Pendulum-v1`, state $\in \mathbb{R}^3$, torque $\in [-2,2]$ |
| Latent dim $d$ | 32 |
| Discount $\gamma$ | 0.99 |
| Network LR | $3\times10^{-4}$; $A$,$B$ at $1.5\times10^{-4}$ |
| Batch size | 512 |
| Buffer size | 100,000 |
| Parallel envs | 10 |
| Warmup | 15,000 steps (random actions) |
| Total steps | 50,000 |
| EMA $\tau$ | 0.005 |
| Noise schedule | Linear $1.0 \to 0.05$ over 40,000 steps from end of warmup |
| reward\_scale $c$ | 1.0 |
| Plan horizon $H$ | 20 |
| Plan iters | 50 |
| Plan lr | 0.1 |

---

## 2. Training Dynamics

### 2.1 Phase 1 — Warmup (steps 0–15,000)

Data collection is purely random uniform $\in [-2,2]$. The world model trains on this diverse data. Key observations:

- **$\mathcal{L}_\text{koop}$** drops from ~0.94 to ~0.09. The linear Koopman approximation converges rapidly on diverse data.
- **$\mathcal{L}_r$** drops from ~0.07 to ~0.002. The reward predictor converges within 5k steps — Pendulum reward is smooth and easily approximated.
- **$\mathcal{L}_q$** grows from ~0.07 to ~0.6. Q bootstraps from a policy that is itself learning, so the target is nonstationary. The critic does not converge during warmup.
- **$\mathcal{L}_\pi$** grows from ~1 to ~7. This reflects $-\mathbb{E}[Q]$ increasing in magnitude as Q estimates grow, not a divergence.
- **$\|A^\top A - I\|_F^2$** falls from $2.5 \times 10^{-2}$ to $7.5 \times 10^{-5}$. The soft orthogonal penalty efficiently constrains $A$ on CPU.
- Episode returns during warmup: approximately $-1100$ to $-1350$, consistent with random-action performance on Pendulum-v1.

### 2.2 Phase 2 — Active Data Collection (steps 15,000–31,000)

The planner takes over data collection. Exploration noise decays linearly from 1.0 toward 0.05.

- **$\mathcal{L}_\text{koop}$** continues falling monotonically: 0.075 → 0.023. No creep. This is in contrast to pre-fix behaviour where the raw-action / normalised-action mismatch caused $B$ to be trained at the wrong scale, leading to Koopman error rising once the planner concentrated data near the upright.
- **$\mathcal{L}_r$** remains stable at ~0.002–0.003 throughout.
- **$\mathcal{L}_q$** fluctuates in the range 0.5–0.7, neither converging nor diverging.
- **$\mathcal{L}_\pi$** falls from ~7 back to ~3. As the policy improves and concentrates states near the upright, Q values over those states stabilise, reducing $-\mathbb{E}[Q]$ in magnitude.
- **Episode returns** begin improving around step 22,000 as noise drops below 0.6. Breakthrough episodes occur at steps 23,000 (ret/20 = −797), 25,000 (−524), and 31,000 (−386). The best checkpoint is saved at step 31,000.
- **Steps per second** drop from ~420 (warmup) to ~180–220 once the planner is active, reflecting the O(H × d × iters) cost of 50-iteration MPC at H=20.

### 2.3 Phase 3 — Noise Floor (steps 31,000–50,000)

Exploration noise reaches its floor of 0.1 around step 30,000 and remains there.

- **$\mathcal{L}_\text{koop}$** continues its gradual fall: 0.023 → 0.013. No saturation or rebound.
- **$\mathcal{L}_r$** stable at ~0.002.
- **$\mathcal{L}_q$** fluctuates around 0.5–0.7.
- **$\mathcal{L}_\pi$** rises slowly from 3.0 to 4.3.
- **Episode returns** regress from the peak of −386 back to −877 by step 50,000. The best checkpoint is never improved upon after step 31,000.

---

## 3. Benchmark Results

The benchmark is run against the **best checkpoint** (step 31,000, ret/20 = −386), not the final weights.

| Planner | Mean return | Std | Wall time |
|---|---|---|---|
| Direct policy $\pi(f_\theta(s))$ | **−144.9** | 97.0 | 0.1 s |
| Toeplitz MPC ($R_\phi$ path + $Q$ terminal) | −463.5 | 400.0 | 4.5 s |

Benchmark horizon H=20, 50 Adam iterations, 20 episodes each.

The direct policy achieves −144.9, which is within the range of well-tuned DDPG on Pendulum-v1 (typically −120 to −200). The Toeplitz planner scores −463.5 with substantially higher variance (std 400 vs 97 for the policy).

---

## 4. Observed Behaviours

### 4.1 Policy Dominates the Planner

Despite the planner having a richer objective ($R_\phi$ path costs + $Q$ terminal over H=20 steps), the direct policy outperforms it by a factor of ~3 in mean return and with much lower variance. The policy's std of 97 indicates consistent upright balancing; the planner's std of 400 indicates episodes that sometimes partially succeed and often fail.

### 4.2 Planner Exhibits Energy-Pumping Behaviour

In trajectory visualisations (`policy_vs_planner.png`), the planner produces oscillating torque patterns — alternating pushes that gradually build angular momentum through energy pumping. The policy, by contrast, applies more committed directional torques to swing the pendulum up aggressively before transitioning to a balancing mode.

Energy pumping is a locally optimal strategy for the Koopman linear model: from any pendulum state, a sequence of alternating torques moves the pendulum along a predictable oscillatory path, and the r_net path rewards each step's local improvement. The planner's optimiser converges to this basin because it initialises at zero torque and the nearest gradient ascent direction from a hanging state favours small, reversible perturbations. The aggressive swing-up strategy, which requires committing to a large torque in one direction before the path rewards improve, is a qualitatively different basin that the planner does not find from a zero initialisation in 50 iterations.

### 4.3 Performance Regression After Peak

The best rolling return of −386 is reached at step 31,000 and is not recovered. Returns degrade back to approximately −800 by step 50,000. This pattern is consistent with distributional shift: as the planner learns to keep the pendulum near-upright, the buffer increasingly contains near-upright transitions, and both the world model and Q-function lose accuracy in the swing-up regime that would be needed to recover from disturbances. However, since $\mathcal{L}_\text{koop}$ does not rebound during this phase — it continues falling — the Koopman model itself remains accurate on the collected data. The degradation appears to be in the Q-function and policy rather than the world model.

### 4.4 Planner Variance

The planner's std of 400 over 20 episodes (against the policy's 97) indicates that the planner sometimes succeeds and sometimes fails dramatically in the same evaluation. This high variance is consistent with the planner's sensitivity to initialisation: from some starting states the zero-initialisation logits lead to an energy-pumping trajectory that happens to reach upright; from others it does not. The policy is deterministic (no noise at evaluation) and produces low-variance trajectories across episodes.

### 4.5 World Model Stability After Scale Fix

In earlier runs with the B-scale bug (L_koop trained on raw actions ∈ [−2,2] but planner using normalised inputs ∈ [−1,1]), $\mathcal{L}_\text{koop}$ climbed from ~0.02 to ~0.12 between steps 20k and 50k. After fixing $B$ to be trained in normalised units, $\mathcal{L}_\text{koop}$ falls monotonically for the entire 50,000-step run. The Koopman model does not degrade when the planner concentrates data — it simply becomes very accurate within the near-upright distribution.

### 4.6 Orthogonality of A

$\|A^\top A - I\|_F^2$ falls from $2.5 \times 10^{-2}$ at step 1,000 to $1.2 \times 10^{-5}$ by step 50,000 using only the soft penalty ($\lambda_\text{ortho} = 1.0$) on CPU. The Toeplitz superposition assumption $Z = Z_\text{IR} + W_\text{Toeplitz} X$ becomes increasingly accurate as training progresses.

---

*Generated from run: `python experiments/pendulum_kgp.py --steps 50000 --seed 0`*
