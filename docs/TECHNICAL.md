# Koopman-RL: Technical Reference

A detailed account of the theory, derivations, and implementation choices behind the Koopman Gradient Planner with Block-Toeplitz GEMM and actor-critic continuous control.

---

## Table of Contents

1. [Koopman Operator Theory](#1-koopman-operator-theory)
2. [Linear Latent Dynamics Model](#2-linear-latent-dynamics-model)
3. [Orthogonal A: SVD Procrustes Constraint](#3-orthogonal-a-svd-procrustes-constraint)
4. [Network Architecture](#4-network-architecture)
5. [Training Losses](#5-training-losses)
6. [Planner Variants](#6-planner-variants)
7. [Block-Toeplitz GEMM Derivation](#7-block-toeplitz-gemm-derivation)
8. [Ornstein-Uhlenbeck Exploration](#8-ornstein-uhlenbeck-exploration)
9. [Training Protocol](#9-training-protocol)
10. [Experimental Results](#10-experimental-results)
11. [Design Decisions and Trade-offs](#11-design-decisions-and-trade-offs)

---

## 1. Koopman Operator Theory

### The Basic Idea

Consider a nonlinear dynamical system

$$s_{t+1} = f(s_t, a_t)$$

where $s_t \in \mathbb{R}^n$ is the state and $a_t$ is the action. Nonlinear dynamics are hard to optimise over because gradients through $f$ can vanish or explode, and long-horizon planning requires chaining many nonlinear calls.

The **Koopman operator** is an infinite-dimensional linear operator $\mathcal{K}$ acting on the space of observable functions $g: \mathbb{R}^n \to \mathbb{R}$:

$$(\mathcal{K} g)(s) = g(f(s))$$

The key property: even when $f$ is nonlinear, $\mathcal{K}$ is always **linear**. If we can find a finite set of observables $\{g_1, \ldots, g_d\}$ that is closed under $\mathcal{K}$ (i.e., $\mathcal{K} g_i$ is a linear combination of the $g_j$'s), then the dynamics in observable space are **exactly linear**:

$$z_{t+1} = A z_t, \qquad z_t = [g_1(s_t), \ldots, g_d(s_t)]^\top$$

This converts a nonlinear control problem into a linear one.

### Finite-Dimensional Approximation

In practice, an arbitrary nonlinear system rarely has a finite Koopman-invariant subspace. We instead **learn** an approximate embedding:

$$z = f_\theta(s) \in \mathbb{R}^d$$

where $f_\theta$ is a neural encoder, and jointly learn matrices $A \in \mathbb{R}^{d \times d}$, $B \in \mathbb{R}^{d \times m}$ such that

$$f_\theta(s_{t+1}) \approx A\, f_\theta(s_t) + B\, u_t$$

The pair $(A, B)$ is the finite-dimensional Koopman approximation. The reconstruction loss and Koopman consistency loss jointly push the encoder toward a representation where this approximation is accurate.

### Why Orthogonal A?

Constraining $A \in O(d)$ ($A^\top A = I$) gives several benefits:

1. **Isometry**: $\|Az\| = \|z\|$ for all $z$. The latent norm is preserved by $A$.

2. **Stable powers**: $A^k$ remains exactly orthogonal for all $k$. Critical for the Block-Toeplitz planner.

3. **No normalisation needed**: Allows fully linear dynamics, required for Block-Toeplitz superposition.

4. **Gradient stability**: SVD Procrustes gives a smooth, differentiable map from unconstrained weights to $O(d)$.

5. **Numerical stability at float32**: An orthogonal matrix has condition number 1, so $A^H$ is well-conditioned at any horizon. No float64 is needed in the planner.

---

## 2. Linear Latent Dynamics Model

### Controlled Koopman System

The full model is:

$$z_t = f_\theta(s_t) \qquad \text{(encoder)}$$

$$z_{t+1} = A z_t + B u_t \qquad \text{(latent dynamics)}$$

$$\hat{s}_t = g_\phi(z_t) \qquad \text{(decoder, reconstruction only)}$$

where:
- $z_t \in \mathbb{R}^d$ — latent state ($d = 32$ by default)
- $A \in O(d)$ — shared dynamics matrix, orthogonal
- $B \in \mathbb{R}^{d \times m}$ — action input matrix ($m$ = action dimension)
- $u_t \in \mathbb{R}^m$ — control input (one-hot for discrete; continuous vector for continuous envs)

Value estimation is handled by separate networks (see Section 4).

### dyn_step

All callers use a single dispatch function:

```python
def dyn_step(self, z, b_vec):
    return z @ A.T + b_vec
```

With `ortho_a=True` (always the case for continuous envs), this is simply $z' = Az + b_\text{vec}$.

### Discrete vs Continuous Actions

**Discrete** ($n_\text{actions}$ classes): $u_t$ is a one-hot vector. Greedy action selection:

$$a^* = \arg\max_a \; V_\psi\!\left(Az_t + B_{:,a}\right)$$

**Continuous** ($m$-dimensional torque/force): $u_t \in [-1, 1]^m$ via `π(z_t)` (policy) or via gradient-based MPC over tanh-squashed logits.

---

## 3. Orthogonal A: SVD Procrustes Constraint

### Procrustes Problem

Given an unconstrained weight matrix $W \in \mathbb{R}^{d \times d}$, the nearest orthogonal matrix is the solution to:

$$\min_{A \in O(d)} \|A - W\|_F$$

Closed-form solution via the SVD $W = U \Sigma V^\top$:

$$A^* = U V^\top$$

### Differentiable Parametrisation

Implemented as a PyTorch `parametrize` module:

```python
class _SVDOrthogonal(nn.Module):
    def forward(self, W):
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        return U @ Vh
```

Every time `.weight` is accessed, the SVD Procrustes projection is applied. Gradients flow through `torch.linalg.svd` via autograd.

### Gradient Stability

The SVD backward contains terms $1 / (\sigma_i^2 - \sigma_j^2)$, which blow up when singular values are equal. The fix: initialise $W \sim \mathcal{N}(0,\, 1/d)$. A random Gaussian matrix has generically distinct singular values.

```python
nn.init.normal_(self._A_layer.weight, std=1.0 / (d ** 0.5))
```

**Do not** initialise with identity or an orthogonal matrix — all $\sigma = 1$ gives NaN gradients on the first backward pass.

### Device Dispatch

`torch.linalg.svd` on MPS requires a round-trip to CPU. For MPS and CPU devices we skip the hard constraint and use a **soft penalty** instead:

$$\mathcal{L}_\text{ortho} = \lambda \,\|A^\top A - I\|_F^2$$

On CUDA, the hard SVD Procrustes is used and the soft penalty is never computed.

---

## 4. Network Architecture

### Encoder $f_\theta$

```
Linear(state_dim, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, d)
```

The final linear layer uses **orthogonal initialisation** so initial encodings are well-spread across the output space. Output is raw (no normalisation) when `ortho_a=True`, since sphere normalisation breaks the linear superposition principle.

### Discrete path: Value Network $V_\psi$

```
Linear(d, 64) → ReLU → Linear(64, 1)
```

Maps latent state to a scalar value. Used only in discrete environments (GravityBasin).

### Continuous path: Actor-Critic heads

Three networks are added when `continuous=True`. They receive `z.detach()` — the encoder is shaped by $\mathcal{L}_\text{koop} + \mathcal{L}_\text{recon}$ only; RL losses read the latent without reshaping it.

**Reward predictor** $R_\phi(z, a_\text{norm})$:
```
Linear(d + action_dim, 64) → ReLU → Linear(64, 1)
```
Direct regression target `r / reward_scale`. No bootstrap, no target network needed.

**Q-network** $Q_\psi(z, a_\text{norm})$:
```
Linear(d + action_dim, 64) → ReLU → Linear(64, 1)
```
Scalar Q-value. Trained with Bellman TD using a slow-moving target copy.

**Policy** $\pi_\omega(z)$:
```
Linear(d, 64) → ReLU → Linear(64, action_dim) → Tanh
```
Deterministic actor. Output ∈ $[-1, 1]^m$; multiply by `action_scale` for the environment.

### Target Network

An EMA copy of the online networks. Not an `nn.Module` — excluded from optimiser parameters automatically. Updated after every gradient step:

$$\theta_\text{target} \leftarrow (1 - \tau)\,\theta_\text{target} + \tau\,\theta_\text{online}$$

Default $\tau = 0.005$. $A$ and $B$ are **not** EMA-tracked — the dynamics matrices always reflect the current model state.

- **Discrete path**: tracks `encoder` + `v_net`
- **Continuous path**: tracks `encoder` + `q_net` + `pi_net` (`r_net` is direct regression — no target needed)

### Decoder $g_\phi$

A single `Linear(d, state_dim)` used only for the reconstruction loss. Not used during planning or evaluation.

---

## 5. Training Losses

### Discrete environments

$$\mathcal{L} = \lambda_\text{koop}\,\mathcal{L}_\text{koop} + \lambda_v\,\mathcal{L}_v + \lambda_\text{recon}\,\mathcal{L}_\text{recon} + \mathcal{L}_\text{ortho}$$

**Koopman Consistency** $\mathcal{L}_\text{koop}$:

$$\mathcal{L}_\text{koop} = \mathbb{E}\!\left[\left\|\text{dyn\_step}(z_t,\, B_{:,a}) - \bar{z}_{t+1}\right\|^2 \cdot (1 - \text{done})\right]$$

**Value (Double-DQN TD)** $\mathcal{L}_v$:

$$y = r + \gamma \cdot V_{\psi_\text{target}}\!\left(\bar{z}_{t+1}\right), \qquad \mathcal{L}_v = \mathbb{E}\!\left[\left(V_\psi(z_t\!\;{\color{gray}.\text{detach}()}) - y\right)^2\right]$$

**Reconstruction** $\mathcal{L}_\text{recon} = \mathbb{E}\!\left[\left\|g_\phi(z_t) - s_t\right\|^2\right]$

### Continuous environments

The continuous training uses **two separate backward passes** to prevent the actor loss from contributing gradients to the Q-network via double-counting.

#### World update (one backward)

$$\mathcal{L}_\text{world} = \lambda_\text{koop}\,\mathcal{L}_\text{koop} + \lambda_\text{recon}\,\mathcal{L}_\text{recon} + \mathcal{L}_r + \mathcal{L}_q + \mathcal{L}_\text{ortho}$$

**Koopman** $\mathcal{L}_\text{koop}$: same as discrete; gradient flows through encoder, $A$, $B$.

**Reward predictor** $\mathcal{L}_r$ (no bootstrap):

$$\mathcal{L}_r = \mathbb{E}\!\left[\left(R_\phi(z_t\!\;.\text{detach}(),\; a / \text{scale}) - r / \text{scale}\right)^2\right]$$

**Critic** $\mathcal{L}_q$ (Bellman TD):

$$q_\text{tgt} = r/\text{scale} + \gamma\; Q_{\psi_\text{target}}\!\left(\bar{z}_{t+1},\; \pi_{\omega_\text{target}}(\bar{z}_{t+1})\right) \cdot (1 - \text{terminal})$$

$$\mathcal{L}_q = \mathbb{E}\!\left[\left(Q_\psi(z_t\!\;.\text{detach}(),\; a/\text{scale}) - q_\text{tgt}\right)^2\right]$$

#### Actor update (separate backward)

$$\mathcal{L}_\pi = -\mathbb{E}\!\left[Q_\psi\!\left(z_t\!\;.\text{detach}(),\; \pi_\omega(z_t\!\;.\text{detach}())\right)\right]$$

The Q-network is treated as a **fixed scoring function** during this backward pass (its parameters receive no gradient from $\mathcal{L}_\pi$, only from $\mathcal{L}_q$).

### Optimisers

**Continuous: two separate Adam instances**

```python
# World model: encoder + decoder + r_net + q_net  (NOT pi_net)
opt_world = Adam([
    {"params": world_neural, "lr": 3e-4},
    {"params": koop_params,  "lr": 1.5e-4},   # A, B at half LR
])
# Actor: only pi_net
opt_pi = Adam(agent.pi_net.parameters(), lr=3e-4)
```

Keeping the actor on its own optimiser prevents actor gradients from contaminating the Q-network's Adam moments, and prevents world-model updates from accumulating to the actor's second-moment estimates.

Gradient clipping: `max_norm = 10.0` applied to each optimiser group separately.

---

## 6. Planner Variants

All planners take the current state, encode it, and optimise over an action sequence of length $H$ (horizon).

### Greedy (Discrete)

$$a^* = \arg\max_a \; V_\psi\!\left(Az + B_{:,a}\right)$$

One forward pass; no planning loop.

### Random Shooting / Beam Search (Discrete)

Random shooting samples $N$ random $H$-step sequences and scores by $V_\psi(z_H)$. Beam search keeps the top-$k$ partial sequences at each step. Both unchanged from the original implementation.

### Gumbel-Softmax MPC (Discrete)

Optimise action logits $\Theta \in \mathbb{R}^{H \times |\mathcal{A}|}$ with Adam. `hard=True` forward pass gives strict one-hot actions (no ghost-state blending); STE backward flows gradients through the argmax.

### Direct Policy (Continuous, data collection)

```python
a_t = π(enc(s_t)) * action_scale   # O(d) forward pass
```

Used for all data collection after warmup. Replaces MPC for training-time action selection, making collection $\mathcal{O}(d)$ instead of $\mathcal{O}(H \cdot d \cdot \text{iters})$.

### Sequential MPC (Continuous, benchmark)

Parameterise $u \in \mathbb{R}^{H \times m}$, squash by tanh. Adam for `plan_iters` steps:

```python
for t in range(H):
    z_t = dyn_step(z_t, B @ tanh(u[t]))
# Terminal: Q(z_H, π(z_H)) instead of V(z_H)
a_H  = π(z_H)
loss = -Q(cat([z_H, a_H], dim=-1)).mean()
```

The $Q + \pi$ terminal makes the planner aware of the action-energy penalty accumulated along the path, unlike the old `V(z_H)` which was action-blind.

### Block-Toeplitz GEMM (Continuous, benchmark)

Replaces the sequential rollout with a single GEMM (see Section 7 for derivation). Objective:

$$\mathcal{L}_\text{plan} = -\left[\sum_{t=0}^{H-1} \gamma^t R_\phi(z_t, u_t) + \gamma^H Q_\psi(z_H, \pi_\omega(z_H))\right]$$

The `r_net` path costs provide a dense per-step signal within the horizon; the Q-terminal bootstraps the value beyond it. `cumulative=True/False` is replaced by this principled separation.

---

## 7. Block-Toeplitz GEMM Derivation

### Motivation

The sequential planner requires $H$ sequential calls to `dyn_step`. These cannot be parallelised — autograd graph depth is $\mathcal{O}(H)$. If $A \in O(d)$ and dynamics are exactly linear (no normalisation), we can precompute the entire influence of $A$ on the trajectory.

### Linear Superposition

With $z' = Az + Bu$ (no sphere projection), any trajectory decomposes into:

1. **Zero-Input Response (ZIR)**: trajectory if $u_t \equiv 0$
2. **Zero-State Response**: contribution of actions from $z_0 = 0$

By discrete-time variation of parameters:

$$z_k = A^k z_0 + \sum_{j=0}^{k-1} A^{k-1-j} B u_j$$

### Matrix Form

Write the $H$-step trajectory as a block equation. Let $X_t = B u_t \in \mathbb{R}^d$:

$$\begin{bmatrix} z_1 \\ z_2 \\ \vdots \\ z_H \end{bmatrix} = \underbrace{\begin{bmatrix} A z_0 \\ A^2 z_0 \\ \vdots \\ A^H z_0 \end{bmatrix}}_{\mathbf{Z}_\text{IR}} + \underbrace{\begin{bmatrix} A^0 & 0 & \cdots & 0 \\ A^1 & A^0 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ A^{H-1} & \cdots & A^1 & A^0 \end{bmatrix}}_{\mathbf{W} \in \mathbb{R}^{Hd \times Hd}} \begin{bmatrix} X_0 \\ X_1 \\ \vdots \\ X_{H-1} \end{bmatrix}$$

$\mathbf{W}$ is a **lower-triangular Block-Toeplitz** matrix: $\mathbf{W}_{ij} = A^{i-j}$ for $i \geq j$.

### Implementation

**Precompute** (once, outside Adam loop, `torch.no_grad()`):

```python
A_pows = [torch.eye(d)]
for _ in range(H): A_pows.append(A_pows[-1] @ A)
A_stack = torch.stack(A_pows)  # [H+1, d, d]

ZIR = einsum('kij,nj->nki', A_stack[1:], z0)  # [N, H, d] — batched over N states

row, col  = arange(H).unsqueeze(1), arange(H).unsqueeze(0)
W_blocks  = A_stack[(row - col).clamp(0)] * (row >= col).float()[..., None, None]
W_toeplitz = W_blocks.permute(0,2,1,3).reshape(H*d, H*d)
```

**Adam loop** (`plan_iters` steps):

```python
u_logits = zeros(N, H, m, requires_grad=True)
opt = Adam([u_logits], lr=0.1)

for _ in range(plan_iters):
    u      = tanh(u_logits)                               # [N, H, m]
    X_flat = (u @ B.T).reshape(N, H*d)

    Z = ZIR + (X_flat @ W_toeplitz.T).reshape(N, H, d)   # single GEMM

    # r_net path costs + Q terminal
    ZU        = cat([Z, u], dim=-1)                       # [N, H, d+m]
    disc_path = (gammas_path * r_net(ZU).squeeze(-1)).sum(1)  # [N]
    z_H, a_H  = Z[:, -1], pi_net(Z[:, -1])
    q_H       = q_net(cat([z_H, a_H], -1)).squeeze(-1)   # [N]

    loss = -(disc_path + gamma**H * q_H).mean()
    u_logits.grad, = autograd.grad(loss, u_logits, only_inputs=True)
    opt.step()
```

### Complexity

| Quantity | Sequential planner | Toeplitz planner |
|---|---|---|
| Per-iter forward | $\mathcal{O}(Hd^2)$ sequential matmuls | $\mathcal{O}(H^2 d^2)$ single GEMM |
| Autograd graph depth | $\mathcal{O}(H)$ | $\mathcal{O}(1)$ |
| GPU parallelism | Low (sequential deps) | High (cuBLAS GEMM) |
| Precompute cost | None | $\mathcal{O}(H d^2)$, cached until $A$ changes |

For Pendulum with $H=5$, $d=32$: GEMM is $160 \times 160$, fitting in L1 cache. Precompute is paid once and amortised over `plan_iters` steps. The autograd graph has depth $\mathcal{O}(1)$ since $\mathbf{W}$ and $\mathbf{Z}_\text{IR}$ are precomputed with `torch.no_grad()`.

---

## 8. Ornstein-Uhlenbeck Exploration

### The Problem with i.i.d. Gaussian Noise

White noise gives zero expected net angular impulse — insufficient for building up momentum in a swing-up task.

### Ornstein-Uhlenbeck Process

Mean-reverting stochastic process producing temporally correlated noise:

$$dx = \theta(\mu - x)\,dt + \sigma\,dW$$

Discrete Euler-Maruyama form:

$$x_{t+1} = x_t + \theta(\mu - x_t)\,\Delta t + \sigma\sqrt{\Delta t}\;\xi_t, \qquad \xi_t \sim \mathcal{N}(0, I)$$

Parameters: $\theta = 0.15$, $\mu = 0$, $\sigma = 0.2$, $\Delta t = 0.05$.

`reset()` is called at episode boundaries to prevent noise from carrying across episodes.

---

## 9. Training Protocol

### Pendulum-v1 Configuration

| Parameter | Value |
|---|---|
| Environment | `Pendulum-v1` (Gymnasium) |
| State | $[\cos\theta,\; \sin\theta,\; \dot\theta]$, dim 3 |
| Action | Scalar torque $\in [-2, 2]$, dim 1 |
| Latent dimension $d$ | 32 |
| $A$ constraint | $O(d)$ via SVD Procrustes (hard on CUDA) |
| Encoder output | Raw (no normalisation) |
| Discount $\gamma$ | 0.99 |
| Network LR | $3 \times 10^{-4}$ |
| Koopman LR scale | 0.5 ($A$, $B$ updated at $1.5 \times 10^{-4}$) |
| Batch size | 256 |
| Buffer size | 100,000 |
| Warmup steps | 5,000 (pure random actions) |
| Total steps | 30,000 (default) |
| EMA $\tau$ | 0.005 |
| Exploration noise decay | Linear from 1.0 to 0.1 over 15,000 steps |
| Reward scale | 10.0 (divide rewards before TD) |

### Training Loop

```
for step in 1..N_STEPS:
    if step <= WARMUP:
        action = random_action()
    else:
        action = π(enc(state)) * action_scale
        action += noise * noise_scale   [OU or Gaussian, decaying]

    next_state, reward, done = env.step(action)
    buffer.push(s, a, r, s', done)

    if done: ou_noise.reset()

    if step > WARMUP and buffer.ready():
        batch = buffer.sample(256)
        z_src = encoder(s)
        z_tgt = target.encoder(s')            # stop-grad

        # World backward
        L_koop  = ||dyn_step(z_src, a @ B.T) - z_tgt||² · (1 - done)
        L_recon = ||decoder(z_src) - s||²
        L_r     = ||r_net(z_src.detach(), a_norm) - r/scale||²
        a_next  = target.pi_net(z_tgt)
        q_tgt   = r/scale + γ * target.q_net(z_tgt, a_next) * (1 - terminal)
        L_q     = ||q_net(z_src.detach(), a_norm) - q_tgt||²
        (L_koop + L_recon + L_r + L_q + L_ortho).backward()
        opt_world.step()
        target.update(agent, tau=0.005)

        # Actor backward
        L_pi = -q_net(z_src.detach(), π(z_src.detach()))
        L_pi.backward()
        opt_pi.step()

    every 1000 steps:
        print L_koop, L_r, L_q, L_pi, ret/20
        if ret20 > best_ret: save_checkpoint(...)

# End of training: three-variant benchmark
benchmark("direct policy",    π(enc(s)))
benchmark("sequential MPC",   plan_continuous_batch(Q+π terminal))
benchmark("toeplitz MPC",     plan_toeplitz_batch(r_net path + Q+π terminal))
```

### Best-Model Checkpointing

Saved every 1,000 steps when rolling return over the last 20 episodes improves. Captures peak performance rather than final state, important because Koopman models can be unstable mid-training.

### Logging

Every 1,000 steps:
```
step  5000  noise=0.850  L_koop=0.0124  L_r=0.3217  L_q=0.0891  L_pi=-0.1240
  ret/20= -850.3  ‖AᵀA-I‖²=0.0e+00  sps=1234
  [best] ret/20=-732.4 → checkpoints/pendulum/kgp_pendulum_policy_s0_best.pt
```

---

## 10. Experimental Results

### Pendulum-v1: Pre-actor-critic (V-net baseline, 40k steps)

Config: `ortho_a=True`, OU noise, float64 planner (now removed — see Section 11).

| Run | Planner | Seed | Best ret/20 | Solved? |
|---|---|---|---|---|
| toe_s3 | Toeplitz | 3 | $-180$ | Yes |
| toe_s0 | Toeplitz | 0 | $-293$ | Yes |
| toe_s2 | Toeplitz | 2 | $-352$ | Yes |
| toe_s4 | Toeplitz | 4 | $-362$ | Yes |
| seq_s0 | Sequential | 0 | $-369$ | Yes |
| toe_s1 | Toeplitz | 1 | $-883$ | No |

Success criterion: best rolling return $> -300$.

### GravityBasin: Best Config Results

Config: `ortho_raw` (`ortho_a=True`, `no_normalize=True`). 40k steps.

- 100% greedy success rate
- Mean episode length on success: 20.6 steps

---

## 11. Design Decisions and Trade-offs

### Why a Separate r_net Instead of Q Alone?

The Q-network is trained by bootstrapped TD, which means its targets depend on the current (evolving) policy and value estimates. During early training, these targets are unreliable — `L_q → ∞` was observed in the V-net baseline (0.005 → 4.27 over 50k steps).

`r_net` is trained by **direct regression** against observed rewards. No bootstrap, no target network, no moving target. This decouples stable regression (r_net) from bootstrapped regression (Q), and gives the Toeplitz planner a reliable per-step signal that doesn't depend on Q convergence.

### Why Two Separate Optimisers?

A single optimiser for the world model and actor would cause the actor loss $\mathcal{L}_\pi = -\mathbb{E}[Q(z, \pi(z))]$ to place gradients on Q-network parameters via the backward pass. These gradients would also contaminate `opt_world`'s Adam second-moment estimates for Q, creating an inconsistent training signal: Q is being pulled both toward Bellman targets (correct) and toward maximising $\pi$ outputs (incorrect).

Two separate `Adam` instances with disjoint parameter sets prevent this entirely. `opt_pi` only updates `pi_net`; `opt_world` only updates the world model (encoder, decoder, r_net, q_net, A, B).

### Why No Target Network for r_net?

Target networks exist to stabilise bootstrapped TD — the slow-moving target breaks the feedback loop between prediction and target. For `r_net`, the target is a fixed observed reward $r$, not a network output. There is no feedback loop to stabilise.

### Why Policy-Based Data Collection (Not MPC)?

MPC-based data collection costs $\mathcal{O}(H \cdot d \cdot \text{plan\_iters})$ per step — effectively running the network $H \times \text{plan\_iters}$ times for each environment step. For $H=5$, `plan_iters=10`, $d=32$: that's 50 network forward passes per step.

Direct policy `π(z)` costs one encoder call + one `pi_net` call, an $\mathcal{O}(d)$ operation. With exploration noise layered on top, the exploration-exploitation balance is controlled entirely by the noise schedule — no MPC quality is needed during data collection.

MPC is retained for the **end-of-training benchmark**, where the goal is evaluation rather than speed.

### V(z) Was Blind to Action Cost

The old `V(z_H)` terminal objective in MPC cannot see the $0.001 u^2$ energy penalty accumulated along the trajectory. High-torque and low-torque trajectories that land at the same terminal latent state look identical to `V`. This meant the planner would freely use large torques to reach a high-value terminal state.

`Q(z_H, π(z_H))` evaluates the value of the policy at the terminal state, which has learned to balance reward and action cost. Combined with `r_net` per-step costs in the Toeplitz planner, the full action-energy profile is now visible to MPC.

### Encoder Gradient Isolation

The encoder is updated **only** by $\mathcal{L}_\text{koop} + \mathcal{L}_\text{recon}$. All RL losses (`L_r`, `L_q`, `L_pi`) receive `z.detach()` — the encoder is never updated by value signals. This separates two distinct learning objectives:

- The encoder learns a Koopman-consistent representation.
- The value/policy heads learn to read that representation.

Allowing RL losses to reshape the encoder corrupts the Koopman structure that the planner depends on. The `z.detach()` discipline is enforced at every RL loss computation in the trainer.

### float64 in the Planner (Removed)

Earlier versions used float64 inside the Toeplitz planner to address numerical noise in $\mathbf{W}_\text{Toeplitz} \mathbf{x}$. This was necessary when the terminal objective was `V(z)`, where small perturbations in $z_H$ directly corrupted the value signal.

With the actor-critic upgrade, the terminal is `Q(z_H, π(z_H))`. Both Q and π are neural networks with bounded Lipschitz constants — small perturbations in $z_H$ produce bounded, smooth perturbations in the output. More fundamentally, $A \in O(d)$ has condition number 1 by definition, so $A^H$ is exactly isometric at any horizon. For $H \leq 20$ and $d = 32$, float32 numerical error in the GEMM is well within the noise floor of the neural networks. float64 kills GPU throughput with no benefit, and has been removed.

---

*Updated 2026-03-20.*
