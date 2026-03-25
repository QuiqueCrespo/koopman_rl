# Koopman-RL: Technical Reference

A detailed account of the theory, derivations, and implementation choices behind the Koopman Gradient Planner with Block-Toeplitz GEMM, CEM hybrid planner, and actor-critic continuous control.

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

2. **Stable powers**: $A^k$ remains exactly orthogonal for all $k$. Critical for the Block-Toeplitz planner — without orthogonality, $A^k$ diverges exponentially if any singular value exceeds 1.

3. **No normalisation needed**: Allows fully linear dynamics, required for Block-Toeplitz superposition.

4. **Gradient stability**: SVD Procrustes gives a smooth, differentiable map from unconstrained weights to $O(d)$.

5. **Numerical stability at float32**: An orthogonal matrix has condition number 1, so $A^H$ is well-conditioned at any horizon. No float64 is needed in the planner.

**Ablation evidence**: removing the orthogonality constraint causes $\|A^\top A - I\|_F^2$ to stabilise at $\approx 2.5$ within the first few thousand steps. The Toeplitz planner then degrades to mean return $-1831$ (from $\approx -165$ with constraint), because $A^k$ diverges and the imagined $H$-step trajectory is meaningless. See Section 10.

---

## 2. Linear Latent Dynamics Model

### Controlled Koopman System

The full model is:

$$z_t = f_\theta(s_t) \qquad \text{(encoder)}$$

$$z_{t+1} = A z_t + B u_t \qquad \text{(latent dynamics)}$$

$$\hat{s}_t = g_\phi(z_t) \qquad \text{(decoder, reconstruction only)}$$

where:
- $z_t \in \mathbb{R}^d$ — latent state ($d = 16$)
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

With `ortho_a=True`, this is simply $z' = Az + b_\text{vec}$.

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

### Block-Toeplitz GEMM Value-Based (Continuous, primary)

Replaces the sequential rollout with a single GEMM (see Section 7 for derivation). Objective:

$$\max_{u_0,\ldots,u_{H-1}} \sum_{t=0}^{H-1} \gamma^t R_\phi(z_t, u_t) + \gamma^H Q_\psi(z_H, \pi_\omega(z_H))$$

The `r_net` path costs provide a dense per-step signal within the horizon; the Q-terminal bootstraps the value beyond it. Solved by `plan_iters` Adam steps on tanh-squashed logits $\ell \in \mathbb{R}^{H \times m}$.

### Value-Free Closed-Form (Continuous)

**Objective**: $\min_{u} \|z_H - z_\text{goal}\|^2$, where $z_\text{goal} = f_\theta(s_\text{goal})$ is the encoding of the target state.

**Key insight**: unrolling the Koopman recursion to horizon $H$ gives a linear relationship between the action sequence and the terminal state:

$$z_H = \underbrace{A^H z_0}_{z_H^\text{free}} + W_\text{reach}\, u_\text{flat}$$

where the **reachability matrix** $W_\text{reach} \in \mathbb{R}^{d \times Hm}$ is the last $d$ rows of the Toeplitz matrix composed with $B$:

$$W_\text{reach} = \mathbf{W}_\text{Toeplitz}[-d:]\;\cdot\;\text{block\_diag}(B,\ldots,B)$$

This is a linear least-squares problem with **closed-form solution** (no Adam iterations required):

$$u^* = \underbrace{(z_\text{goal} - A^H z_0)}_\text{residual at horizon} \cdot \text{pinv}(W_\text{reach})^\top$$

The pseudo-inverse is computed once per call. Since $d=16$ and $H \cdot m = 10$ for Pendulum, $W_\text{reach}$ is $16 \times 10$ (overdetermined — the system cannot exactly reach all goal states in $H=10$ steps with 1-dim torque; the pseudo-inverse gives the minimum-norm torque sequence that minimises residual distance).

**Timing**: 0.09 ms/step vs ~11 ms for gradient-based (100 Adam iterations). The closed-form exploits linearity fully — no iterations are possible or needed once the model is linear.

### CEM Hybrid Planner (Continuous)

Two-phase planner combining zero-order search with gradient polish:

**Phase 1 — CEM (zero-order, inside `torch.no_grad()`)**: Sample $S$ action sequences from $\mathcal{N}(\mu, \sigma^2)$, score each by:
$$J_s = \sum_{t=0}^{H-1} \gamma^t R_\phi(z_t^s, u_t^s) + \gamma^H Q_\psi(z_H^s, \pi_\omega(z_H^s))$$
Update $\mu \leftarrow$ mean of top-$k$ elites, $\sigma \leftarrow$ std of top-$k$ elites. Repeat for `cem_iters` rounds.

**Phase 2 — Gradient polish**: Warm-start logits at $\mu_\text{final}$ from CEM, then run `grad_iters` Adam steps on the same Toeplitz objective.

**Why CEM finds what gradient descent misses**: from the hanging equilibrium, the gradient of $J$ w.r.t. $u$ is nearly zero (flat energy landscape). Gradient descent from $u=0$ is trapped. CEM samples globally — with high probability some sample commits to a large swing-up torque, which CEM identifies as an elite and shifts $\mu$ toward. Gradient polish then refines within this discovered basin.

**Why CEM is faster than pure gradient**: Phase 1 runs inside `torch.no_grad()` (no backward pass). Phase 2 uses only `grad_iters=20` steps vs 100 for pure gradient. The batched GEMM over $S$ samples in Phase 1 is cheap relative to one backward pass.

### Warm-Start Variants (Stateful)

`CEMPlannerWarmStart` and `ToeplitzPlannerWarmStart` are stateful wrappers that implement **receding-horizon warm-starting**:

After each timestep, the committed action $u_0$ is discarded and the remaining solution $[u_1, \ldots, u_H]$ is shifted forward by one step, zero-padded:

$$\mu_\text{new} = [u_1, u_2, \ldots, u_{H-1}, 0]$$

This warm-starts the next planning call with the previous solution's tail rather than zero-initialising. At evaluation time, both wrappers call `.reset()` at episode boundaries to discard stale warm-start state.

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

### Reachability Matrix

The terminal state $z_H$ depends only on the last block row of $\mathbf{W}$:

$$z_H = A^H z_0 + W_\text{reach}\, u_\text{flat}, \qquad W_\text{reach} = \mathbf{W}[-d:] \cdot \text{block\_diag}(B, \ldots, B)$$

$W_\text{reach} \in \mathbb{R}^{d \times Hm}$ is the **reachability matrix** for horizon $H$. If $d \leq Hm$ and $W_\text{reach}$ has full row rank, the system is reachable (any terminal latent state can be reached exactly). For Pendulum ($d=16$, $H=10$, $m=1$): $Hm=10 < d=16$ — underdetermined in terms of degrees of freedom, so exact reachability is not guaranteed; the pseudo-inverse gives the minimum-norm least-squares solution.

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

**Value-free closed-form**:

```python
# W_reach: last d rows of W_toeplitz times block_diag(B,...,B)
IB      = block_diag(*[B] * H)                     # [H*d, H*m]
W_reach = W_toeplitz[-d:] @ IB                     # [d, H*m]

# z_H_free = A^H z_0
ZIR_H   = einsum('ij,nj->ni', A_stack[H], z0)      # [N, d]

# Closed-form OLS
residual = z_goal.expand(N, -1) - ZIR_H            # [N, d]
u_flat   = residual @ pinv(W_reach).T               # [N, H*m]
u0       = u_flat.reshape(N, H, m)[:, 0, :].clamp(-scale, scale)
```

### Complexity

| Quantity | Sequential planner | Toeplitz planner |
|---|---|---|
| Per-iter forward | $\mathcal{O}(Hd^2)$ sequential matmuls | $\mathcal{O}(H^2 d^2)$ single GEMM |
| Autograd graph depth | $\mathcal{O}(H)$ | $\mathcal{O}(1)$ |
| GPU parallelism | Low (sequential deps) | High (cuBLAS GEMM) |
| Precompute cost | None | $\mathcal{O}(H d^2)$, cached until $A$ changes |
| Value-free | N/A | $\mathcal{O}(1)$ iterations (pinv, closed-form) |

For Pendulum with $H=10$, $d=16$: GEMM is $160 \times 160$. Precompute is paid once and amortised over `plan_iters` steps. The autograd graph has depth $\mathcal{O}(1)$ since $\mathbf{W}$ and $\mathbf{Z}_\text{IR}$ are precomputed with `torch.no_grad()`.

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
| Latent dimension $d$ | 16 |
| $A$ constraint | $O(d)$ via SVD Procrustes (hard on CUDA, soft penalty on CPU/MPS) |
| Encoder output | Raw (no normalisation) |
| Discount $\gamma$ | 0.99 |
| Network LR | $3 \times 10^{-4}$ |
| Koopman LR scale | 0.5 ($A$, $B$ updated at $1.5 \times 10^{-4}$) |
| Batch size | 512 |
| Buffer size | 100,000 |
| Warmup steps | 20,000 (pure random actions) |
| Total steps | 50,000–100,000 |
| EMA $\tau$ | 0.005 |
| Exploration noise | Decaying Gaussian, $1.0 \to 0.1$ |
| Reward scale | 1.0 |
| Plan horizon $H$ | 10 |
| Plan iters | 100 (gradient-based); 10 CEM rounds + 20 grad polish |
| CEM samples $S$ | 200 |
| CEM elites $k$ | 20 |

### Training Loop

```
for step in 1..N_STEPS:
    if step <= WARMUP:
        action = random_action()
    else:
        action = π(enc(state)) * action_scale
        action += noise * noise_scale   [Gaussian, decaying]

    next_state, reward, done = env.step(action)
    buffer.push(s, a, r, s', done)

    if step > WARMUP and buffer.ready():
        batch = buffer.sample(512)
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
        agent.invalidate_toeplitz_cache()

        # Actor backward
        L_pi = -q_net(z_src.detach(), π(z_src.detach()))
        L_pi.backward()
        opt_pi.step()

# End of training: multi-planner benchmark (20 episodes each)
benchmark("direct policy",         π(enc(s)))
benchmark("toeplitz MPC",          act_plan_continuous(...))
benchmark("grad value-free",       _value_free_act_batch(...))   # closed-form
benchmark("toeplitz warm-start",   ToeplitzPlannerWarmStart(...))
benchmark("CEM value-based",       CEMPlannerWarmStart(...))
benchmark("CEM value-free",        CEMPlannerWarmStart(..., objective="value_free"))
```

### Best-Model Checkpointing

Saved every 1,000 steps when rolling return over the last 20 episodes improves. Captures peak performance rather than final state. The benchmark at the end of training loads this best checkpoint before evaluation.

---

## 10. Experimental Results

### Planner Timing (Pendulum-v1, CPU, d=16, H=10)

| Planner | Time/step |
|---|---|
| Direct policy $\pi$ | 0.04 ms |
| Grad value-free (closed-form) | 0.09 ms |
| Grad value-based (100 Adam iters) | ~23 ms |
| CEM value-free (10×200 + 20 grad) | ~13 ms |
| CEM value-based (10×200 + 20 grad) | ~28 ms |

The closed-form value-free planner is ~250× faster than the gradient-based value-based planner, as it has zero Adam iterations.

### Orthogonality Ablation (Pendulum-v1, 100k steps, seed 0)

Running with `ortho_a=False` (unconstrained $A$, no penalty):

| Planner | Mean return (ortho_a=False) |
|---|---|
| Direct policy | -166.8 |
| Grad value-free (closed-form) | -159.4 |
| CEM value-free | -296.9 |
| Toeplitz warm-start | -722.5 |
| CEM value-based | -1035.5 |
| **Toeplitz MPC (r_net+Q)** | **-1831.0** |

$\|A^\top A - I\|_F^2$ stabilised at $\approx 2.5$ (vs $\leq 10^{-5}$ with constraint). The Toeplitz planner catastrophically fails because $A^k$ powers diverge, making imagined trajectories meaningless. Planners that call $A$ only once per step (policy, closed-form value-free via pseudo-inverse) are unaffected.

### GravityBasin: Best Config Results

Config: `ortho_a=True`, `no_normalize=True`. 40k steps.

- 100% greedy success rate
- Mean episode length on success: 20.6 steps

---

## 11. Design Decisions and Trade-offs

### Why a Separate r_net Instead of Q Alone?

The Q-network is trained by bootstrapped TD, which means its targets depend on the current (evolving) policy and value estimates. During early training, these targets are unreliable.

`r_net` is trained by **direct regression** against observed rewards. No bootstrap, no target network, no moving target. This decouples stable regression (r_net) from bootstrapped regression (Q), and gives the Toeplitz planner a reliable per-step signal that doesn't depend on Q convergence.

### Why Two Separate Optimisers?

A single optimiser for the world model and actor would cause the actor loss $\mathcal{L}_\pi = -\mathbb{E}[Q(z, \pi(z))]$ to place gradients on Q-network parameters via the backward pass. Two separate `Adam` instances with disjoint parameter sets prevent this entirely.

### Why No Target Network for r_net?

Target networks exist to stabilise bootstrapped TD. For `r_net`, the target is a fixed observed reward $r$. There is no feedback loop to stabilise.

### Why Orthogonality Is Necessary for the Toeplitz Planner

The Toeplitz planner precomputes $A^0, \ldots, A^H$ and uses them to build $\mathbf{W}_\text{Toeplitz}$ and $W_\text{reach}$. If $A$ is not orthogonal, its largest singular value $\sigma_1 > 1$ causes $\|A^k\| \sim \sigma_1^k$ to grow exponentially. For $H=10$ and $\sigma_1 = 1.5$: $\sigma_1^{10} \approx 57$ — the imagined state $z_H$ is 57× larger than the real state. $R_\phi$ and $Q_\psi$, trained on states near the training distribution, produce garbage outputs on these out-of-distribution latent vectors.

The direct policy and closed-form value-free planner are unaffected because they call $A$ at most once (through `dyn_step`), so a single factor of $\sigma_1$ does not cause catastrophic blowup.

### The Natural Dynamics Gap

Visualising the uncontrolled ($u=0$) system reveals that $z_{t+1} = Az_t$ does **not** accurately track the real zero-control pendulum trajectory. The latent divergence $\|z_t^\text{imag} - f_\theta(s_t^\text{real})\|$ grows monotonically from the first few steps.

This is expected: $A$ is trained on the full controlled distribution, not the zero-control distribution specifically. The Koopman factorisation $z_{t+1} = Az_t + Bu_t$ does not cleanly separate autonomous dynamics from control-driven dynamics unless training specifically enforces this. In practice, $A$ encodes a mixture of dynamics that depends on the action distribution seen during training.

This gap is one contributor to planning underperformance: imagined trajectories diverge from reality even when the controller applies actions close to the training distribution.

### Why Policy-Based Data Collection (Not MPC)?

MPC-based data collection costs $\mathcal{O}(H \cdot d \cdot \text{plan\_iters})$ per step. Direct policy `π(z)` costs one encoder call + one `pi_net` call. With decaying Gaussian exploration, the policy provides sufficient coverage. MPC is retained for the end-of-training benchmark.

### Encoder Gradient Isolation

The encoder is updated **only** by $\mathcal{L}_\text{koop} + \mathcal{L}_\text{recon}$. All RL losses (`L_r`, `L_q`, `L_pi`) receive `z.detach()`. Allowing RL losses to reshape the encoder corrupts the Koopman structure that the planner depends on.

### float64 in the Planner (Removed)

With $A \in O(d)$, the condition number of $A^H$ is exactly 1 at any horizon. Float32 numerical error in the GEMM is well within the noise floor of the neural networks. Float64 kills GPU throughput with no benefit.

---

*Updated 2026-03-25.*
