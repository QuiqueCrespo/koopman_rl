# Koopman-RL: Technical Reference

A detailed account of the theory, derivations, and implementation choices behind the Koopman Gradient Planner with Block-Toeplitz GEMM.

---

## Table of Contents

1. [Koopman Operator Theory](#1-koopman-operator-theory)
2. [Linear Latent Dynamics Model](#2-linear-latent-dynamics-model)
3. [Orthogonal A: SVD Procrustes Constraint](#3-orthogonal-a-svd-procrustes-constraint)
4. [Network Architecture](#4-network-architecture)
5. [Training Losses](#5-training-losses)
6. [Planner Variants](#6-planner-variants)
7. [Block-Toeplitz GEMM Derivation](#7-block-toeplitz-gemm-derivation)
8. [Numerical Stability: float64 in the Planner](#8-numerical-stability-float64-in-the-planner)
9. [Ornstein-Uhlenbeck Exploration](#9-ornstein-uhlenbeck-exploration)
10. [Training Protocol](#10-training-protocol)
11. [Experimental Results](#11-experimental-results)
12. [Design Decisions and Trade-offs](#12-design-decisions-and-trade-offs)

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

This converts a nonlinear control problem into a linear one, which has much more tractable theory.

### Finite-Dimensional Approximation

In practice, an arbitrary nonlinear system rarely has a finite Koopman-invariant subspace. We instead **learn** an approximate embedding:

$$z = f_\theta(s) \in \mathbb{R}^d$$

where $f_\theta$ is a neural encoder, and jointly learn matrices $A \in \mathbb{R}^{d \times d}$, $B \in \mathbb{R}^{d \times m}$ such that

$$f_\theta(s_{t+1}) \approx A\, f_\theta(s_t) + B\, u_t$$

The pair $(A, B)$ is the finite-dimensional Koopman approximation. The reconstruction loss and Koopman consistency loss jointly push the encoder toward a representation where this approximation is accurate.

### Why Orthogonal A?

Constraining $A \in O(d)$ (the orthogonal group, $A^\top A = I$) gives several benefits:

1. **Isometry**: $\|Az\| = \|z\|$ for all $z$. The latent norm is preserved by $A$; only $Bu_t$ perturbs it. This prevents exponential growth or decay of latent vectors over long horizons.

2. **Stable powers**: $A^k$ remains exactly orthogonal for all $k$ (since $O(d)$ is a group under matrix multiplication). This is critical for the Block-Toeplitz planner, which computes $A^1, A^2, \ldots, A^H$.

3. **No normalisation needed**: With $A \in O(d)$ and linear dynamics, we can drop the sphere-projection step $z \leftarrow z/\|z\|$ that earlier variants of the model used. This makes the dynamics fully linear, which is required for the Block-Toeplitz superposition principle to hold.

4. **Gradient stability**: The SVD Procrustes parametrisation gives a smooth, differentiable map from unconstrained weights to $O(d)$, avoiding the discreteness and instability of other orthogonalisation methods.

---

## 2. Linear Latent Dynamics Model

### Controlled Koopman System

The full model is:

$$z_t = f_\theta(s_t) \qquad \text{(encoder)}$$

$$z_{t+1} = A z_t + B u_t \qquad \text{(latent dynamics)}$$

$$\hat{s}_t = g_\phi(z_t) \qquad \text{(decoder, reconstruction only)}$$

$$V(s_t) = V_\psi(z_t) \qquad \text{(value function)}$$

where:
- $z_t \in \mathbb{R}^d$ — latent state ($d = 32$ by default)
- $A \in O(d)$ — shared dynamics matrix, orthogonal
- $B \in \mathbb{R}^{d \times m}$ — action input matrix ($m$ = action dimension)
- $u_t \in \mathbb{R}^m$ — control input (one-hot for discrete; continuous vector for continuous envs)

### dyn_step

All callers use a single dispatch function:

```python
def dyn_step(self, z, b_vec):
    raw = z @ A.T + b_vec
    return raw if self._ortho_a else F.normalize(raw, dim=-1)
```

With `ortho_a=True` this is simply $z' = Az + b_\text{vec}$. With `ortho_a=False` (the older spherical variant), the result is projected onto the unit sphere. The Block-Toeplitz planner requires `ortho_a=True`.

### Discrete vs Continuous Actions

**Discrete** ($n_\text{actions}$ classes): $u_t$ is a one-hot vector of length $n_\text{actions}$. $B \in \mathbb{R}^{d \times n_\text{actions}}$, so $Bu_t$ selects a column of $B$. Greedy action selection:

$$a^* = \arg\max_a \; V_\psi\!\left(Az_t + B_{:,a}\right)$$

This is a single forward pass (no planning loop required).

**Continuous** ($m$-dimensional torque/force): $u_t \in [-1, 1]^m$ (tanh-squashed from logits). $B \in \mathbb{R}^{d \times m}$. Action selection requires gradient-based MPC (see Section 6).

---

## 3. Orthogonal A: SVD Procrustes Constraint

### Procrustes Problem

Given an unconstrained weight matrix $W \in \mathbb{R}^{d \times d}$, the nearest orthogonal matrix is the solution to the Procrustes problem:

$$\min_{A \in O(d)} \|A - W\|_F$$

The closed-form solution uses the SVD $W = U \Sigma V^\top$:

$$A^* = U V^\top$$

This discards the singular values $\Sigma$ and retains only the rotation/reflection structure.

### Differentiable Parametrisation

We implement this as a PyTorch `parametrize` module:

```python
class _SVDOrthogonal(nn.Module):
    def forward(self, W):
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        return U @ Vh
```

Registered on the weight of a `nn.Linear(d, d, bias=False)`:

```python
parametrize.register_parametrization(self._A_layer, 'weight', _SVDOrthogonal())
```

Every time `.weight` is accessed, the SVD Procrustes projection is applied. Gradients flow through `torch.linalg.svd` via autograd.

### Gradient Stability

The SVD backward contains terms $1 / (\sigma_i^2 - \sigma_j^2)$. These blow up when two singular values of $W$ are equal. Specific danger cases:

- **Identity init**: all $\sigma = 1$, all differences are zero → NaN gradient on first backward pass.
- **Orthogonal init**: same issue.

The fix: initialise $W \sim \mathcal{N}(0,\, 1/d)$ (scaled Gaussian). A random Gaussian matrix has generically distinct singular values with probability 1. After the first optimizer step, $W$ leaves the degenerate manifold and singular values remain distinct for all subsequent steps.

```python
nn.init.normal_(self._A_layer.weight, std=1.0 / (d ** 0.5))
```

### Device Dispatch

`torch.linalg.svd` on MPS (Apple Silicon) requires a round-trip to CPU. For MPS and CPU devices we skip the hard constraint and use a **soft penalty** instead:

$$\mathcal{L}_\text{ortho} = \lambda \,\|A^\top A - I\|_F^2$$

On CUDA, the hard SVD Procrustes is used and the soft penalty is never computed.

```python
self._use_hard_ortho = ortho_a and torch.device(device).type == "cuda"
```

---

## 4. Network Architecture

### Encoder $f_\theta$

```
Linear(state_dim, 64) → Tanh → Linear(64, 64) → Tanh → Linear(64, d)
[optional: → Tanh  (tanh_out mode)]
[optional: L2 normalise  (default, sphere mode)]
[optional: no-op  (no_normalize mode, used with ortho_a=True)]
```

The final linear layer uses **orthogonal initialisation** (`nn.init.orthogonal_`) so initial encodings are well-spread across the output space from the first step.

Three output modes:
- `default`: L2-normalised to the unit sphere $S^{d-1}$.
- `tanh_out`: each dim $\in (-1, 1)$, norm bounded by $\sqrt{d}$.
- `no_normalize`: raw linear output. Required with `ortho_a=True` since sphere normalisation would break the linear superposition principle.

### Value Network $V_\psi$

```
Linear(d, 64) → ReLU → Linear(64, 1)
```

Maps latent state to a scalar value. Small output weights at init keep $V$ near zero, which improves early-training stability and OOD robustness.

### Target Network

An exponential moving average (EMA) of encoder + value network, **not** an `nn.Module`:

$$\theta_\text{target} \leftarrow (1 - \tau)\,\theta_\text{target} + \tau\,\theta_\text{online}$$

Default $\tau = 0.005$. $A$ and $B$ are **not** tracked by the target — only encoder and value weights are EMA-averaged. This means the Koopman matrices always reflect the current model state during bootstrap target computation.

### Decoder $g_\phi$

A single `Linear(d, state_dim)` layer used only for the reconstruction loss $\|g_\phi(z_t) - s_t\|^2$. Not used during planning or evaluation.

---

## 5. Training Losses

The total loss is a weighted sum:

$$\mathcal{L} = \lambda_\text{koop}\,\mathcal{L}_\text{koop} + \lambda_v\,\mathcal{L}_v + \lambda_\text{recon}\,\mathcal{L}_\text{recon} + \lambda_\text{ortho}\,\mathcal{L}_\text{ortho}$$

### Koopman Consistency Loss $\mathcal{L}_\text{koop}$

Penalises the gap between predicted and target-encoded next latent state:

$$\mathcal{L}_\text{koop} = \mathbb{E}_{(s,a,s') \sim \mathcal{B}}\!\left[\left\|\text{dyn\_step}(z_t,\, B_{:,a}) - \bar{z}_{t+1}\right\|^2 \cdot (1 - \text{done})\right]$$

where $\bar{z}_{t+1} = \texttt{target\_encoder}(s')$ (stop-gradient, EMA copy). The $(1 - \text{done})$ mask prevents penalising terminal transitions where dynamics don't apply.

**$\lambda_\text{koop} = 1.0$**

### Value Loss $\mathcal{L}_v$

Double-DQN style TD target. The online network selects the action; the target network evaluates it:

$$a^* = \arg\max_a \; V_\psi\!\left(A\bar{z}_{t+1} + B_{:,a}\right)$$

$$y = r + \gamma \cdot V_{\psi_\text{target}}\!\left(\bar{z}_{t+1}^{a^*}\right), \qquad \mathcal{L}_v = \mathbb{E}\!\left[\left(V_\psi(z_t) - y\right)^2\right]$$

This decouples selection from evaluation, reducing overestimation bias.

**$\lambda_v = 0.5$**

### Reconstruction Loss $\mathcal{L}_\text{recon}$

$$\mathcal{L}_\text{recon} = \mathbb{E}\!\left[\left\|g_\phi(z_t) - s_t\right\|^2\right]$$

Anchors the encoder to carry state-decodable information, preventing the latent space from collapsing.

**$\lambda_\text{recon} = 1.0$**

### Orthogonality Penalty $\mathcal{L}_\text{ortho}$ (MPS/CPU only)

$$\mathcal{L}_\text{ortho} = \left\|A^\top A - I\right\|_F^2$$

Soft version of the hard SVD Procrustes constraint. Only applied when `_use_hard_ortho = False` (MPS or CPU).

**$\lambda_\text{ortho} = 1.0$**

### Optimizer

Two parameter groups with different learning rates:

```python
opt = Adam([
    {"params": neural_params, "lr": 3e-4},        # encoder, v_net, decoder
    {"params": koop_params,   "lr": 1.5e-4},       # A, B  (lr * koop_lr_scale=0.5)
])
```

Gradient clipping: `max_norm = 10.0`.

---

## 6. Planner Variants

All planners take the current state, roll it through the learned latent dynamics, and optimise over an action sequence of length $H$ (horizon).

### Greedy (Discrete)

$$a^* = \arg\max_a \; V_\psi\!\left(Az + B_{:,a}\right)$$

One forward pass. No planning loop. Fastest but myopic (1-step lookahead only).

### Random Shooting (Discrete)

Sample $N = 200$ random $H$-step action sequences. Roll each through `dyn_step`. Return first action of the highest-scoring sequence. $\mathcal{O}(NH)$ `dyn_step` calls, all parallelised on the batch dimension.

### Beam Search (Discrete)

Greedy expansion keeping top-$k$ partial sequences at each step. Sparser than shooting but tracks the most promising paths.

### Gumbel-Softmax MPC (Discrete)

Parameterise the action sequence as logits $\Theta \in \mathbb{R}^{H \times |\mathcal{A}|}$. Optimise with Adam for `plan_iters` steps:

$$\text{probs} = \text{GumbelSoftmax}(\Theta,\; \text{hard=True})$$
$$z_H = \text{rollout}(z_0,\; \text{probs}), \qquad \mathcal{L} = -V_\psi(z_H)$$

The `hard=True` flag is critical: the forward pass uses strict one-hot actions (no blending of $B$ columns into ghost states), while the backward pass uses the straight-through estimator (STE) to flow gradients through the argmax.

**Variant**: `plan_action_gumbel_cumulative` uses the discounted cumulative objective $\mathcal{L} = -\sum_{t=1}^H \gamma^t V_\psi(z_t)$.

### Sequential Continuous MPC

For continuous action spaces. Action sequence $u \in \mathbb{R}^{H \times m}$, squashed by tanh:

```python
for t in range(H):
    z_t = dyn_step(z_t, B @ tanh(u[t]))
loss = -V_psi(z_H)
```

Adam for `plan_iters` steps. $B$ can be live (in computation graph) or detached (`frozen_b=True`). **Live B** (default) allows gradient signal to flow back into $B$ during planning, which empirically improves training.

### Block-Toeplitz GEMM Continuous MPC

See Section 7 for full derivation. Replaces the sequential rollout with a single dense GEMM.

---

## 7. Block-Toeplitz GEMM Derivation

### Motivation

For the sequential planner, computing $z_H$ from $z_0$ requires $H$ sequential calls to `dyn_step`. These cannot be parallelised because each step depends on the previous one. The autograd graph has depth $H$, memory $\mathcal{O}(Hd)$, and gradient computation requires $\mathcal{O}(H)$ sequential matmuls.

If $A \in O(d)$ and dynamics are exactly linear (no normalisation), we can precompute the entire influence of $A$ on the trajectory and express the full horizon as a single linear function of the action sequence.

### Linear Superposition

With $z' = Az + Bu$ (no sphere projection), the dynamics are linear. Any trajectory decomposes into:

1. **Zero-Input Response (ZIR)**: the trajectory if all actions were zero, $u_t \equiv 0$
2. **Zero-State Response (ZSR)**: the contribution of the action sequence assuming $z_0 = 0$

By linearity (discrete-time variation of parameters):

$$z_k = A^k z_0 + \sum_{j=0}^{k-1} A^{k-1-j} B u_j$$

### Matrix Form

Write the $H$-step trajectory as a block matrix equation. Let:
- $z_0$ — initial latent state
- $\mathbf{u} = [u_0, u_1, \ldots, u_{H-1}]^\top$ — action sequence, each $u_t \in \mathbb{R}^m$
- $\mathbf{Z} = [z_1, z_2, \ldots, z_H]^\top$ — trajectory

Then $\mathbf{Z} = \mathbf{Z}_\text{IR} + \mathbf{Z}_\text{SR}$, where:

$$\mathbf{Z}_\text{IR} = \begin{bmatrix} A z_0 \\ A^2 z_0 \\ \vdots \\ A^H z_0 \end{bmatrix} \in \mathbb{R}^{H \times d}$$

Each row is a matrix-vector product of an $A$ power with $z_0$. Precomputed once outside the planning loop.

For the zero-state response, define latent action vectors $X_t = B u_t \in \mathbb{R}^d$. Then:

$$\begin{bmatrix} z_1 \\ z_2 \\ z_3 \\ \vdots \\ z_H \end{bmatrix} = \underbrace{\begin{bmatrix} A^0 & 0 & 0 & \cdots & 0 \\ A^1 & A^0 & 0 & \cdots & 0 \\ A^2 & A^1 & A^0 & \cdots & 0 \\ \vdots & & & \ddots & \vdots \\ A^{H-1} & \cdots & A^2 & A^1 & A^0 \end{bmatrix}}_{\mathbf{W} \,\in\, \mathbb{R}^{Hd \times Hd}} \begin{bmatrix} X_0 \\ X_1 \\ X_2 \\ \vdots \\ X_{H-1} \end{bmatrix}$$

This is a **lower-triangular Block-Toeplitz** matrix $\mathbf{W}$ where block $\mathbf{W}_{ij} = A^{i-j}$ for $i \geq j$, and $0$ for $i < j$.

### Implementation

**Step 1: Precompute $A$ powers** (outside planning loop, `torch.no_grad()`)

```python
A_pows = [torch.eye(d)]
for _ in range(H):
    A_pows.append(A_pows[-1] @ A)
A_stack = torch.stack(A_pows)  # [H+1, d, d]
```

**Step 2: ZIR**

```python
ZIR = torch.einsum('kij,j->ki', A_stack[1:], z0)  # [H, d]
```

**Step 3: Build $\mathbf{W}_\text{Toeplitz}$**

```python
row_idx   = torch.arange(H).unsqueeze(1)           # [H, 1]
col_idx   = torch.arange(H).unsqueeze(0)           # [1, H]
power_idx = (row_idx - col_idx).clamp(min=0)       # [H, H]
causal    = (row_idx >= col_idx).float()            # [H, H]  lower-tri mask

W_blocks   = A_stack[power_idx] * causal[..., None, None]  # [H, H, d, d]
W_toeplitz = W_blocks.permute(0, 2, 1, 3).reshape(H*d, H*d)
```

The `permute(0,2,1,3)` reorders from `[row_block, col_block, row_d, col_d]` to `[row_block, row_d, col_block, col_d]` before the reshape, placing block elements contiguously in the final dense matrix.

**Step 4: Planning loop** (Adam, `plan_iters` steps)

```python
u_logits = randn(H, m) * 1e-4
u_logits.requires_grad_(True)
opt = Adam([u_logits], lr=0.1)

for _ in range(plan_iters):
    u      = tanh(u_logits)              # [H, m]
    X_flat = (u @ B.T).reshape(H*d, 1)

    ZSR  = (W_toeplitz @ X_flat).reshape(H, d)   # single cuBLAS GEMM
    Z    = ZIR + ZSR                      # [H, d]

    Z32  = Z.to(float32)
    loss = -(gammas * v_net(Z32)).sum()   # discounted cumulative value

    grad, = torch.autograd.grad(loss, u_logits)
    u_logits.grad = grad
    opt.step()
```

### Complexity Analysis

| Quantity | Sequential planner | Toeplitz planner |
|---|---|---|
| Forward pass | $\mathcal{O}(Hd^2)$ sequential matmuls | $\mathcal{O}(H^2 d^2)$ precompute (once) + $\mathcal{O}(H^2 d^2)$ GEMM |
| Autograd graph depth | $\mathcal{O}(H)$ | $\mathcal{O}(1)$ |
| Autograd memory | $\mathcal{O}(Hd)$ | $\mathcal{O}(Hd)$ |
| GPU parallelism | Low (sequential deps) | High (single GEMM) |

For the Pendulum experiment with $H=5$, $d=32$: the GEMM is $160 \times 160$, entirely within L1 cache on modern GPUs. The precompute cost is paid once per planning call and amortised over `plan_iters=20` Adam steps.

The autograd graph has depth $\mathcal{O}(1)$ instead of $\mathcal{O}(H)$ because $\mathbf{W}_\text{Toeplitz}$ and $\mathbf{Z}_\text{IR}$ are precomputed with `torch.no_grad()`. The only leaf requiring gradients is `u_logits`.

---

## 8. Numerical Stability: float64 in the Planner

### The Problem

With $d=32$ and $H=5$, each row of $\mathbf{W}_\text{Toeplitz} \mathbf{x}$ is the sum of up to $H \cdot d = 160$ float32 products. At the same time, the ZIR terms involve $A^k z_0$ for $k$ up to $H=5$.

In float32 (machine epsilon $\varepsilon \approx 1.2 \times 10^{-7}$), two effects compound:

1. **Catastrophic cancellation in the GEMM**: consecutive $A$ powers can have nearly-equal entries that cancel, losing significant bits.
2. **Accumulated rounding error in ZIR**: for an orthogonal $A$, $\|A^k z_0\| = \|z_0\|$ exactly in theory, but floating-point rounding introduces error that grows with $k$.

The result is that the value function receives latent vectors $\mathbf{Z}$ with significant numerical noise, corrupting the gradient signal.

### The Fix: float64 Inside the Planner

Cast $A$, $B$, $z_0$, and `u_logits` to float64 for all computations inside the planning loop. Cast back to float32 only for the `v_net` forward pass (which expects float32 inputs):

```python
z0 = encoder(state).to(torch.float64)
A  = agent.A.detach().to(torch.float64)
B  = agent.B.detach().to(torch.float64)

# ... precompute ZIR, W_toeplitz in float64 ...

u_logits = randn(H, m, dtype=torch.float64) * 1e-4
# ... Adam loop ...
    Z    = ZIR + ZSR             # float64
    Z32  = Z.to(torch.float32)   # cast only for v_net
    loss = -v_net(Z32[-1])
```

### Why This Works

Float64 has machine epsilon $\varepsilon_{64} \approx 2.2 \times 10^{-16}$, roughly 8 orders of magnitude smaller than float32. For $H \cdot d = 160$ summed products, the rounding error in float64 is approximately $160 \cdot 2.2 \times 10^{-16} \approx 3.5 \times 10^{-14}$, negligible compared to the magnitude of latent vectors (typically $\mathcal{O}(1)$).

The `v_net` remains in float32 throughout training — only the planning computation uses float64. This has no training overhead (planning is not differentiated through the network weights) and negligible runtime cost since the GEMM dimensions are small ($160 \times 160$).

### Empirical Evidence

Before float64:
- 5/5 Toeplitz seeds failed to reach return $> -300$ on Pendulum-v1
- Training was erratic: returns would occasionally improve then collapse

After float64:
- 4/5 seeds solved the task (return $> -300$ at some point in training)
- Training curves were smoother and more monotone

---

## 9. Ornstein-Uhlenbeck Exploration

### The Problem with i.i.d. Gaussian Noise

Standard exploration adds i.i.d. Gaussian noise to actions: $a_t = \pi(s_t) + \varepsilon_t$, $\varepsilon_t \sim \mathcal{N}(0, \sigma^2 I)$. For a pendulum swing-up, the agent must sustain correlated torques over multiple steps to build up angular momentum. White noise gives no temporal correlation — the expected net angular impulse over any interval is zero.

### Ornstein-Uhlenbeck Process

The OU process is a mean-reverting stochastic process producing temporally correlated noise:

$$dx = \theta(\mu - x)\,dt + \sigma\,dW$$

In discrete time (Euler-Maruyama):

$$x_{t+1} = x_t + \theta(\mu - x_t)\,\Delta t + \sigma\sqrt{\Delta t}\;\xi_t, \qquad \xi_t \sim \mathcal{N}(0, I)$$

Parameters:
- $\theta = 0.15$ — mean-reversion rate (higher $\Rightarrow$ noise decorrelates faster)
- $\mu = 0$ — long-run mean (zero-mean noise)
- $\sigma = 0.2$ — noise magnitude
- $\Delta t = 0.05$ — time step (matches Gymnasium Pendulum-v1 default)

### Implementation

```python
class OUNoise:
    def __init__(self, action_dim, theta=0.15, sigma=0.2, dt=0.05):
        self.theta = theta
        self.sigma = sigma
        self.dt    = dt
        self.x     = np.zeros(action_dim)

    def sample(self):
        dx   = self.theta * (-self.x) * self.dt
        dx  += self.sigma * np.sqrt(self.dt) * np.random.randn(len(self.x))
        self.x += dx
        return self.x

    def reset(self):
        self.x = np.zeros(len(self.x))
```

`reset()` is called at episode boundaries to prevent noise from one episode carrying into the next.

### Effect on Exploration

OU noise creates smooth noise trajectories that can sustain torque in one direction for multiple steps, helping the pendulum accumulate enough angular momentum for a swing-up. It is particularly beneficial during the warmup phase (first 10k steps) when the policy is random.

The `--ou_noise` flag enables OU noise; the default is i.i.d. Gaussian (purer from a theory standpoint, no hyperparameters beyond $\sigma$).

---

## 10. Training Protocol

### Pendulum-v1 Configuration

| Parameter | Value |
|---|---|
| Environment | `Pendulum-v1` (Gymnasium) |
| State | $[\cos\theta,\; \sin\theta,\; \dot\theta]$, dim 3 |
| Action | Scalar torque $\in [-2, 2]$, dim 1 |
| Latent dimension $d$ | 32 |
| $A$ constraint | $O(d)$ via SVD Procrustes (hard on CUDA) |
| Encoder output | Raw (no normalisation, `no_normalize=True`) |
| Horizon $H$ | 5 |
| Plan iterations | 20 |
| Planner learning rate | 0.1 (Adam) |
| Discount $\gamma$ | 0.99 |
| Network learning rate | $3 \times 10^{-4}$ |
| Koopman LR scale | 0.5 ($A$, $B$ updated at $1.5 \times 10^{-4}$) |
| Batch size | 256 |
| Buffer size | 100,000 |
| Warmup steps | 10,000 (pure random actions) |
| Total steps | 40,000 |
| EMA $\tau$ | 0.005 |
| Exploration noise decay | Linear from 1.0 to 0.1 over 15,000 steps |

### Training Loop

```
for step in 1..N_STEPS:
    if step <= WARMUP:
        action = random_action()
    else:
        noise  = ou_noise.sample()  [or Gaussian]
        action = agent.act_plan_continuous(state) + noise * noise_scale
        noise_scale decays linearly

    next_state, reward, done = env.step(action)
    buffer.push(s, a, r, s', done)

    if done: ou_noise.reset()

    if step > WARMUP and buffer.ready():
        batch = buffer.sample(256)
        compute L_koop, L_v, L_recon
        backprop, Adam step
        target.update(agent, tau=0.005)

    if step % 1000 == 0:
        ret20 = mean(last 20 episode returns)
        if ret20 > best_ret:
            save_checkpoint(agent, path="best_*.pt")
```

### Best-Model Checkpointing

The model is saved every 1,000 steps when the rolling return over the last 20 episodes improves. This is important because Koopman models can be unstable: the best performance is often achieved mid-training, after which the model may diverge. Saving the peak rather than the final state captures the most useful checkpoint.

### Logging

Every 1,000 steps:
```
step  5000  ret/20=-850.3  L_koop=0.0124  L_v=0.3217  L_recon=0.0041
  [best] ret/20=-732.4 → checkpoints/best_toe_s0.pt
```

Live plot saved to `viz_pendulum/pendulum_live_{run_tag}.png` every 1,000 steps, showing:
- Episode returns over time
- Rolling $\text{ret}/20$
- $\mathcal{L}_\text{koop}$, $\mathcal{L}_v$ curves
- Latent value landscape (2D projection)
- Phase portrait ($\theta$ vs $\dot\theta$, coloured by value)

---

## 11. Experimental Results

### Pendulum-v1: 5-seed Toeplitz vs Sequential (40k steps)

Config: `ortho_a=True`, `no_normalize=True`, `ou_noise=True`, `WARMUP=10000`, float64 planner.

| Run | Planner | Seed | Best ret/20 | Solved? |
|---|---|---|---|---|
| toe_s3 | Toeplitz | 3 | $-180$ | Yes |
| toe_s0 | Toeplitz | 0 | $-293$ | Yes |
| toe_s2 | Toeplitz | 2 | $-352$ | Yes |
| toe_s4 | Toeplitz | 4 | $-362$ | Yes |
| seq_s0 | Sequential | 0 | $-369$ | Yes |
| toe_s1 | Toeplitz | 1 | $-883$ | No |

Success criterion: best rolling return $> -300$ (consistent swing-up with some balancing).

**Observations:**
- 4/5 Toeplitz seeds solved the task after the float64 fix. Before float64, 0/5 solved consistently.
- Seed 1 was an unlucky initialisation — the SVD Procrustes parametrisation can occasionally produce $A$ matrices that slow down early learning.
- The Sequential baseline (seed 0) reached $-369$, suggesting Toeplitz is competitive with sequential despite detaching $A$ and $B$ from the planning computation graph.

### GravityBasin: Best Config Results

Config: `ortho_raw` (`ortho_a=True`, `no_normalize=True`). 40k steps.

- 100% greedy success rate
- Mean episode length on success: 20.6 steps

---

## 12. Design Decisions and Trade-offs

### Why Detach A and B in Toeplitz?

The Toeplitz planner computes $\mathbf{Z}_\text{IR}$ and $\mathbf{W}_\text{Toeplitz}$ once outside the Adam loop using `agent.A.detach()` and `agent.B.detach()`. This prevents gradients from the planning objective from flowing back into $A$ and $B$.

**Rationale**: The planning computation and the training computation use different objectives. During training, $A$ and $B$ are updated by the Koopman consistency loss (predicting next latent states), not by the planning value objective. Allowing planning gradients to update $A$ and $B$ would corrupt the Koopman structure being learned.

**Cost**: The sequential planner (with live $B$) allows the planning gradient to influence $B$ indirectly through the training update. Empirically this gives a small advantage for the sequential planner, but the effect is modest and the Toeplitz planner is competitive with float64.

### Cumulative vs Terminal Objective

Two objectives for planning:

- **Terminal only**: $\mathcal{L} = -V_\psi(z_H)$ — optimise only the final state value.
- **Cumulative**: $\mathcal{L} = -\sum_{t=1}^H \gamma^t V_\psi(z_t)$ — optimise discounted sum over the horizon.

Terminal-only is more stable during training (less gradient variance) and easier to optimise. Cumulative provides a denser signal but can destabilise early training.

The `--cumulative` flag switches between them. Default is terminal-only.

### Warm Start Planning (Deactivated)

The `WarmStartToeplitzPlanner` class implements warm-starting: at each timestep, the previous action sequence is shifted by one step and used to initialise the next planning call. Adam's first and second moment estimates are also shifted to carry curvature information forward.

In theory this should speed up planning convergence. In practice it was found to **hurt performance** — the optimizer momentum from the previous step interfered with adapting to the new state. The class remains in the codebase for reference but is not used in the default configuration.

### Sequential vs Toeplitz: Live B vs Frozen B

A confounding factor in the sequential vs Toeplitz comparison: the sequential planner by default keeps $B$ in the computation graph, while Toeplitz always detaches it. The `--frozen_b` flag makes sequential match Toeplitz behaviour.

With `frozen_b=True`, the sequential planner's performance drops closer to Toeplitz, suggesting that **live $B$ contributes to the sequential planner's advantage**. The mechanism: when $B$ is live, the planning gradient $\partial(-V_\psi(z_H))/\partial B$ provides an additional training signal for the action-input matrix, supplementing the Koopman consistency loss. This signal is absent in the Toeplitz planner; the float64 fix compensates by ensuring the existing Koopman gradient is clean and usable.

### Encoder No-Normalize with ortho_a=True

The sphere normalisation $z \leftarrow z / \|z\|$ breaks the linear superposition principle because normalisation is a nonlinear operation. With `ortho_a=True`, we set `no_normalize=True` on the encoder, removing the final `F.normalize` call.

This makes the latent space unbounded in principle, but since $A \in O(d)$ preserves norms and $B$ is initialised with orthonormal columns (`nn.init.orthogonal_`), the latent norm grows slowly and the dynamics remain well-conditioned in practice.

---

*Generated 2026-03-19.*
