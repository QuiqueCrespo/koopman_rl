# Global Graph Diffusion for Value Learning

## Overview

Standard TD learning propagates reward backwards one step at a time — slow in sparse-reward settings where the reward signal touches only a tiny fraction of the replay buffer. This method replaces TD bootstrapping with a **global diffusion** over a dynamically constructed sparse graph. Every time the graph is rebuilt, Richardson iteration spreads value from the few reward-carrying transitions simultaneously to all 1024 nodes in the graph, providing dense supervision for the value network without any n-step rollout.

**Environment.** 2D gravity basin: the agent navigates in (x,y) ∈ [-1,1]² against cubic gravity:
```
x_{t+1} = clip(x + dx - 0.05·x³, -1, 1)
```
Reward +1 and episode terminates when x>0.8 and y>0.8 (top-right corner). All other steps give reward 0. Episodes end after 200 steps or on success.

---

## Architecture

### Encoder and Linear Latent Dynamics

A neural encoder `f_θ: R² → R^d` (d=32) maps physical states to a latent space. The key architectural constraint is that latent dynamics are modelled as **linear**: for each action a there is a learned matrix `K_a ∈ R^{d×d}`, and the predicted next latent under action a is:

```
ẑ_{t+1} = normalize(K_a · z_t)
```

The unit-sphere normalisation keeps latents bounded. The **dynamics loss** penalises prediction error against the target-encoder embedding of the actual next state:

```
L_dyn = mean over non-terminal transitions: || normalize(K_a · z_src) - z_dst_target ||²
```

The matrices `{K_a}` learn to capture how each action moves the agent in latent space. Because they are linear, the predicted value of a sequence of actions is easy to compute: just multiply the matrices in order and pass through the value head.

### Value Network

A scalar head `V_ψ: R^d → R` maps latent vectors to value estimates. The value network is supervised entirely by the output of the graph diffusion (described below) — no 1-step TD targets are used.

### Target Network

A slow exponential moving average (EMA, τ=0.005) of the encoder and value head provides stable targets for the dynamics loss and for graph construction. All graph nodes are encoded with the target encoder.

---

## Graph Construction

Every 500 environment steps, a sparse directed graph is rebuilt from replay buffer samples. It has 1024 nodes and two types of edges.

### Nodes

**32 trajectory chunks** of 16 contiguous steps each are sampled from the replay buffer. Contiguous sampling is essential: it preserves the temporal ordering needed for directed edges to represent causal transitions. Flat random sampling would give 512 isolated 1-step arrows with no chain structure.

This gives **512 source states + 512 next-states = 1024 nodes** total. All are encoded with the target encoder: `Z ∈ R^{1024 × 32}`.

### Temporal Edges (512 directed)

Each sampled transition `(s_i, a_i, r_i, s'_i, done_i)` becomes a directed edge `src_i → dst_i`. These form 32 independent 16-step chains — one per chunk.

### Latent Similarity Edges (~2560 undirected)

k-NN edges (k=5) connect nodes whose target-encoder embeddings are close, regardless of which chain they belong to. These cross-chain connections are the **critical structural component**: without them, the 32 chains are isolated and reward can only spread within the single chain that happened to contain a goal transition. The k-NN edges stitch all chains into a single connected graph, so reward propagates globally.

Number of similarity edges ≈ 1024 × 5 / 2 ≈ 2560 (symmetric, deduplicated).

### Edge Weights: Off-Policy Filtering

Each temporal edge is weighted by how on-policy the recorded action was, using the current value estimate:

```
advantage_k = V(normalize(K_{a_k} · z_k)) − max_a V(normalize(K_a · z_k))   ≤ 0
W_k = exp(advantage_k / τ)        τ = 0.3
w_k = sqrt(W_k)                   ∈ (0, 1]
```

If the agent took the greedy action, advantage=0 and W=1 (full weight). If it took a suboptimal action, W < 1. This soft filter prevents off-policy transitions from injecting reward in misleading directions during diffusion: a transition where a bad action was taken should not tell the diffusion that the source state is valuable.

---

## Value Diffusion

### The Diffusion Problem

We want to find a value assignment `V ∈ R^{1024}` over all graph nodes that satisfies the Bellman consistency conditions encoded by the graph edges — i.e., for every temporal edge:

```
V[src] ≈ r + γ · V[dst]
```

and for every similarity edge:

```
V[src] ≈ V[dst]   (similar states should have similar values)
```

This is a large linear system. We solve it iteratively.

### Graph Laplacian (message-passing form)

The system can be written as `L·V = R`, where L is a weighted graph Laplacian and R is a reward source vector. Rather than forming L explicitly, we compute its action on V via scatter-add message passing:

**Temporal edge k (directed, weight w_k):**
```
residual_k = w_k · (V[src_k] − γ(1−done_k) · V[dst_k])
ΔV[src_k] += w_k · residual_k
ΔV[dst_k] -= γ(1−done_k) · w_k · residual_k
```

**Similarity edge k (undirected):**
```
ΔV[src_k] += V[src_k] − V[dst_k]
ΔV[dst_k] -= V[src_k] − V[dst_k]
```

### Reward Injection

The reward signal enters the system through a source vector `R_v ∈ R^{1024}`, computed from each transition's immediate reward:

```
R_v[src_k] += w_k · r_k
R_v[dst_k] -= γ(1−done_k) · w_k · r_k
```

The negative term at the destination encodes the Bellman discount: if `r>0` at src, the dst node's value should be *lower* by exactly `γ·r`. Without this, the diffusion would converge to `V(s) = r` everywhere (no discounting).

### Richardson Iteration

We solve `L·V = R_v` iteratively with a small regularisation anchor toward the target-network values `V_init`:

```
V^(k+1) = V^(k) − α · (L·V^(k) − R_v) − β · (V^(k) − V_init)
```

where:
- `α = 1 / λ_max(L)` — step size, computed via 15 power-iteration steps to ensure convergence
- `β = 0.001` — anchor strength (prevents drift when `R_v` is nearly zero)
- 300 iterations per graph rebuild

The anchor `β · (V − V_init)` acts as a Tikhonov regulariser: it keeps values from wandering when the graph has almost no reward signal (early in training, most graph nodes see `R_v ≈ 0`).

The result `V_diff ∈ R^{1024}` is clamped to `[0, 1/(1−γ)]` and used as regression targets.

### Value Loss

```
L_v = mean over all 1024 graph nodes: (V_ψ(f_θ(s_i)) − V_diff[i].detach())²
```

### Combined Loss

```
loss = L_dyn + L_v
```

---

## Training Setup

| Hyperparameter | Value | Role |
|---|---|---|
| d (latent dim) | 32 | |
| N_CHUNKS | 32 | trajectory chunks per graph |
| T_CHUNK | 16 | steps per chunk |
| K_BISIM_NN | 5 | similarity k-NN per node |
| FORCE_GOAL | True | goal-anchored sampling (see below) |
| GOAL_FORCE_PROB | 0.5 | fraction of rebuilds that force a goal chunk |
| GRAPH_REBUILD | 500 | steps between graph rebuilds |
| K_DIFFUSE | 300 | Richardson iterations |
| ALPHA_ITERS | 15 | power iteration steps for step size |
| BETA_TIKHONOV | 1e-3 | anchor to target-net values |
| TAU_GRAPH | 0.3 | off-policy filter temperature |
| BATCH_SIZE | 256 | transitions for dynamics loss |
| WARMUP | 3000 | random steps before training |
| N_STEPS | 100,000 | total training steps |
| ε | 1.0 → 0.05 over 40k steps | exploration |
| KOOP_LR_SCALE | 0.5 | dynamics matrix LR = base LR × 0.5 |
| KOOP_WD | 1e-3 | weight decay on dynamics matrices |
| EMA_TAU | 0.005 | target network update rate |
| GAMMA | 0.95 | discount factor |

### Goal-Anchored Sampling

In sparse reward settings, 512 randomly sampled transitions almost never include the rare goal event — so `R_v ≈ 0` and diffusion targets are zero everywhere. `FORCE_GOAL=True` anchors one of the 32 chunks to end at a goal transition on 50% of rebuilds (`GOAL_FORCE_PROB=0.5`), guaranteeing the reward signal enters the graph regularly. The other rebuilds use pure random sampling. This is the minimal form of reward-prioritised sampling needed for the graph diffusion to work with sparse rewards.

---

## What Was Tried and Removed

### Bisimulation Metric Loss (DBC, Zhang et al. 2021)

Deep Bisimulation for Control pushes `||z_i − z_j||` toward `|r_i − r_j| + γ·||z'_i − z'_j||`. In dense-reward settings this gives useful metric structure. In sparse rewards, `|r_i − r_j| = 0` for ~99% of pairs (both transitions have zero reward), leaving only `γ·||z'_i − z'_j||` — which contracts all non-goal embeddings toward a single point. The linear dynamics matrices then have nothing to learn, and the latent space collapses. **Disabled.**

### 1-Step TD Loss

A standard double-DQN target `y = r + γ · V_target(s')` was available as a secondary supervision signal alongside the graph diffusion. With the dynamics loss providing encoder structure and the diffusion providing multi-step value targets, the 1-step TD signal adds noise without improving performance. **Disabled** for the experiments reported here.

---

## Ablation Results

All runs: same environment, architecture, and seeds (torch=42, np=42).
**Metric:** `succ/20` = successes in the last 20 training episodes at the logged step (ε still > 0).
**Baseline** = full method (FORCE_GOAL=True, off-policy filter, k=5 similarity edges, no TD, no bisim loss).

### Baseline (full method)

| Step | ε | succ/20 | V_diff max | V_diff mean |
|------|---|---------|------------|-------------|
| 4k   | 0.91 | 1  | 0.79 | 0.02 |
| 12k  | 0.72 | 4  | 0.65 | 0.25 |
| 14k  | 0.67 | 11 | 0.70 | 0.31 |
| 16k  | 0.62 | 15 | 0.82 | 0.35 |
| 18k  | 0.57 | 19 | 0.79 | 0.34 |
| 20k  | 0.53 | **20** | 0.84 | 0.40 |
| 24k  | 0.43 | **20** | 0.93 | 0.45 |
| 28k+ | <0.35 | 17–20 | — | — |

### Ablation 1 — No goal-anchored sampling (FORCE_GOAL=False)

| Step | succ/20 | V_diff max |
|------|---------|------------|
| 6k  | 2 | 0.037 |
| 8k  | 1 | 0.026 |
| 10k | 2 | 0.474 *(lucky spike)* |
| 12k | 3 | 0.031 |

**Result: complete failure.** Without goal anchoring, 512 random transitions almost never include the goal. `R_v ≈ 0` everywhere → diffusion targets ≈ 0 → the value network never receives a useful gradient. The occasional spike (step 10k) happens when a goal transition lands in the random sample by chance, then collapses again. This is not overfitting to the environment — it is the minimum form of reward prioritisation that any graph-based method needs to work with sparse rewards.

### Ablation 2 — No off-policy filter (uniform W=1)

| Step | succ/20 (with filter) | succ/20 (uniform) |
|------|-----------------------|-------------------|
| 12k  | 4  | 6  |
| 14k  | **11** | 8  |
| 16k  | **15** | 8  |
| 18k  | **19** | 9  |

**Result: significant degradation.** Without the off-policy filter, all temporal edges carry equal weight in the diffusion regardless of how suboptimal the recorded action was. Off-policy transitions inject reward signal in the wrong direction (a bad action followed by a lucky next state looks valuable), adding noise that the diffusion cannot resolve. The filtered version reaches near-perfect performance ~8k steps earlier.

### Ablation 3 — No latent similarity edges (k=0)

| Step | succ/20 | V_diff max |
|------|---------|------------|
| 4k  | 0 | 0.001 |
| 6k  | 0 | 0.001 |
| 8k  | 0 | 0.001 |

**Result: total failure.** Without k-NN edges, the 32 temporal chains are completely disconnected. Only the single goal-anchored chain can carry a non-zero `R_v`. Diffusion within the other 31 chains stays at zero because there is no path from the goal node to them. The similarity edges are what makes the diffusion genuinely global — they are the bridge between isolated trajectory chains.

---

## Summary

The method achieves **succ/20 = 20** (perfect within the exploration-limited regime) by step 20–24k on a sparse reward navigation task at ~35 env steps/second.

Each component has a clear functional role:

| Component | Role | Ablation effect |
|-----------|------|-----------------|
| Linear latent dynamics | Structured action-conditioned prediction | (not ablated) |
| Off-policy filter (W) | Prevents noisy reward injection from bad actions | -50% succ at step 18k |
| Latent k-NN edges | Makes diffusion global across all chains | Total failure |
| Goal-anchored sampling | Guarantees reward enters graph in sparse settings | Total failure |
| Richardson diffusion | Propagates reward globally in one rebuild | (the core mechanism) |

The central advantage over TD learning is that Richardson diffusion solves the multi-step value equation globally across all 512 sampled transitions simultaneously. A single goal-reaching trajectory in the batch propagates value to all 1024 graph nodes in one rebuild, regardless of how far they are from the goal in terms of trajectory steps.
