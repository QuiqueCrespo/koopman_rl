# Handover — KoopmanGradientPlanner (sheaf_rl package)

## Current state: BROKEN on MPS

`sheaf_rl/model.py` uses `torch.nn.utils.parametrizations.orthogonal(..., orthogonal_map="matrix_exp")`
for the `ortho_a=True` branch. **`torch.matrix_exp` is not implemented on MPS** — the run crashes immediately.

### Decision needed before relaunching

| Option | Constraint | Speed | MPS works? |
|--------|-----------|-------|-----------|
| Soft penalty `\|\|AᵀA−I\|\|²_F` | Approximate | Fastest (pure matmuls) | ✅ |
| `parametrizations.orthogonal(matrix_exp)` | Exact | Fast (matmuls) | ❌ |
| `parametrizations.orthogonal(cayley)` | Exact | Fast (solve) | ❌ (linalg.solve not on MPS) |
| SVD Procrustes | Exact | Slow (CPU round-trip) | ⚠️ with `PYTORCH_ENABLE_MPS_FALLBACK=1` |

**On CUDA**: use `parametrizations.orthogonal` — fully native, no workaround needed.
**On MPS**: only the soft penalty avoids CPU fallback.

### To fix for MPS — revert to soft penalty

In `sheaf_rl/model.py`:
1. Remove `import torch.nn.utils.parametrizations as parametrizations`
2. Replace the `if ortho_a:` block in `__init__` with just `self.A = nn.Parameter(torch.eye(d))`
3. Remove the `__getattr__` override
4. Restore `ortho_penalty()` method:
   ```python
   def ortho_penalty(self):
       I = torch.eye(self.d, device=self.A.device)
       return (self.A.T @ self.A - I).pow(2).sum()
   ```
5. Restore `koop_parameters()` to `return [self.A, self.B]`

In `sheaf_rl/algorithms.py`:
- Restore `LAMBDA_ORTHO = a.lambda_ortho` and `L_ortho = agent.ortho_penalty() if m.ortho_a else 0.0` in the loss

### To fix for CUDA — keep hard constraint

The current `model.py` code is already correct for CUDA. Just remove the `__getattr__`
workaround and access `self._A_layer.weight` directly (or add a `@property`).

---

## Completed experiments (all on MPS, seed=42, 100k steps, warmup=20k)

All three hit **100% greedy success** with the soft penalty + SVD hard constraint (pre-this-session):

| Config | Encoder | 20/20 from | Mean steps |
|--------|---------|-----------|-----------|
| `configs/ortho_raw.py` | free (no normalization) | step 40k | **20.6** |
| `configs/ortho_tanh.py` | tanh ∈ (−1,1)^d | step 45k | 21.9 |
| `configs/ortho_l2.py` | L2 unit sphere | step 45k | 23.4 |

**Finding**: fully linear (ortho_raw) is fastest. L2 normalization slightly hurts by
collapsing different-magnitude states to the same point on S^{d−1}.

---

## Package structure

```
sheaf_rl/
  model.py        — KoopmanGradientPlanner, Encoder, ValueNetwork, TargetNetwork
  algorithms.py   — train(), evaluate(), evaluate_planner(), directed_value_iteration()
  config.py       — Config, EnvConfig, ModelConfig, AlgoConfig, TrainConfig, PlannerConfig
  planner.py      — plan_action_{shooting,beam,gumbel,gumbel_cumulative,softmax,...}
  env.py          — GravityBasin
  buffer.py       — ReplayBuffer
  viz.py          — plot_live(), visualize_graph()
configs/
  ortho_l2.py     — ortho_a=True, L2 encoder
  ortho_tanh.py   — ortho_a=True, tanh encoder
  ortho_raw.py    — ortho_a=True, no normalization (best)
scripts/
  train.py        — entry point: python scripts/train.py --config configs/ortho_raw.py
```

## Key config flags

```python
AlgoConfig:
  no_graph=True        # bisim graph disabled by default (pure TD)
  no_normalize=True    # skip F.normalize in encoder.forward() (ortho_raw)
  lambda_ortho=1.0     # weight of soft ||AᵀA−I||²_F penalty

ModelConfig:
  ortho_a=True         # A constrained to O(d) (soft or hard depending on device)
  tanh_out=True        # tanh as final encoder layer (ortho_tanh)

TrainConfig:
  warmup=20_000        # critical — GravityBasin random walk finds goal ~1% of episodes
  eps_end=0.15         # high floor needed (sparse reward, cubic gravity pulls to origin)
  eps_decay=70_000
```

## Next steps (suggested)

1. Fix the MPS/CUDA ortho_a dispatch (device-conditional in `__init__`)
2. Implement parallel horizon unroll: `z_H = A^H z_0 + Σ A^{H-1-t} B a_t`
   — pre-compute `[A^1, ..., A^H]` as a single `torch.linalg.matrix_power` batch,
   evaluate entire lookahead in one einsum instead of a Python for-loop
3. Test on harder environments (Pendulum-v1 continuous, experiments/pendulum_kgp.py)
