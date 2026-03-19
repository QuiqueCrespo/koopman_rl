"""
Latent Affine Koopman-RL: Global Richardson Diffusion ("Ferrari")
===============================================================
Replaces the T-step Tree-Backup chain solver with full Richardson diffusion
over a dynamically constructed sparse Koopman graph.

Graph structure (rebuilt every GRAPH_REBUILD env steps):
  - M_SRC=512 source states + M_SRC next-states = 2*M_SRC nodes total
  - M_SRC temporal edges (src_i → dst_i) with Koopman restriction maps
  - k-NN bisimulation edges (K_BISIM_NN=5) over target-encoder embeddings

Incidence matrix B encodes the graph Laplacian structure:
  stalk = D + 1  (D latent dimensions + 1 value slot)
  Each edge contributes two blocks to B (source restriction, dest restriction).

Richardson iteration (no explicit Delta_F = B^T B formation):
  X^(k+1) = X^(k) - alpha * B^T(B X^(k)) + alpha * R
  with Dirichlet anchors re-applied each step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gravity_basin import (
    GravityBasin, Encoder, ValueNetwork, KoopmanAgent, TargetNetwork,
    ReplayBuffer, compute_bisimulation_loss, evaluate,
    N_ACTIONS, STATE_DIM, D, GAMMA, EMA_TAU, LR, MAX_EP_STEPS,
    LAMBDA_KOOP, LAMBDA_BISIM, BUFFER_SIZE,
    _goal_patch, ACTION_COLORS, ACTION_NAMES, DELTA,
)

# Re-import plot helpers we will reuse
from gravity_basin import _value_grid, _policy_grid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CHUNKS        = 32      # number of contiguous trajectory chunks for graph construction
T_CHUNK         = 16      # steps per chunk
M_SRC           = N_CHUNKS * T_CHUNK   # 512 temporal edges total
K_BISIM_NN      = 5        # bisimulation k-NN per node (set 0 to disable bisim edges)
FORCE_GOAL      = True    # anchor one chunk to a goal transition on some rebuilds
GRAPH_REBUILD   = 500     # env steps between graph rebuilds
K_DIFFUSE       = 300     # Richardson iterations
ALPHA_ITERS     = 15      # power iteration steps to estimate lambda_max
BETA_TIKHONOV   = 1e-3    # Tikhonov anchor strength in Richardson diffusion
LAMBDA_V        = 0.5     # weight for sheaf-diffused value loss (small: L_dyn dominates)
TAU_GRAPH       = 0.1     # off-policy filter temperature (lower = stricter)
KOOP_LR_SCALE   = 0.5     # dynamics (A, B) LR = LR * KOOP_LR_SCALE
KOOP_WD         = 0.0     # no weight decay: L_koop bounds scale naturally (no F.normalize)
BATCH_SIZE      = 256     # flat transitions per gradient step
WARMUP          = 3_000   # random-policy steps before training
N_STEPS         = 100_000
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 40_000
LOG_EVERY       = 2_000
PLOT_EVERY      = 2_000  # save live plot to koopman_rl_live.png

# Sparse graph ops live on CPU to avoid MPS sparse limitations.
# Neural forward passes can use DEVICE (MPS / CUDA / CPU).
GRAPH_DEVICE  = torch.device("cpu")
DEVICE        = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("cpu")
)

# ---------------------------------------------------------------------------
# 1–3. Message-passing graph Laplacian (replaces sparse incidence matrix B)
# ---------------------------------------------------------------------------
# Mathematical equivalence: B^T(B X) = scatter/gather over edges.
# The B matrix has zero off-diagonal blocks between latent and value stalk
# dimensions, so the value iteration is fully decoupled from latents.
# Since latents are re-pinned (Dirichlet) every Richardson step, we only
# iterate on the N-dimensional value vector — no [N*(d+1)] sparse ops needed.
#
# Temporal edge k  (src → dst, weight w_k = W_sqrt[k]):
#   edge message:  e_k = w_k * (v_src_k − γ*(1−done_k)*v_dst_k)
#   src receives:  +w_k * e_k
#   dst receives:  −γ*(1−done_k) * w_k * e_k
#
# Bisimulation edge k (undirected, identity restriction maps):
#   src receives:  +(v_src_k − v_dst_k)
#   dst receives:  −(v_src_k − v_dst_k)
# ---------------------------------------------------------------------------

def _lap_v(
    X_v:    torch.Tensor,   # [N]
    src:    torch.Tensor,   # [E_temp]
    dst:    torch.Tensor,   # [E_temp]
    W_sqrt: torch.Tensor,   # [E_temp]  sqrt of advantage weights
    dones:  torch.Tensor,   # [E_temp]  float terminal flags
    bs_src: torch.Tensor,   # [E_bisim]
    bs_dst: torch.Tensor,   # [E_bisim]
    gamma:  float,
) -> torch.Tensor:           # [N]
    """Compute (B^T B X)_value via scatter-add message passing."""
    N      = X_v.shape[0]
    coeff  = gamma * (1.0 - dones)                             # [E_temp]
    e_v    = W_sqrt * (X_v[src] - coeff * X_v[dst])           # [E_temp]
    delta  = torch.zeros(N, device=X_v.device, dtype=X_v.dtype)
    delta.scatter_add_(0, src,  W_sqrt * e_v)
    delta.scatter_add_(0, dst, -coeff * W_sqrt * e_v)
    if bs_src.numel() > 0:
        diff = X_v[bs_src] - X_v[bs_dst]
        delta.scatter_add_(0, bs_src,  diff)
        delta.scatter_add_(0, bs_dst, -diff)
    return delta


def power_iteration_alpha(
    src:    torch.Tensor,
    dst:    torch.Tensor,
    W_sqrt: torch.Tensor,
    dones:  torch.Tensor,
    bs_src: torch.Tensor,
    bs_dst: torch.Tensor,
    gamma:  float,
    N:      int,
    iters:  int,
    device: torch.device,
) -> float:
    """Estimate 1/lambda_max(B^T B) on value space via power iteration."""
    v   = torch.randn(N, device=device)
    v  /= v.norm()
    lam = 1.0
    for _ in range(iters):
        Lv  = _lap_v(v, src, dst, W_sqrt, dones, bs_src, bs_dst, gamma)
        lam = Lv.norm().item()
        if lam < 1e-12:
            break
        v = Lv / lam
    return 1.0 / (lam + 1e-6)


def richardson_diffuse(
    alpha:   float,
    X_v0:    torch.Tensor,   # [N]  initial values (target-net estimates)
    R_v:     torch.Tensor,   # [N]  reward source  (B^T R_edges in value space)
    V_anch:  torch.Tensor,   # [N]  Tikhonov anchor
    src:     torch.Tensor,
    dst:     torch.Tensor,
    W_sqrt:  torch.Tensor,
    dones:   torch.Tensor,
    bs_src:  torch.Tensor,
    bs_dst:  torch.Tensor,
    gamma:   float,
    K_steps: int,
    beta:    float = 0.01,
) -> torch.Tensor:            # [N]
    """
    Richardson iteration on the value stalk:
        v^{k+1} = v^k − α (L_v v^k − R_v + β(v^k − V_anch))
    """
    X_v = X_v0.clone()
    for _ in range(K_steps):
        grad = _lap_v(X_v, src, dst, W_sqrt, dones, bs_src, bs_dst, gamma) \
               - R_v + beta * (X_v - V_anch)
        X_v  = X_v - alpha * grad
    return X_v


# (legacy stub kept for reference — no longer used)
def build_incidence_coo(
    N:         int,               # total number of nodes  (2 * M_SRC)
    E_temp:    int,               # number of temporal edges  (M_SRC)
    src_nodes: torch.Tensor,      # [E_temp] — source node indices
    dst_nodes: torch.Tensor,      # [E_temp] — destination node indices
    actions:   torch.Tensor,      # [E_temp] int64 — action taken on each edge
    K_mats:    list,              # list of N_ACTIONS tensors, each [d, d], on cpu
    bisim_src: torch.Tensor,      # [E_bisim] — source node indices for bisim edges
    bisim_dst: torch.Tensor,      # [E_bisim] — dest node indices for bisim edges
    d:         int,               # latent dimension D
    gamma:     float,             # discount factor
    device:    torch.device,      # target device (GRAPH_DEVICE = cpu)
    W_sqrt:    torch.Tensor,      # [E_temp] — sqrt(Koopman advantage weights), on device
    dones:     torch.Tensor,      # [E_temp] — float (0 or 1) terminal flags
) -> torch.sparse.Tensor:
    """
    Build the incidence matrix B in COO format.

    B has shape [E_total * stalk, N * stalk] where:
      E_total = E_temp + E_bisim
      stalk   = d + 1

    For temporal edge k (src_nodes[k] → dst_nodes[k], action a):
      Each row of edge k is scaled by W_sqrt[k] = sqrt(exp(A_k/tau)),
      where A_k = V(K_a z_k) - max_a' V(K_a' z_k) <= 0.
      This means B^T B = B^T W B automatically (weighted Laplacian),
      so the diffusion minimises a weighted residual that down-weights
      suboptimal (off-policy) transitions.

      Source block (row group: edge k, col group: src_nodes[k]):
        upper-left d×d  = W_sqrt[k] * K_a  (scaled Koopman operator)
        bottom-right 1×1 = W_sqrt[k]
      Dest block (row group: edge k, col group: dst_nodes[k], negated):
        upper-left d×d  = -W_sqrt[k] * I_d
        bottom-right 1×1 = -W_sqrt[k] * gamma * (1 - done[k])
        The (1-done) mask zeros the dest value coupling for terminal transitions,
        keeping the incidence matrix consistent with the (1-done) masking in the
        TD target. Without it: V_src = td_target + γ·V_dst > 1 whenever V_dst > 0.

    For bisimulation edge k (bisim_src[k] ↔ bisim_dst[k]):
      Source block: +I_{stalk}
      Dest block:   -I_{stalk}

    All index arithmetic is done with vectorized tensor ops; one Python loop
    over N_ACTIONS to handle per-action K_a blocks.
    """
    stalk   = d + 1
    E_bisim = bisim_src.shape[0]
    E_total = E_temp + E_bisim

    # We accumulate (row, col, val) triples as lists, then cat and build COO.
    rows_list = []
    cols_list = []
    vals_list = []

    # ------------------------------------------------------------------
    # Temporal edges — source block: K_a in upper-left, +1 bottom-right
    # ------------------------------------------------------------------

    # Source Koopman block: for each action a, select subset of temporal edges.
    for a in range(N_ACTIONS):
        mask   = (actions == a).nonzero(as_tuple=True)[0]   # [E_a]
        if mask.numel() == 0:
            continue
        E_a    = mask.numel()
        # Row base for edge group (each edge occupies `stalk` rows)
        row_base = mask * stalk                               # [E_a]
        # Col base for source nodes
        col_base = src_nodes[mask] * stalk                   # [E_a]

        Ka = K_mats[a].to(device)   # [d, d]
        # Expand K_a for all E_a edges: [E_a, d, d]
        Ka_exp = Ka.unsqueeze(0).expand(E_a, d, d)           # [E_a, d, d]

        # Build [E_a, d, d] index tensors:
        #   ri[e, i, j] = row_base[e] + i   (row i of the K_a block, any column j)
        #   ci[e, i, j] = col_base[e] + j   (column j of the K_a block)
        # K_a[i, j] occupies position (row_base+i, col_base+j) in B.
        r_off = torch.arange(d, device=device)                 # [d]
        # ri: row index repeats across j dimension  →  [E_a, d, 1] → broadcast [E_a, d, d]
        ri = (row_base[:, None, None] + r_off[None, :, None]).expand(E_a, d, d)
        # ci: col index repeats across i dimension  →  [E_a, 1, d] → broadcast [E_a, d, d]
        ci = (col_base[:, None, None] + r_off[None, None, :]).expand(E_a, d, d)

        rows_list.append(ri.reshape(-1))
        cols_list.append(ci.reshape(-1))
        # Scale each edge's K_a block rows by W_sqrt[mask[e]]
        W_sqrt_a = W_sqrt[mask]   # [E_a]
        vals_list.append((Ka_exp * W_sqrt_a[:, None, None]).reshape(-1))

    # Source value slot: +1.0 at (edge*stalk + d, src_node*stalk + d)
    # Row index uses the edge counter (not src_node), since B rows are indexed by edge.
    edge_idx    = torch.arange(E_temp, device=device)
    src_val_row = edge_idx * stalk + d     # [E_temp]
    src_val_col = src_nodes * stalk + d    # [E_temp]
    rows_list.append(src_val_row)
    cols_list.append(src_val_col)
    vals_list.append(W_sqrt)   # [E_temp] — scaled source value slot

    # ------------------------------------------------------------------
    # Temporal edges — dest block: -I_d in upper-left, -gamma bottom-right
    # ------------------------------------------------------------------

    # Dest latent block (-I_d): diagonal entries only
    # For edge k, latent dimension i: row = k*stalk + i, col = dst_nodes[k]*stalk + i
    lat_i   = torch.arange(d, device=device)
    # edge_idx already = arange(E_temp)
    di_row  = (edge_idx[:, None] * stalk + lat_i[None, :]).reshape(-1)    # [E_temp * d]
    di_col  = (dst_nodes[:, None] * stalk + lat_i[None, :]).reshape(-1)   # [E_temp * d]
    rows_list.append(di_row)
    cols_list.append(di_col)
    # Scale each edge's -I_d block rows by W_sqrt[k]: [E_temp * d]
    vals_list.append(-W_sqrt.unsqueeze(1).expand(E_temp, d).reshape(-1))

    # Dest value slot: -gamma*(1-done) at (edge*stalk + d, dst_node*stalk + d)
    # Terminal edges (done=1) get 0 here — the dest node does not participate in
    # the value coupling, consistent with the (1-done) masking in td_targets.
    dst_val_row = edge_idx * stalk + d     # [E_temp]
    dst_val_col = dst_nodes * stalk + d    # [E_temp]
    rows_list.append(dst_val_row)
    cols_list.append(dst_val_col)
    vals_list.append(-gamma * W_sqrt * (1.0 - dones))   # [E_temp]

    # ------------------------------------------------------------------
    # Bisimulation edges — source +I_{stalk}, dest -I_{stalk}
    # ------------------------------------------------------------------
    if E_bisim > 0:
        bisim_edge_idx  = torch.arange(E_bisim, device=device)
        # Bisimulation edges occupy rows E_temp*stalk .. (E_temp+E_bisim)*stalk - 1
        bisim_row_base  = (E_temp + bisim_edge_idx) * stalk   # [E_bisim]

        stalk_i = torch.arange(stalk, device=device)

        # Source: +I_{stalk}
        bs_row = (bisim_row_base[:, None] + stalk_i[None, :]).reshape(-1)  # [E_bisim * stalk]
        bs_col = (bisim_src[:, None] * stalk + stalk_i[None, :]).reshape(-1)
        rows_list.append(bs_row)
        cols_list.append(bs_col)
        vals_list.append(torch.ones(E_bisim * stalk, device=device))

        # Dest: -I_{stalk}
        bd_row = (bisim_row_base[:, None] + stalk_i[None, :]).reshape(-1)  # same rows
        bd_col = (bisim_dst[:, None] * stalk + stalk_i[None, :]).reshape(-1)
        rows_list.append(bd_row)
        cols_list.append(bd_col)
        vals_list.append(torch.full((E_bisim * stalk,), -1.0, device=device))

    # ------------------------------------------------------------------
    # Assemble COO tensor
    # ------------------------------------------------------------------
    all_rows = torch.cat(rows_list)
    all_cols = torch.cat(cols_list)
    all_vals = torch.cat(vals_list)

    B_mat = torch.sparse_coo_tensor(
        torch.stack([all_rows, all_cols]),
        all_vals,
        size=(E_total * stalk, N * stalk),
        device=device,
    ).coalesce()

    return B_mat  # (legacy stub, no longer called)


# ---------------------------------------------------------------------------
# 4. Graph build + diffuse (called every GRAPH_REBUILD steps)
# ---------------------------------------------------------------------------

def build_and_diffuse(
    agent:        KoopmanAgent,
    target:       TargetNetwork,
    buf:          ReplayBuffer,
    graph_device: torch.device,   # kept for API compat; diffusion now runs on train_device
    train_device: torch.device,
) -> tuple | None:
    """
    1. Sample N_CHUNKS contiguous trajectory chunks of length T_CHUNK from buf.
       One chunk is anchored to a goal transition; the rest are random.
    2. Build node set: [src_states | dst_states], shape [2*M_SRC, STATE_DIM].
    3. Encode all nodes with target encoder → Z_tgt [2*M_SRC, d].
    4. Build k-NN bisimulation edges from Z_tgt (dedup keeping i < j).
    5. Compute W_sqrt (Koopman advantage edge weights).
    6. Run power iteration to find alpha = 1 / lambda_max (message passing).
    7. Build reward source R_v via scatter_add (replaces B^T R_edges).
    8. Run Richardson diffusion on value-only stalk (message passing).
    9. Return (all_states, V_diff, w_mean, w_min) on train_device.

    Chunk sampling is essential: flat random sampling shatters temporal topology
    into isolated 1-step arrows. Contiguous chunks create T_CHUNK-step "pipes"
    that carry reward T steps backward instantly; bisim edges stitch pipes together.
    """
    d = D
    N = 2 * M_SRC

    # ------------------------------------------------------------------
    # Step 1 & 2: Sample contiguous chunks to preserve temporal topology
    # ------------------------------------------------------------------
    graph_batch = buf.sample_chunks(N_CHUNKS, T_CHUNK, force_goal=FORCE_GOAL)
    if graph_batch is None:
        return None

    states_np  = graph_batch["states"]    # [M_SRC, 2]
    next_np    = graph_batch["next_s"]    # [M_SRC, 2]
    actions_np = graph_batch["actions"]   # [M_SRC]
    dones_np   = graph_batch["dones"]     # [M_SRC]
    rewards_np = graph_batch["rewards"]   # [M_SRC]

    all_np = np.concatenate([states_np, next_np], axis=0)   # [N, 2]
    all_t  = torch.from_numpy(all_np).to(train_device)      # [N, 2]

    # ------------------------------------------------------------------
    # Step 3: Encode with target network (on train_device)
    # ------------------------------------------------------------------
    target.encoder.to(train_device)
    target.v_net.to(train_device)
    with torch.no_grad():
        Z_tgt  = target.encoder(all_t)   # [N, d]
        V_init = target.v_net(Z_tgt)     # [N]

    # ------------------------------------------------------------------
    # Step 4: k-NN bisimulation edges
    # ------------------------------------------------------------------
    dists = torch.cdist(Z_tgt, Z_tgt, p=2)
    dists.fill_diagonal_(float("inf"))
    k_actual  = min(K_BISIM_NN, N - 1)
    _, nn_idx = dists.topk(k_actual, dim=1, largest=False)   # [N, k]

    row_idx   = torch.arange(N, device=nn_idx.device).unsqueeze(1).expand_as(nn_idx).reshape(-1)
    col_idx   = nn_idx.reshape(-1)
    keep      = row_idx < col_idx
    bisim_src = row_idx[keep]   # [E_bisim] on train_device
    bisim_dst = col_idx[keep]   # [E_bisim] on train_device

    # ------------------------------------------------------------------
    # Step 5: Edge indices and Koopman advantage weights W_sqrt
    # A_k = V(K_{a_k} z_k) - max_a V(K_a z_k)  <= 0
    # W_k = exp(A_k / TAU_GRAPH),  W_sqrt_k = sqrt(W_k)
    # ------------------------------------------------------------------
    src_nodes     = torch.arange(M_SRC, dtype=torch.long, device=train_device)
    dst_nodes     = torch.arange(M_SRC, N, dtype=torch.long, device=train_device)
    actions_t     = torch.from_numpy(actions_np).long().to(train_device)
    dones_t       = torch.from_numpy(dones_np).float().to(train_device)
    rewards_t     = torch.from_numpy(rewards_np).float().to(train_device)

    Z_src = Z_tgt[:M_SRC]   # [M_SRC, d]
    with torch.no_grad():
        z_taken       = torch.zeros_like(Z_src)
        V_all_actions = torch.zeros(M_SRC, N_ACTIONS, device=train_device)
        for a in range(N_ACTIONS):
            b_a  = agent.B.detach()[:, a]                                          # [d]
            z_Ka = F.normalize(Z_src @ agent.A.detach().T + b_a, dim=-1)  # [M_SRC, d]
            V_all_actions[:, a] = target.v_net(z_Ka)
            m = (actions_t == a)
            if m.any():
                z_taken[m] = z_Ka[m]
        V_taken = target.v_net(z_taken)
        V_max   = V_all_actions.max(dim=1).values
        W_sqrt  = torch.exp((V_taken - V_max) / TAU_GRAPH).sqrt()   # [M_SRC]

    # ------------------------------------------------------------------
    # Step 6: alpha via message-passing power iteration on value space
    # ------------------------------------------------------------------
    alpha = power_iteration_alpha(
        src_nodes, dst_nodes, W_sqrt, dones_t,
        bisim_src, bisim_dst,
        GAMMA, N, ALPHA_ITERS, train_device,
    )

    # ------------------------------------------------------------------
    # Step 7: Reward source in value space  (replaces B^T R_edges)
    # R_v[src_k] += W_sqrt[k] * r_k
    # R_v[dst_k] -= W_sqrt[k] * γ*(1-done_k) * r_k
    # ------------------------------------------------------------------
    coeff = GAMMA * (1.0 - dones_t)
    R_v   = torch.zeros(N, device=train_device)
    R_v.scatter_add_(0, src_nodes,  W_sqrt * rewards_t)
    R_v.scatter_add_(0, dst_nodes, -coeff * W_sqrt * rewards_t)

    # ------------------------------------------------------------------
    # Step 8: Richardson diffusion on value stalk only
    # ------------------------------------------------------------------
    V_graph = V_init.detach()   # [N] — initial values and Tikhonov anchor

    V_diff = richardson_diffuse(
        alpha, V_graph, R_v, V_graph,
        src_nodes, dst_nodes, W_sqrt, dones_t,
        bisim_src, bisim_dst,
        GAMMA, K_DIFFUSE, beta=BETA_TIKHONOV,
    )   # [N] on train_device

    V_MAX  = 1.0 / (1.0 - GAMMA)
    V_diff = V_diff.clamp(0.0, V_MAX)

    w_mean = W_sqrt.mean().item()
    w_min  = W_sqrt.min().item()

    graph_data = {
        # Node positions: first M_SRC = src states, last M_SRC = dst states
        "positions":  all_t.cpu().numpy(),          # [N, 2]  (x,y) ∈ [-1,1]²
        # Temporal edge connectivity
        "src_nodes":  src_nodes.cpu().numpy(),       # [M_SRC] ints
        "dst_nodes":  dst_nodes.cpu().numpy(),       # [M_SRC] ints
        "actions":    actions_np,                    # [M_SRC] int — action per edge
        "W_sqrt":     W_sqrt.cpu().numpy(),          # [M_SRC] float — edge trust weights
        "rewards":    rewards_np,                    # [M_SRC] float — immediate reward per edge
        # Bisimulation edge connectivity
        "bisim_src":  bisim_src.cpu().numpy(),       # [E_bisim] ints
        "bisim_dst":  bisim_dst.cpu().numpy(),       # [E_bisim] ints
        # Diffusion outputs
        "R_v":        R_v.cpu().numpy(),             # [N] float — reward injected per node
        "V_diff":     V_diff.cpu().numpy(),          # [N] float — diffused values
    }
    return all_t, V_diff, w_mean, w_min, graph_data


# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------

def train() -> dict:
    """
    Online DQN-style training with sheaf Richardson diffusion targets.

    Key differences from gravity_basin.py:
      - Flat BATCH_SIZE=256 transitions per step (no chunk sampling)
      - Every GRAPH_REBUILD steps: rebuild sparse graph and diffuse
      - L_v supervises V_ψ against sheaf-diffused values
      - Two optimizer param-groups: encoder+v_net at LR, K matrices at LR*0.5
      - No Tree-Backup backward recursion
    """
    env    = GravityBasin()
    buf    = ReplayBuffer(capacity=BUFFER_SIZE)
    agent  = KoopmanAgent()
    target = TargetNetwork(agent)

    # Two param groups: neural params at LR, dynamics (A, B) at lower LR + WD
    neural_params = list(agent.encoder.parameters()) + list(agent.v_net.parameters())
    koop_params   = [agent.A, agent.B]
    opt = optim.Adam([
        {"params": neural_params, "lr": LR},
        {"params": koop_params,   "lr": LR * KOOP_LR_SCALE, "weight_decay": KOOP_WD},
    ])

    # Move agent and target to DEVICE for neural forward passes.
    # Note: gravity_basin.KoopmanAgent.act() creates a hard-coded CPU tensor,
    # so for DEVICE != cpu we wrap act() to move the input tensor to DEVICE.
    agent.to(DEVICE)
    target.encoder.to(DEVICE)
    target.v_net.to(DEVICE)

    if DEVICE.type != "cpu":
        @torch.no_grad()
        def _act_device(self_ref, state: np.ndarray, epsilon: float = 0.0) -> int:
            if random.random() < epsilon:
                return random.randint(0, N_ACTIONS - 1)
            z = self_ref.encoder(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            )
            best_a, best_v = -1, -float("inf")
            for a in range(N_ACTIONS):
                b_a    = self_ref.B[:, a]
                z_next = F.normalize(z @ self_ref.A.T + b_a, dim=-1)
                v_next = self_ref.v_net(z_next).item()
                if v_next > best_v:
                    best_v, best_a = v_next, a
            return best_a
        # Bind the patched method to this specific agent instance only
        import types
        agent.act = types.MethodType(_act_device, agent)

    # Graph data: refreshed every GRAPH_REBUILD steps
    graph_states  = None   # [2*M_SRC, 2] on DEVICE
    graph_v_diff  = None   # [2*M_SRC]    on DEVICE
    graph_built   = False
    last_w_mean   = float("nan")
    last_w_min    = float("nan")
    last_graph_data = None

    state     = GravityBasin.reset()
    ep_return = 0.0
    ep_steps  = 0
    episode_returns = []
    koop_losses, v_losses, bisim_losses = [], [], []
    recent_koop, recent_v, recent_bisim = [], [], []
    t0 = time.time()

    print("=" * 68)
    print("  Ferrari Koopman-RL — Global Richardson Diffusion")
    print(f"  State: (x,y)∈[-1,1]²   GravityBasin   d={D}   device={DEVICE}")
    print(f"  Graph: {2*M_SRC} nodes  |  rebuilt every {GRAPH_REBUILD} steps")
    print(f"  Diffusion: {K_DIFFUSE} Richardson iterations")
    print("=" * 68)
    print(f"\n[Warmup: collecting {WARMUP} random transitions...]\n")

    min_buf = 2 * M_SRC + BATCH_SIZE   # need enough for graph + batch

    for step in range(1, N_STEPS + 1):
        # --- Epsilon-greedy action ---
        eps    = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY)
        action = agent.act(
            state,
            epsilon=(eps if step > WARMUP else 1.0)
        )

        # --- Environment step ---
        next_state, reward, done = GravityBasin.step(state, action)
        buf.push(state, action, reward, next_state, done)
        ep_return += reward
        ep_steps  += 1

        if done or ep_steps >= MAX_EP_STEPS:
            episode_returns.append(ep_return)
            ep_return, ep_steps = 0.0, 0
            state = GravityBasin.reset()
        else:
            state = next_state

        # --- Skip gradient steps during warmup or insufficient buffer ---
        if step <= WARMUP or not buf.ready(min_buf):
            continue

        # ----------------------------------------------------------------
        # Graph rebuild every GRAPH_REBUILD steps (The problem comes from here the graph diffusin is flooding the value estimates)
        # ----------------------------------------------------------------
        if step % GRAPH_REBUILD == 0:
            result = build_and_diffuse(agent, target, buf, GRAPH_DEVICE, DEVICE)
            if result is not None:
                graph_states, graph_v_diff, last_w_mean, last_w_min, last_graph_data = result
                visualize_graph(last_graph_data, step)
                if not graph_built:
                    E_temp   = M_SRC
                    k_actual = min(K_BISIM_NN, 2 * M_SRC - 1)
                    print(f"\n[Graph built] N={2*M_SRC} nodes  "
                          f"E_temp={E_temp}  "
                          f"k-NN={k_actual} per node  "
                          f"step={step}")
                    graph_built = True

        # ----------------------------------------------------------------
        # Mini-batch: flat random transitions for stable TD gradients
        # ----------------------------------------------------------------
        batch  = buf.sample_transitions(BATCH_SIZE)
        s_b    = torch.from_numpy(batch["states"]).to(DEVICE)           # [B, 2]
        ns_b   = torch.from_numpy(batch["next_s"]).to(DEVICE)           # [B, 2]
        a_b    = torch.from_numpy(batch["actions"]).long().to(DEVICE)   # [B]
        r_b    = torch.from_numpy(batch["rewards"]).to(DEVICE)          # [B]
        d_b    = torch.from_numpy(batch["dones"]).to(DEVICE)            # [B]

        # ----------------------------------------------------------------
        # Encode with online encoder
        # ----------------------------------------------------------------
        z_src = agent.encode(s_b)    # [B, d]

        # Target encoder embeddings for next states (no grad)
        with torch.no_grad():
            z_dst_tgt = target.encoder(ns_b)   # [B, d]

        # ----------------------------------------------------------------
        # Koopman predictions: K_{a_t} z_t for each transition
        # ----------------------------------------------------------------
        # z' = A z + B a  (unnormalised) — a is one-hot, B a = B[:, action_idx]
        z_A    = z_src @ agent.A.T                           # [B, d] shared term
        b_all  = agent.B.T[a_b]                              # [B, d] per-sample action column
        z_pred = z_A + b_all                                 # [B, d]

        # ----------------------------------------------------------------
        # Loss 1 — Koopman linearity: ||K_a z_t - z_{t+1}||^2
        # Mask out terminal transitions (no valid s_{t+1} after done)
        # ----------------------------------------------------------------
        # Both z_pred and z_dst_tgt are unit-norm (on S^{d-1}).
        # L_koop = ||normalize(K_a z_src) - z_dst_target||^2 in angle space.
        koop_mask = 1.0 - d_b                                          # [B]
        L_koop    = ((z_pred - z_dst_tgt.detach()).pow(2)
                      .sum(dim=-1)
                      .mul(koop_mask)).mean()

        # ----------------------------------------------------------------
        # Loss 2a — Local 1-Step Double TD Learning (The Safety Net)
        # ----------------------------------------------------------------
        with torch.no_grad():
            # 1. SELECT best action using the ONLINE network
            # z' = A z + B e_a  (unnormalised) for all a at once
            z_A_next   = z_dst_tgt @ agent.A.detach().T                   # [B, d]
            B_cols     = agent.B.detach().T                                # [A, d]
            # broadcast: [B,1,d] + [1,A,d] → [B,A,d]
            z_next_all = F.normalize(
                z_A_next.unsqueeze(1) + B_cols.unsqueeze(0), dim=-1
            )   # [B, A, d]

            Bs, A_size, d_dim = z_next_all.shape
            v_all_next   = agent.v_net(z_next_all.reshape(Bs * A_size, d_dim)).reshape(Bs, A_size)
            best_actions = v_all_next.argmax(dim=1)

            # 2. EVALUATE that action using the TARGET network
            v_target_next = target.v_net(z_next_all.reshape(Bs * A_size, d_dim)).reshape(Bs, A_size)
            
            # Gather the target value of the action chosen by the online net
            V_next_double = v_target_next.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # In Loss 2a (Local TD)
            y_td = r_b + GAMMA * V_next_double * (1.0 - d_b)
            # ONLY clamp the ceiling. Let it be negative!

        V_pred_local = agent.v_net(z_src)
        L_v_local = (V_pred_local - y_td).pow(2).mean()

        # ----------------------------------------------------------------
        # Loss 2b — Global Sheaf Diffusion
        # Online encoder for the value loss — must match what act() uses.
        # V_ψ is trained to map online encoder embeddings to diffusion targets;
        # act() also queries V_ψ on online encoder embeddings → consistent.
        # ----------------------------------------------------------------
        if graph_states is not None and graph_v_diff is not None:
            V_pred_graph = agent.v_net(agent.encode(graph_states))

            L_v_global = (V_pred_graph - graph_v_diff.detach()).pow(2).mean()
        else:
            L_v_global = torch.tensor(0.0, device=DEVICE)

        # TD loss disabled — pure global sheaf diffusion only
        L_v = L_v_global
        # L_v = L_v_local + L_v_global

        # ----------------------------------------------------------------
        # Loss 3 — Bisimulation metric (disabled)
        # ----------------------------------------------------------------
        # In sparse reward settings |r_i - r_j| = 0 for ~99% of pairs,
        # which causes bisim to contract all non-goal embeddings to zero
        # and destroy Koopman geometry. Kept at 0 until a fix is found.
        L_bisim = torch.tensor(0.0, device=DEVICE)
       

        # ----------------------------------------------------------------
        # Combined loss and gradient step
        # ----------------------------------------------------------------
        loss = LAMBDA_KOOP * L_koop + LAMBDA_V * L_v + LAMBDA_BISIM * L_bisim
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
        opt.step()
        target.update(agent, tau=EMA_TAU)

        recent_koop.append(L_koop.item())
        recent_v.append(L_v.item())
        recent_bisim.append(L_bisim.item())

        # ----------------------------------------------------------------
        # Periodic logging
        # ----------------------------------------------------------------
        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            sps     = LOG_EVERY / elapsed
            t0      = time.time()
            recent20  = episode_returns[-20:] if episode_returns else []
            success   = sum(r == 1.0 for r in recent20)
            mk        = np.mean(recent_koop)
            mv        = np.mean(recent_v)
            mb        = np.mean(recent_bisim)
            vd_max    = graph_v_diff.max().item()  if graph_v_diff is not None else float("nan")
            vd_mean   = graph_v_diff.mean().item() if graph_v_diff is not None else float("nan")
            print(f"  step {step:6d}  ε={eps:.3f}  "
                  f"L_koop={mk:.4f}  L_v={mv:.4f}  L_bisim={mb:.4f}  "
                  f"succ/20={success}  sps={sps:.0f}  "
                  f"W̄={last_w_mean:.3f}  Wmin={last_w_min:.3f}  "
                  f"Vd_max={vd_max:.3f}  Vd_μ={vd_mean:.3f}")
            koop_losses.append(mk)
            v_losses.append(mv)
            bisim_losses.append(mb)
            recent_koop.clear()
            recent_v.clear()
            recent_bisim.clear()

        if step % PLOT_EVERY == 0:
            plot_live(step, agent, koop_losses, v_losses, bisim_losses,
                      episode_returns, graph_v_diff)
            print(f"  [plot saved → koopman_rl_live.png]")

    return {
        "agent":           agent,
        "koop_losses":     koop_losses,
        "v_losses":        v_losses,
        "bisim_losses":    bisim_losses,
        "episode_returns": episode_returns,
    }


# ---------------------------------------------------------------------------
# 6. Visualisation
# ---------------------------------------------------------------------------

def visualize_graph(graph_data: dict, step: int) -> None:
    """
    Save a 1×2 diagnostic figure to sheaf_graph_live.png.
    Left:  graph topology — action-colored arrows, bisim bridges, nodes by V_diff.
    Right: diffused values — W_sqrt gray edges, gold stars at reward-injection nodes.
    Called every GRAPH_REBUILD steps.
    """
    from matplotlib.collections import LineCollection
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    positions  = graph_data["positions"]   # [N, 2]
    src_nodes  = graph_data["src_nodes"]   # [M_SRC]
    dst_nodes  = graph_data["dst_nodes"]   # [M_SRC]
    actions    = graph_data["actions"]     # [M_SRC]
    W_sqrt_arr = graph_data["W_sqrt"]      # [M_SRC]
    rewards    = graph_data["rewards"]     # [M_SRC]
    bisim_src  = graph_data["bisim_src"]   # [E_bisim]
    bisim_dst  = graph_data["bisim_dst"]   # [E_bisim]
    R_v        = graph_data["R_v"]         # [N]
    V_diff     = graph_data["V_diff"]      # [N]

    N       = positions.shape[0]
    E_bisim = bisim_src.shape[0]

    # Autoscale colormap to actual data range so small values are still visible.
    # (V_MAX = 1/(1-γ) = 20 is far too wide when rewards are sparse +1 events.)
    v_lo = 0.0
    v_hi = max(V_diff.max(), 1e-2)

    # Pre-build edge segment arrays — shape [E, 2, 2] for LineCollection
    temp_segs = np.stack([positions[src_nodes], positions[dst_nodes]], axis=1)
    if E_bisim > 0:
        bisim_segs = np.stack([positions[bisim_src], positions[bisim_dst]], axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ------------------------------------------------------------------
    # Left panel — Graph Topology
    # ------------------------------------------------------------------

    # Bisim edges (bottom layer): thin gray dashed
    if E_bisim > 0:
        bisim_lc = LineCollection(bisim_segs, colors="gray", linewidths=0.4,
                                  linestyles="dashed", alpha=0.15, zorder=1)
        ax1.add_collection(bisim_lc)

    # Temporal edges: per-edge RGBA (color=action, alpha∝W_sqrt), width∝W_sqrt
    alphas = np.clip(W_sqrt_arr, 0.1, 1.0)
    widths = 0.3 + 1.2 * W_sqrt_arr   # range [0.3, 1.5]
    edge_rgba = np.array([
        (*mcolors.to_rgb(ACTION_COLORS[int(a)]), float(alphas[i]))
        for i, a in enumerate(actions)
    ])
    temp_lc = LineCollection(temp_segs, colors=edge_rgba, linewidths=widths, zorder=2)
    ax1.add_collection(temp_lc)

    # Direction arrows for high-confidence edges (W_sqrt > 0.5)
    hi_mask = W_sqrt_arr > 0.5
    if hi_mask.any():
        hi_sp  = positions[src_nodes[hi_mask]]
        hi_dp  = positions[dst_nodes[hi_mask]]
        hi_act = actions[hi_mask]
        ax1.quiver(hi_sp[:, 0], hi_sp[:, 1],
                   hi_dp[:, 0] - hi_sp[:, 0], hi_dp[:, 1] - hi_sp[:, 1],
                   color=[ACTION_COLORS[int(a)] for a in hi_act],
                   alpha=0.7, scale=1, scale_units="xy",
                   width=0.003, headwidth=6, headlength=6, zorder=3)

    # Nodes colored by V_diff
    ax1.scatter(positions[:, 0], positions[:, 1],
                c=V_diff, cmap="plasma", s=15, zorder=4, vmin=v_lo, vmax=v_hi)

    # Overdraw reward-carrying edges in gold
    rew_mask = rewards > 0
    if rew_mask.any():
        rsp = positions[src_nodes[rew_mask]]
        rdp = positions[dst_nodes[rew_mask]]
        ax1.quiver(rsp[:, 0], rsp[:, 1],
                   rdp[:, 0] - rsp[:, 0], rdp[:, 1] - rsp[:, 1],
                   color="gold", alpha=0.9, scale=1, scale_units="xy",
                   width=0.005, headwidth=7, headlength=7, zorder=5)

    ax1.add_patch(_goal_patch())
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.set_aspect("equal")
    action_patches = [mpatches.Patch(color=ACTION_COLORS[a], label=ACTION_NAMES[a])
                      for a in range(N_ACTIONS)]
    bisim_line = plt.Line2D([0], [0], color="gray", linestyle="dashed",
                            linewidth=0.8, label=f"bisim ({E_bisim})")
    ax1.legend(handles=action_patches + [bisim_line], fontsize=7,
               loc="lower left", ncol=2)
    ax1.set_title(f"Graph topology  |  step {step}  |  {N} nodes  {E_bisim} bisim edges")

    # ------------------------------------------------------------------
    # Right panel — Reward Flow & Diffused Values
    # ------------------------------------------------------------------

    # Temporal edges: gray colormap by W_sqrt (white=0, black=1)
    w_colors = plt.cm.Greys(np.clip(W_sqrt_arr, 0.0, 1.0))
    temp_lc2 = LineCollection(temp_segs, colors=w_colors, linewidths=0.8,
                               alpha=0.4, zorder=1)
    ax2.add_collection(temp_lc2)

    if E_bisim > 0:
        bisim_lc2 = LineCollection(bisim_segs, colors="gray", linewidths=0.4,
                                   linestyles="dashed", alpha=0.1, zorder=1)
        ax2.add_collection(bisim_lc2)

    # Nodes colored by V_diff with colorbar
    sc2 = ax2.scatter(positions[:, 0], positions[:, 1],
                      c=V_diff, cmap="plasma", s=20, zorder=4, vmin=v_lo, vmax=v_hi)
    plt.colorbar(sc2, ax=ax2, label="V_diff")

    # Gold star markers for reward-injection nodes
    rew_node_mask = R_v > 1e-4
    if rew_node_mask.any():
        sizes = np.clip(R_v[rew_node_mask] * 200, 30, 300)
        ax2.scatter(positions[rew_node_mask, 0], positions[rew_node_mask, 1],
                    marker="*", c="gold", s=sizes, zorder=6,
                    edgecolors="orange", linewidths=0.5, label="R_v > 0")
        ax2.legend(fontsize=8, loc="lower left")

    ax2.add_patch(_goal_patch())
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_aspect("equal")
    stats = (f"V_diff: μ={V_diff.mean():.3f}  max={V_diff.max():.3f}\n"
             f"W_sqrt: μ={W_sqrt_arr.mean():.3f}  min={W_sqrt_arr.min():.3f}")
    ax2.text(0.02, 0.98, stats, transform=ax2.transAxes, fontsize=8,
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax2.set_title(f"Diffused values  |  β={BETA_TIKHONOV}  K={K_DIFFUSE} iters")

    plt.tight_layout()
    plt.savefig("sheaf_graph_live.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _smooth(x, w=10):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_live(
    step:            int,
    agent:           "KoopmanAgent",
    koop_losses:     list,
    v_losses:        list,
    bisim_losses:    list,
    episode_returns: list,
    graph_v_diff:    "torch.Tensor | None" = None,
) -> None:
    """
    Save a 2×2 monitoring figure to koopman_rl_live.png.
    Called every PLOT_EVERY steps during training.
    macOS Preview / most image viewers auto-refresh on file change.
    """
    cpu = torch.device("cpu")
    agent.to(cpu)   # briefly move for grid evaluation

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Koopman-RL  —  step {step:,}", fontsize=13)

    # --- Loss curves ---
    ax = axes[0, 0]
    if koop_losses:
        ax.semilogy(_smooth(koop_losses), label=r"$\mathcal{L}_{Koop}$",  color="steelblue")
        ax.semilogy(_smooth(v_losses),    label=r"$\mathcal{L}_{V}$",     color="tomato")
        ax.semilogy(_smooth(bisim_losses),label=r"$\mathcal{L}_{bisim}$", color="purple")
    ax.set_xlabel("Log interval")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Episode returns + diffused value histogram ---
    ax = axes[0, 1]
    if graph_v_diff is not None:
        vd = graph_v_diff.cpu().numpy()
        ax.hist(vd, bins=50, color="tomato", alpha=0.7, density=True)
        ax.axvline(vd.mean(), color="darkred", linestyle="--",
                   label=f"μ={vd.mean():.3f}  max={vd.max():.3f}")
        ax.set_xlabel("Diffused V")
        ax.set_ylabel("Density")
        ax.set_title("Sheaf-Diffused Value Distribution")
        ax.legend(fontsize=9)
    else:
        if episode_returns:
            ax.plot(episode_returns, alpha=0.25, color="gray", linewidth=0.6)
            w = min(50, len(episode_returns))
            if len(episode_returns) >= w:
                roll = np.convolve(episode_returns, np.ones(w) / w, mode="valid")
                ax.plot(np.arange(w - 1, len(episode_returns)), roll,
                        color="royalblue", linewidth=1.5, label=f"Rolling mean ({w} ep)")
            ax.legend()
        ax.set_title("Episode Returns")
        ax.set_xlabel("Episode"); ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)

    # --- Value map ---
    ax = axes[1, 0]
    try:
        XX, YY, V = _value_grid(agent, res=80)
        im = ax.pcolormesh(XX, YY, V, cmap="plasma", shading="auto")
        plt.colorbar(im, ax=ax, label=r"$V_\psi(f_\theta(s))$")
        ax.add_patch(_goal_patch(label="Goal zone"))
        ax.legend(loc="lower left", fontsize=8)
    except Exception:
        ax.text(0.5, 0.5, "Value grid unavailable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title(r"Learned Value Map $V(x,y)$")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # --- Koopman dynamics (PCA) ---
    ax = axes[1, 1]
    try:
        with torch.no_grad():
            test_pts = torch.tensor(
                np.array([[x, y] for x in np.linspace(-0.9, 0.9, 10)
                                 for y in np.linspace(-0.9, 0.9, 10)],
                         dtype=np.float32))
            Z = agent.encode(test_pts)
        Z_np = Z.numpy()
        Z_c  = Z_np - Z_np.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Z_c, full_matrices=False)
        proj = Z_c @ Vt[:2].T
        ax.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=10, zorder=2)
        for a in range(N_ACTIONS):
            with torch.no_grad():
                b_a    = agent.B[:, a]
                Z_next = F.normalize(Z @ agent.A.T + b_a, dim=-1).numpy()
            pn = (Z_next - Z_np.mean(axis=0)) @ Vt[:2].T
            ax.quiver(proj[:, 0], proj[:, 1],
                      pn[:, 0] - proj[:, 0], pn[:, 1] - proj[:, 1],
                      color=ACTION_COLORS[a], alpha=0.5,
                      scale=1, scale_units="xy", width=0.004, headwidth=5,
                      label=ACTION_NAMES[a])
        ax.legend(fontsize=7, loc="best")
    except Exception:
        ax.text(0.5, 0.5, "Dynamics plot unavailable", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Linear Dynamics (PCA-2D)")
    ax.set_xlabel("PC$_1$"); ax.set_ylabel("PC$_2$")

    plt.tight_layout()
    plt.savefig("koopman_rl_live.png", dpi=120)
    plt.close(fig)

    agent.to(DEVICE)   # move back for training


def plot_results(history: dict) -> None:
    agent       = history["agent"]
    koop_losses = history["koop_losses"]
    v_losses    = history["v_losses"]
    bisim_losses= history["bisim_losses"]
    ep_returns  = history["episode_returns"]

    # Move agent to cpu for plotting
    agent.to(torch.device("cpu"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ----------------------------------------------------------------
    # Subplot 1: Loss curves (log scale)
    # ----------------------------------------------------------------
    ax = axes[0, 0]
    ax.semilogy(_smooth(koop_losses), label=r"$\mathcal{L}_{Koop}$", color="steelblue")
    ax.semilogy(_smooth(v_losses),    label=r"$\mathcal{L}_{V}$ (sheaf diffusion)",
                color="tomato")
    ax.semilogy(_smooth(bisim_losses),label=r"$\mathcal{L}_{bisim}$", color="purple")
    ax.set_xlabel("Log interval")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Losses")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # Subplot 2: Episode returns
    # ----------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(ep_returns, alpha=0.25, color="gray", linewidth=0.6, label="Raw return")
    w2 = min(50, len(ep_returns))
    if len(ep_returns) >= w2:
        roll = np.convolve(ep_returns, np.ones(w2) / w2, mode="valid")
        ax.plot(np.arange(w2 - 1, len(ep_returns)), roll,
                color="royalblue", linewidth=1.5, label=f"Rolling mean ({w2} ep)")
    ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Episode Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ----------------------------------------------------------------
    # Subplot 3: Value map V(x,y)
    # ----------------------------------------------------------------
    ax = axes[1, 0]
    XX, YY, V = _value_grid(agent, res=100)
    im = ax.pcolormesh(XX, YY, V, cmap="plasma", shading="auto")
    plt.colorbar(im, ax=ax, label=r"$V_\psi(f_\theta(s))$")
    ax.add_patch(_goal_patch(label="Goal zone"))
    ax.set_title(r"Learned Value Map $V(x,y)$  [sheaf diffusion targets]")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect("equal")

    # ----------------------------------------------------------------
    # Subplot 4: Koopman dynamics in PCA-projected latent space
    # ----------------------------------------------------------------
    ax = axes[1, 1]
    with torch.no_grad():
        test_pts = torch.tensor(
            np.array([[x, y] for x in np.linspace(-0.9, 0.9, 12)
                             for y in np.linspace(-0.9, 0.9, 12)],
                     dtype=np.float32))
        Z = agent.encode(test_pts)   # [144, d]

    Z_np = Z.numpy()
    Z_c  = Z_np - Z_np.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Z_c, full_matrices=False)
    proj = Z_c @ Vt[:2].T   # [144, 2]

    ax.scatter(proj[:, 0], proj[:, 1], c="lightgray", s=12, zorder=2, label=r"$z_s$")
    for a in range(N_ACTIONS):
        with torch.no_grad():
            b_a    = agent.B[:, a]
            Z_next = F.normalize(Z @ agent.A.T + b_a, dim=-1)
        Z_next_np = Z_next.numpy()
        proj_n    = (Z_next_np - Z_np.mean(axis=0)) @ Vt[:2].T
        ax.quiver(proj[:, 0], proj[:, 1],
                  proj_n[:, 0] - proj[:, 0],
                  proj_n[:, 1] - proj[:, 1],
                  color=ACTION_COLORS[a], alpha=0.5,
                  scale=1, scale_units="xy", width=0.003, headwidth=5,
                  label=ACTION_NAMES[a])

    ax.legend(fontsize=8, loc="best")
    ax.set_title("Linear Dynamics in Latent Space (PCA-2D)\n"
                 r"Arrows: $\mathrm{normalize}(K_a z) \to$ predicted $z_{t+1}$")
    ax.set_xlabel("PC$_1$")
    ax.set_ylabel("PC$_2$")

    plt.tight_layout()
    plt.savefig("koopman_global_results.png", dpi=150)
    print("\nSaved -> koopman_global_results.png")
    plt.close()


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    history = train()

    agent = history["agent"]

    # Evaluate on the same device used for training (agent may be on MPS/CUDA).
    # plot_results() will move agent to CPU internally before plotting.
    sr, ms = evaluate(agent, n_episodes=100)
    print(f"\nFinal evaluation (100 episodes, greedy):")
    print(f"  Success rate : {sr*100:.1f}%")
    print(f"  Mean steps   : {ms:.1f}  (successful episodes only)")

    n_ep     = len(history["episode_returns"])
    n_succ   = sum(r == 1.0 for r in history["episode_returns"])
    mean_trn = np.mean(history["episode_returns"]) if n_ep else 0.0
    print(f"\n  Training episodes    : {n_ep}")
    print(f"  Training successes   : {n_succ}  ({100*n_succ/max(n_ep,1):.1f}%)")
    print(f"  Mean training return : {mean_trn:.2f}")

    plot_results(history)


if __name__ == "__main__":
    main()
