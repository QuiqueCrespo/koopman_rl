"""
Sheaf-RL: Directed Latent Value Iteration (main algorithm).

Contains:
  directed_value_iteration  — max-propagation over episodic memory graph
  build_and_propagate       — graph construction + VI
  train                     — full training loop (accepts Config or legacy dict)
  evaluate                  — greedy rollout evaluation
  evaluate_planner          — comparison of all planner variants
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from sheaf_rl.config import Config
from sheaf_rl.env import GravityBasin
from sheaf_rl.model import KoopmanGradientPlanner, TargetNetwork
from sheaf_rl.buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Directed Value Iteration
# ---------------------------------------------------------------------------

def directed_value_iteration(
    V_init:     torch.Tensor,   # [N]   target-net values (prior + lower bound)
    src:        torch.Tensor,   # [M]   temporal edge sources
    dst:        torch.Tensor,   # [M]   temporal edge destinations
    rewards:    torch.Tensor,   # [M]   immediate reward per edge
    dones:      torch.Tensor,   # [M]   terminal flags
    bisim_src:  torch.Tensor,   # [E]   bisim edge sources
    bisim_dst:  torch.Tensor,   # [E]   bisim edge destinations
    bisim_dist: torch.Tensor,   # [E]   L2 distance penalty per bisim edge
    gamma:      float,
    K_steps:    int,
) -> torch.Tensor:              # [N]   propagated values
    """
    Directed Bellman max-propagation over the episodic memory graph.

    Key properties:
    - Causal: value flows strictly backwards through temporal edges.
      V[src] ← r + γ V[dst].  V[dst] is read-only; no forward flooding.
    - Monotone: V >= V_init at every step (max with prior guarantees this).
    - Convergent: bounded above by 1/(1-γ); no step-size tuning needed.
    - Bisim: penalised by latent distance — value teleports from high-value
      states to similar ones, decaying with distance.
    """
    V = V_init.clone()

    for _ in range(K_steps):
        # Temporal backward Bellman: value flows src ← dst
        Q_temp = rewards + gamma * (1.0 - dones) * V[dst]   # [M]

        # Bisim teleport (undirected, penalised by latent distance)
        Q_to_src = V[bisim_dst] - bisim_dist   # [E]
        Q_to_dst = V[bisim_src] - bisim_dist   # [E]

        # Accumulate on V; nodes with no incoming edges keep current V (include_self=True)
        V_new = V.clone()
        V_new.scatter_reduce_(0, src, Q_temp, reduce="amax", include_self=True)
        if bisim_src.numel() > 0:
            V_new.scatter_reduce_(0, bisim_src, Q_to_src, reduce="amax", include_self=True)
            V_new.scatter_reduce_(0, bisim_dst, Q_to_dst, reduce="amax", include_self=True)

        V = V_new

    return V


# ---------------------------------------------------------------------------
# Graph build + propagate (called every graph_rebuild steps)
# ---------------------------------------------------------------------------

def build_and_propagate(
    agent:  KoopmanGradientPlanner,
    target: TargetNetwork,
    buf:    ReplayBuffer,
    cfg:    Config,
    device: torch.device,
) -> tuple | None:
    """
    1. Sample n_chunks contiguous trajectory chunks of length t_chunk.
    2. Build node set: [src_states | dst_states], shape [2*m_src, state_dim].
    3. Encode all nodes with target encoder → Z_tgt [2*m_src, d].
    4. Build k-NN bisim edges; record pairwise L2 distances as penalties.
    5. Run directed_value_iteration k_diffuse times.
    6. Return (all_states, V_diff, graph_data) on device.
    """
    a = cfg.algo
    n_chunks = a.n_chunks
    t_chunk  = a.t_chunk
    m_src    = n_chunks * t_chunk
    N        = 2 * m_src

    graph_batch = buf.sample_chunks(n_chunks, t_chunk,
                                    force_goal=a.force_goal,
                                    stratified=a.stratified)
    if graph_batch is None:
        return None

    states_np  = graph_batch["states"]    # [m_src, state_dim]
    next_np    = graph_batch["next_s"]    # [m_src, state_dim]
    actions_np = graph_batch["actions"]   # [m_src]
    dones_np   = graph_batch["dones"]     # [m_src]
    rewards_np = graph_batch["rewards"]   # [m_src]

    all_np = np.concatenate([states_np, next_np], axis=0)   # [N, state_dim]
    all_t  = torch.from_numpy(all_np).to(device)            # [N, state_dim]

    target.encoder.to(device)
    target.v_net.to(device)
    with torch.no_grad():
        Z_tgt  = target.encoder(all_t)   # [N, d]
        V_init = target.v_net(Z_tgt)     # [N]

    # k-NN bisim edges + pairwise L2 distances as penalties
    k_bisim = a.k_bisim_nn
    if k_bisim > 0:
        dists = torch.cdist(Z_tgt, Z_tgt, p=2)
        dists.fill_diagonal_(float("inf"))
        k_actual          = min(k_bisim, N - 1)
        top_dists, nn_idx = dists.topk(k_actual, dim=1, largest=False)

        row_idx    = torch.arange(N, device=device).unsqueeze(1).expand_as(nn_idx).reshape(-1)
        col_idx    = nn_idx.reshape(-1)
        edge_dists = top_dists.reshape(-1)

        keep       = row_idx < col_idx
        bisim_src  = row_idx[keep]
        bisim_dst  = col_idx[keep]
        bisim_dist = edge_dists[keep] * a.bisim_penalty_scale
    else:
        empty      = torch.zeros(0, dtype=torch.long, device=device)
        bisim_src  = empty
        bisim_dst  = empty
        bisim_dist = torch.zeros(0, device=device)

    src_nodes = torch.arange(m_src, dtype=torch.long, device=device)
    dst_nodes = torch.arange(m_src, N, dtype=torch.long, device=device)
    dones_t   = torch.from_numpy(dones_np).float().to(device)
    rewards_t = torch.from_numpy(rewards_np).float().to(device)

    V_diff = directed_value_iteration(
        V_init, src_nodes, dst_nodes, rewards_t, dones_t,
        bisim_src, bisim_dst, bisim_dist,
        a.gamma, a.k_diffuse,
    )

    V_MAX  = 1.0 / (1.0 - a.gamma)
    V_diff = V_diff.clamp(0.0, V_MAX)

    R_v = torch.zeros(N, device=device)
    R_v.scatter_add_(0, src_nodes, rewards_t)

    graph_data = {
        "positions":  all_t.cpu().numpy(),
        "src_nodes":  src_nodes.cpu().numpy(),
        "dst_nodes":  dst_nodes.cpu().numpy(),
        "actions":    actions_np,
        "rewards":    rewards_np,
        "bisim_src":  bisim_src.cpu().numpy(),
        "bisim_dst":  bisim_dst.cpu().numpy(),
        "bisim_dist": bisim_dist.cpu().numpy(),
        "R_v":        R_v.cpu().numpy(),
        "V_diff":     V_diff.cpu().numpy(),
    }
    return all_t, V_diff, graph_data


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg=None) -> dict:
    """
    Full Sheaf-RL directed VI training loop.

    Accepts:
      cfg=None              → uses default Config()
      cfg=Config(...)       → typed config
      cfg={"KEY": value}    → legacy flat ablation dict (mapped via from_ablation_dict)
    """
    if isinstance(cfg, dict):
        cfg = Config.from_ablation_dict(cfg)
    elif cfg is None:
        cfg = Config()

    device = _resolve_device(cfg.device)
    a      = cfg.algo
    t      = cfg.train
    m      = cfg.model

    # Module-level constants still used from gravity_basin shim in legacy code;
    # here we read directly from cfg.
    GAMMA          = a.gamma
    LAMBDA_KOOP    = a.lambda_koop
    LAMBDA_BISIM   = 0.0   # bisim is disabled in directed VI (uses bisim graph edges instead)
    LAMBDA_RECON   = a.lambda_recon
    LAMBDA_V       = a.lambda_v
    KOOP_LR_SCALE  = a.koop_lr_scale
    KOOP_WD        = 0.0
    BATCH_SIZE     = cfg.buffer.batch_size

    env    = GravityBasin(cfg.env)
    buf    = ReplayBuffer.from_cfg(cfg)
    agent  = KoopmanGradientPlanner.from_cfg(cfg)

    if a.no_normalize:
        agent.encoder.no_normalize = True

    target = TargetNetwork(agent)   # deepcopy inherits no_normalize if set

    neural_params = (list(agent.encoder.parameters()) +
                     list(agent.v_net.parameters()) +
                     list(agent.decoder.parameters()))
    koop_params   = [agent.B] if a.fix_a else agent.koop_parameters()
    opt = optim.Adam([
        {"params": neural_params, "lr": m.lr},
        {"params": koop_params,   "lr": m.lr * KOOP_LR_SCALE, "weight_decay": KOOP_WD},
    ])

    agent.to(device)
    target.encoder.to(device)
    target.v_net.to(device)

    m_src = a.n_chunks * a.t_chunk
    N_nodes = 2 * m_src

    graph_states    = None
    graph_v_diff    = None
    graph_built     = False
    last_graph_data = None

    state     = env.reset()
    ep_return = 0.0
    ep_steps  = 0
    episode_returns = []
    koop_losses, v_losses, bisim_losses = [], [], []
    recent_koop, recent_v, recent_bisim = [], [], []
    t0 = time.time()

    print("=" * 68)
    print("  Ferrari Sheaf-RL v2 — Directed Latent Value Iteration")
    print(f"  State: (x,y)∈[-1,1]²   GravityBasin   d={m.d}   device={device}")
    print(f"  Graph: {N_nodes} nodes  |  rebuilt every {a.graph_rebuild} steps")
    print(f"  Value iter: {a.k_diffuse} directed Bellman steps")
    print("=" * 68)
    print(f"\n[Warmup: collecting {t.warmup} random transitions...]\n")

    min_buf = 2 * a.n_chunks * a.t_chunk + BATCH_SIZE

    for step in range(1, t.n_steps + 1):
        eps    = max(t.eps_end, t.eps_start - (t.eps_start - t.eps_end) * step / t.eps_decay)
        action = agent.act(state, epsilon=(eps if step > t.warmup else 1.0))

        next_state, reward, done = env.step(state, action)
        buf.push(state, action, reward, next_state, done)
        ep_return += reward
        ep_steps  += 1

        if done or ep_steps >= env.max_ep_steps:
            episode_returns.append(ep_return)
            ep_return, ep_steps = 0.0, 0
            state = env.reset()
        else:
            state = next_state

        if step <= t.warmup or not buf.ready(min_buf):
            continue

        # Graph rebuild
        if step % a.graph_rebuild == 0 and not a.no_graph:
            result = build_and_propagate(agent, target, buf, cfg, device)
            if result is not None:
                graph_states, graph_v_diff, last_graph_data = result
                if not graph_built:
                    print(f"\n[Graph built] N={N_nodes} nodes  "
                          f"E_temp={m_src}  k-NN={min(a.k_bisim_nn, N_nodes-1)}  step={step}")
                    graph_built = True
                try:
                    from sheaf_rl.viz import visualize_graph
                    visualize_graph(last_graph_data, step, cfg)
                except Exception:
                    pass

        # Mini-batch: flat random transitions
        batch = buf.sample_transitions(BATCH_SIZE)
        s_b   = torch.from_numpy(batch["states"]).to(device)
        ns_b  = torch.from_numpy(batch["next_s"]).to(device)
        a_b   = torch.from_numpy(batch["actions"]).long().to(device)
        r_b   = torch.from_numpy(batch["rewards"]).to(device)
        d_b   = torch.from_numpy(batch["dones"]).to(device)

        z_src = agent.encode(s_b)
        with torch.no_grad():
            z_dst_tgt = target.encoder(ns_b)

        # Loss 1 — Koopman: || dyn_step(z, b_a) − z_{t+1}_target ||²
        z_pred    = agent.dyn_step(z_src, agent.B.T[a_b])
        koop_mask = 1.0 - d_b
        L_koop    = ((z_pred - z_dst_tgt.detach()).pow(2)
                      .sum(dim=-1).mul(koop_mask)).mean()

        # Loss 2 — Reconstruction
        L_recon = (agent.decoder(z_src) - s_b).pow(2).mean()

        # Loss 3a — Local 1-step Double TD (safety net)
        with torch.no_grad():
            z_A_next   = z_dst_tgt @ agent.A.detach().T
            B_cols     = agent.B.detach().T                         # [n_actions, d]
            raw_next   = z_A_next.unsqueeze(1) + B_cols.unsqueeze(0)  # [Bs, A, d]
            Bs, A_size, d_dim = raw_next.shape
            if agent._ortho_a:
                z_next_all = raw_next
            else:
                z_next_all = F.normalize(
                    raw_next.reshape(Bs * A_size, d_dim), dim=-1
                ).reshape(Bs, A_size, d_dim)
            v_all      = agent.v_net(z_next_all.reshape(Bs * A_size, d_dim)).reshape(Bs, A_size)
            best_a_idx = v_all.argmax(dim=1)
            v_tgt      = target.v_net(z_next_all.reshape(Bs * A_size, d_dim)).reshape(Bs, A_size)
            V_next     = v_tgt.gather(1, best_a_idx.unsqueeze(1)).squeeze(1)
            y_td       = r_b + GAMMA * V_next * (1.0 - d_b)

        L_v_local = (agent.v_net(z_src) - y_td).pow(2).mean()

        # Loss 3b — Global directed VI targets
        if graph_states is not None and graph_v_diff is not None:
            V_pred_graph = agent.v_net(agent.encode(graph_states))
            L_v_global   = (V_pred_graph - graph_v_diff.detach()).pow(2).mean()
        else:
            L_v_global = torch.tensor(0.0, device=device)

        if a.no_graph:
            L_v = L_v_local
        elif a.td_plus_vi:
            L_v = L_v_local + L_v_global
        else:
            L_v = L_v_global

        L_bisim = torch.tensor(0.0, device=device)

        loss = LAMBDA_KOOP * L_koop + LAMBDA_V * L_v + LAMBDA_BISIM * L_bisim + LAMBDA_RECON * L_recon
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
        opt.step()
        target.update(agent, tau=cfg.model.ema_tau)

        recent_koop.append(L_koop.item())
        recent_v.append(L_v.item())
        recent_bisim.append(L_recon.item())   # reuse bisim slot for L_recon

        if step % t.log_every == 0:
            elapsed  = time.time() - t0
            sps      = t.log_every / elapsed
            t0       = time.time()
            recent20 = episode_returns[-20:] if episode_returns else []
            success  = sum(r > 0 for r in recent20)
            mk       = np.mean(recent_koop)
            mv       = np.mean(recent_v)
            mr       = np.mean(recent_bisim)
            vd_max   = graph_v_diff.max().item()  if graph_v_diff is not None else float("nan")
            vd_mean  = graph_v_diff.mean().item() if graph_v_diff is not None else float("nan")
            print(f"  step {step:6d}  ε={eps:.3f}  "
                  f"L_koop={mk:.4f}  L_v={mv:.4f}  L_recon={mr:.4f}  "
                  f"succ/20={success}  sps={sps:.0f}  "
                  f"Vd_max={vd_max:.3f}  Vd_μ={vd_mean:.3f}")
            koop_losses.append(mk)
            v_losses.append(mv)
            bisim_losses.append(mr)
            recent_koop.clear(); recent_v.clear(); recent_bisim.clear()

        if step % t.plot_every == 0:
            try:
                from sheaf_rl.viz import plot_live
                plot_live(step, agent, koop_losses, v_losses, bisim_losses,
                          episode_returns, graph_v_diff, cfg=cfg)
            except Exception:
                pass

    return {
        "agent":           agent,
        "koop_losses":     koop_losses,
        "v_losses":        v_losses,
        "bisim_losses":    bisim_losses,
        "episode_returns": episode_returns,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent, cfg=None, n_episodes: int = 100) -> tuple:
    """
    Greedy policy rollout. Returns (success_rate, mean_steps_on_success).

    cfg is optional — if omitted uses default EnvConfig values.
    """
    from sheaf_rl.env import GravityBasin
    env         = GravityBasin(cfg.env if cfg else None)
    max_steps   = env.max_ep_steps
    successes   = 0
    steps_list  = []

    for _ in range(n_episodes):
        state = env.reset()
        for t in range(max_steps):
            action              = agent.act(state, epsilon=0.0)
            state, reward, done = env.step(state, action)
            if done:
                successes += 1
                steps_list.append(t + 1)
                break

    sr = successes / n_episodes
    ms = float(np.mean(steps_list)) if steps_list else float("nan")
    return sr, ms


# ---------------------------------------------------------------------------
# Planner evaluation (convenience wrapper used by ablation_runner shim)
# ---------------------------------------------------------------------------

def evaluate_planner(agent, cfg=None, n_episodes: int = 100,
                     horizon: int = 10, plan_iters: int = 20) -> dict:
    """Compare all planner variants: greedy, gumbel, softmax, shooting, beam."""
    from sheaf_rl.planner import (
        plan_action_gumbel, plan_action_gumbel_cumulative,
        plan_action_softmax, plan_action_softmax_cumulative,
        plan_action_shooting, plan_action_beam,
    )
    from sheaf_rl.env import GravityBasin

    env       = GravityBasin(cfg.env if cfg else None)
    max_steps = env.max_ep_steps

    results = {
        "greedy":       [],
        "gumbel":       [],
        "gumbel_cumul": [],
        "softmax":      [],
        "softmax_cumul":[],
        "shooting_200": [],
        "beam_8":       [],
    }

    for mode, act_fn in [
        ("greedy",        lambda s: agent.act(s, epsilon=0.0)),
        ("gumbel",        lambda s: plan_action_gumbel(agent, s, horizon, plan_iters)),
        ("gumbel_cumul",  lambda s: plan_action_gumbel_cumulative(agent, s, horizon, plan_iters)),
        ("softmax",       lambda s: plan_action_softmax(agent, s, horizon, plan_iters)),
        ("softmax_cumul", lambda s: plan_action_softmax_cumulative(agent, s, horizon, plan_iters)),
        ("shooting_200",  lambda s: plan_action_shooting(agent, s, horizon, n_samples=200)),
        ("beam_8",        lambda s: plan_action_beam(agent, s, horizon, beam_width=8)),
    ]:
        print(f"\n  [{mode}] evaluating {n_episodes} episodes...", flush=True)
        for _ in range(n_episodes):
            s   = env.reset()
            ret = 0.0
            for _ in range(max_steps):
                a       = act_fn(s)
                s, r, d = env.step(s, a)
                ret    += r
                if d:
                    break
            results[mode].append(ret)

    return results
