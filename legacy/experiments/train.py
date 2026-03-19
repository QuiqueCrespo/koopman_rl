"""
Training loop and evaluation for gravity basin Koopman-RL.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env    import GravityBasin, N_ACTIONS, STATE_DIM, MAX_EP_STEPS
from model  import KoopmanAgent, TargetNetwork, D, LR
from buffer import ReplayBuffer, B, T_CHUNK
from losses import (compute_v_targets, compute_bisimulation_loss,
                    compute_contrastive_loss, compute_isometric_loss,
                    LAMBDA_KOOP, LAMBDA_BISIM, LAMBDA_CONTRASTIVE,
                    LAMBDA_ISOMETRIC, TAU_START, TAU_END)

WARMUP    = 3_000
N_STEPS   = 100_000
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = 40_000
LOG_EVERY = 2_000


def train(hook=None, n_steps_override: int = None) -> dict:
    """
    Online Koopman-RL training loop.

    Returns history dict:
      agent, koop_losses, q_losses (alias for v_losses), episode_returns, episode_steps
    """
    env    = GravityBasin()
    buf    = ReplayBuffer()
    agent  = KoopmanAgent()
    target = TargetNetwork(agent)
    opt    = optim.Adam(agent.parameters(), lr=LR)

    state     = env.reset()
    ep_return = 0.0
    ep_steps  = 0
    episode_returns, episode_steps_log = [], []
    koop_losses, v_losses = [], []
    recent_koop, recent_v = [], []

    print("=" * 62)
    print("  Neural Q-Stalk Koopman-RL — 2D Gravity Basin")
    print(f"  State: (x,y)∈[-1,1]²   d={D}   B={B}×T={T_CHUNK}")
    print(f"  M matrix: {B} block-diagonal upper-bidiagonal chains")
    print(f"  Solve: backward recursion  O({B}·{T_CHUNK}) per step")
    print("=" * 62)
    print(f"\n[Warmup: collecting {WARMUP} random transitions...]\n")

    min_buf  = B * T_CHUNK + 2
    n_steps  = n_steps_override if n_steps_override is not None else N_STEPS

    for step in range(1, n_steps + 1):
        eps    = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY)
        action = agent.act(state, epsilon=eps if step > WARMUP else 1.0)

        next_state, reward, done = env.step(state, action)
        buf.push(state, action, reward, next_state, done)
        ep_return += reward
        ep_steps  += 1

        if done or ep_steps >= MAX_EP_STEPS:
            episode_returns.append(ep_return)
            episode_steps_log.append(ep_steps)
            ep_return, ep_steps = 0.0, 0
            state = env.reset()
        else:
            state = next_state

        if step <= WARMUP or not buf.ready(min_buf):
            continue

        # ----------------------------------------------------------------
        # Koopman forward pass
        # ----------------------------------------------------------------
        batch = buf.sample_chunks(B, T_CHUNK)
        s_all = batch["states"]    # [B, T+1, 2]
        a_bt  = batch["actions"]   # [B, T]
        r_bt  = batch["rewards"]   # [B, T]
        d_bt  = batch["dones"]     # [B, T]

        s_src = s_all[:, :-1, :]
        s_dst = s_all[:,  1:, :]

        N_flat  = B * T_CHUNK
        s_src_f = s_src.reshape(N_flat, STATE_DIM)
        s_dst_f = s_dst.reshape(N_flat, STATE_DIM)
        a_f     = a_bt.reshape(N_flat)
        d_f     = d_bt.reshape(N_flat)
        r_f     = r_bt.reshape(N_flat)

        # Step 1 — Target encoder: one pass for all T+1 states
        with torch.no_grad():
            s_flat        = s_all.reshape(-1, STATE_DIM)
            z_tgt_flat    = target.encoder(s_flat)
            v_tgt_flat    = target.v_net(z_tgt_flat)
            v_targets_all = v_tgt_flat.reshape(B, T_CHUNK + 1)
            z_dst_target  = z_tgt_flat.reshape(B, T_CHUNK + 1, -1)[:, 1:].reshape(N_flat, -1)

        # Step 2 — Online encoder
        z_src = agent.encode(s_src_f)
        z_dst = agent.encode(s_dst_f)

        # Step 3 — Koopman predictions K_{a_t} z_t
        z_pred = torch.zeros_like(z_src)
        for a in range(N_ACTIONS):
            mask = (a_f == a)
            if mask.any():
                z_pred[mask] = z_src[mask] @ agent.K[a].T

        # Step 4 — AWR weights (computed before recursion for firewall)
        # F.normalize matches act(): K_a may drift off SO(d), so we re-project
        # onto the unit sphere before querying V_net — same as inference path.
        tau = max(TAU_END, TAU_START - (TAU_START - TAU_END) * step / N_STEPS)
        with torch.no_grad():
            V_taken = agent.v_net(F.normalize(z_pred, p=2, dim=-1))
            V_max   = torch.stack([
                agent.v_net(F.normalize(z_src @ agent.K[a].T, p=2, dim=-1))
                for a in range(N_ACTIONS)
            ], dim=1).max(dim=1).values
            W = torch.exp((V_taken - V_max) / tau).reshape(B, T_CHUNK)

        # Step 5 — Tree-Backup V-targets
        with torch.no_grad():
            V_diff = compute_v_targets(r_bt, d_bt, v_targets_all, W)
        V_tgt_f = V_diff.reshape(N_flat)

        # Losses
        koop_mask = 1.0 - d_f
        L_koop    = ((z_pred - z_dst.detach()).pow(2)
                      .sum(dim=-1).mul(koop_mask)).mean()
        V_pred    = agent.v_net(z_src.detach())
        L_v       = (V_pred - V_tgt_f).pow(2).mean()
        L_bisim       = compute_bisimulation_loss(z_src, z_dst_target, r_f, d_f)
        L_contrastive = compute_contrastive_loss(z_src)
        L_isometric   = compute_isometric_loss(z_src, s_src_f)

        loss = (LAMBDA_KOOP        * L_koop
                + L_v
                + LAMBDA_BISIM    * L_bisim
                + LAMBDA_CONTRASTIVE * L_contrastive
                + LAMBDA_ISOMETRIC   * L_isometric)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
        opt.step()
        target.update(agent)

        recent_koop.append(L_koop.item())
        recent_v.append(L_v.item())

        if step % LOG_EVERY == 0:
            ret20   = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            success = sum(r == 1.0 for r in episode_returns[-20:])
            mk = np.mean(recent_koop)
            mv = np.mean(recent_v)
            mb = L_bisim.item()
            print(f"  step {step:6d}  ε={eps:.3f}  τ={tau:.3f}  "
                  f"L_koop={mk:.4f}  L_v={mv:.4f}  L_bisim={mb:.4f}  "
                  f"ret={ret20:.3f}  succ/20={success}")
            koop_losses.append(mk)
            v_losses.append(mv)
            recent_koop.clear()
            recent_v.clear()

            if hook is not None:
                hook.record(
                    step=step,
                    success_last20=success,
                    z_src=z_src,
                    W=W,
                    z_dst_target=z_dst_target,
                    r_f=r_f,
                )

    return {
        "agent":           agent,
        "koop_losses":     koop_losses,
        "q_losses":        v_losses,      # alias for benchmark.py compatibility
        "episode_returns": episode_returns,
        "episode_steps":   episode_steps_log,
    }


def evaluate(agent: KoopmanAgent, n_episodes: int = 100) -> tuple[float, float]:
    """Greedy policy rollout. Returns (success_rate, mean_steps_on_success)."""
    env = GravityBasin()
    successes, steps_list = 0, []
    for _ in range(n_episodes):
        state = env.reset()
        for t in range(MAX_EP_STEPS):
            action              = agent.act(state, epsilon=0.0)
            state, reward, done = env.step(state, action)
            if done:
                successes += 1
                steps_list.append(t + 1)
                break
    sr = successes / n_episodes
    ms = np.mean(steps_list) if steps_list else float("nan")
    return sr, ms


def main():
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    from plot import plot_results

    history = train()
    agent   = history["agent"]
    sr, ms  = evaluate(agent, n_episodes=200)

    print(f"\nFinal evaluation (200 episodes, greedy):")
    print(f"  Success rate:  {sr * 100:.1f}%")
    print(f"  Mean steps:    {ms:.1f}  (successful episodes only)")

    n_ep   = len(history["episode_returns"])
    n_succ = sum(r == 1.0 for r in history["episode_returns"])
    print(f"\n  Training episodes:  {n_ep}")
    print(f"  Training successes: {n_succ}  ({n_succ/n_ep*100:.1f}%)")

    plot_results(history)


if __name__ == "__main__":
    main()
