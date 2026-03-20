"""
Generic training loop for continuous-action gymnasium environments.

All hyperparameters come from a Config object; env-specific logic is
injected via the on_viz callback.
"""

import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from koopman_rl.config import Config
from koopman_rl.model import KoopmanGradientPlanner, TargetNetwork
from koopman_rl.buffer import ContinuousReplayBuffer
from koopman_rl.noise import OUNoise
from koopman_rl.checkpoint import save_checkpoint, save_checkpoint_async

# Number of evaluation episodes in the end-of-training planner benchmark
_N_EVAL_PLAN = 20


def train_continuous(
    cfg: Config,
    env_id: str,
    device: torch.device,
    on_viz=None,
) -> dict:
    """
    Train KoopmanGradientPlanner on a continuous-action gymnasium environment.

    Parameters
    ----------
    cfg     : Config with all hyperparameters (see config.py).
    env_id  : gymnasium env id, e.g. "Pendulum-v1".
    device  : torch device to train on.
    on_viz  : optional callable invoked every cfg.train.viz_every steps.
              Called with keyword args:
                episode_returns, koop_log, v_log, step, ortho_err, agent, buf

    Returns
    -------
    dict with keys: agent, target, episode_returns, koop_log, v_log, buf
    """
    # ── Seed ─────────────────────────────────────────────────────────────────
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # ── Unpack config ─────────────────────────────────────────────────────────
    state_dim    = cfg.env.state_dim
    action_dim   = cfg.env.n_actions
    action_scale = cfg.env.action_scale
    d            = cfg.model.d
    lr           = cfg.model.lr
    ema_tau      = cfg.model.ema_tau
    gamma        = cfg.algo.gamma
    lambda_koop  = cfg.algo.lambda_koop
    lambda_recon = cfg.algo.lambda_recon
    lambda_v     = cfg.algo.lambda_v
    koop_lr_scale = cfg.algo.koop_lr_scale
    reward_scale = cfg.algo.reward_scale
    n_envs       = cfg.algo.n_envs
    capacity     = cfg.buffer.capacity
    batch_size   = cfg.buffer.batch_size
    n_steps      = cfg.train.n_steps
    warmup       = cfg.train.warmup
    noise_start  = cfg.train.noise_start
    noise_end    = cfg.train.noise_end
    noise_decay  = cfg.train.noise_decay
    viz_every    = cfg.train.viz_every
    ckpt_dir     = cfg.train.ckpt_dir
    plan_horizon = cfg.planner.horizon
    plan_iters   = cfg.planner.plan_iters
    sequential   = cfg.train.planner_type == "sequential"
    cumulative   = cfg.train.cumulative
    frozen_b     = cfg.train.frozen_b
    ou_noise     = cfg.train.ou_noise

    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Model + buffer ────────────────────────────────────────────────────────
    env    = gym.make_vec(env_id, num_envs=n_envs)
    agent  = KoopmanGradientPlanner(state_dim=state_dim, d=d,
                                    n_actions=action_dim,
                                    ortho_a=True, device=device)
    target = TargetNetwork(agent)
    buf    = ContinuousReplayBuffer(capacity, state_dim, action_dim)

    neural_params = (list(agent.encoder.parameters()) +
                     list(agent.v_net.parameters()) +
                     list(agent.decoder.parameters()))
    koop_params   = agent.koop_parameters()
    opt = optim.Adam([
        {"params": neural_params, "lr": lr},
        {"params": koop_params,   "lr": lr * koop_lr_scale},
    ])

    agent.to(device)
    target.encoder.to(device)
    target.v_net.to(device)

    # Target network is inference-only — stays in eval permanently.
    target.encoder.eval()
    target.v_net.eval()
    # Agent defaults to eval; switched to train() only for gradient steps.
    agent.eval()

    best_ckpt = os.path.join(ckpt_dir, f"kgp_{cfg.run_name}_best.pt")

    print("=" * 64)
    print(f"  {env_id} — Continuous Koopman training")
    print(f"  device={device}  hard_ortho={agent._use_hard_ortho}")
    print(f"  action_dim={action_dim}  d={d}  steps={n_steps:,}  envs={n_envs}")
    print(f"  planner={'sequential' if sequential else 'toeplitz'}"
          f"  cumulative={cumulative}  frozen_b={frozen_b}  ou_noise={ou_noise}")
    print(f"  run_name={cfg.run_name}")
    print("=" * 64)
    print(f"\n[Warmup: {warmup} random steps...]\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    states, _ = env.reset()
    ep_returns     = np.zeros(n_envs, dtype=np.float32)
    episode_returns = []
    koop_log, v_log = [], []
    recent_koop, recent_v = [], []
    ou_list  = [OUNoise(action_dim) for _ in range(n_envs)] if ou_noise else None
    best_ret = -float("inf")
    t0       = time.time()

    for env_step in range(n_envs, n_steps + 1, n_envs):
        noise = max(noise_end,
                    noise_start - (noise_start - noise_end)
                    * max(0, env_step - warmup) / noise_decay)

        if env_step <= warmup:
            actions = np.random.uniform(-action_scale, action_scale,
                                        (n_envs, action_dim)).astype(np.float32)
        else:
            if sequential:
                actions = agent.act_plan_continuous_batch(
                    states, horizon=plan_horizon, plan_iters=plan_iters,
                    action_scale=action_scale, frozen_b=frozen_b)
            else:
                actions = agent.act_plan_toeplitz_continuous_batch(
                    states, horizon=plan_horizon, plan_iters=plan_iters,
                    gamma=gamma, action_scale=action_scale,
                    cumulative=cumulative)
            if ou_noise:
                exploration = np.stack([ou_list[i].sample(sigma=noise * action_scale)
                                        for i in range(n_envs)])
            else:
                exploration = np.random.normal(0, noise * action_scale, size=actions.shape)
            actions = np.clip(actions + exploration, -action_scale, action_scale).astype(np.float32)

        next_states, rewards, terminated, truncated, infos = env.step(actions)
        dones = terminated | truncated

        # For done envs, store the final obs before auto-reset
        ns_buf = next_states.copy()
        if "final_observation" in infos and isinstance(infos["final_observation"], np.ndarray):
            ns_buf[dones] = infos["final_observation"][dones]

        buf.push_batch(states, actions, rewards, ns_buf, dones, terminated)

        ep_returns += rewards
        for i in np.where(dones)[0]:
            episode_returns.append(float(ep_returns[i]))
            ep_returns[i] = 0.0
            if ou_list is not None:
                ou_list[i].reset()
        states = next_states

        if buf.size < batch_size:
            continue

        # UTD ratio = 1: one gradient step per env step
        agent.train()
        for _ in range(n_envs):
            batch = buf.sample(batch_size)
            s_b, ns_b, a_b, r_b, d_b, t_b = (
                torch.as_tensor(batch[k], device=device)
                for k in ("states", "next_s", "actions", "rewards", "dones", "terminals")
            )

            z_src = agent.encode(s_b)
            with torch.no_grad():
                z_dst_tgt = target.encoder(ns_b)

            # Koopman loss — mask episode boundaries (terminated | truncated)
            z_pred = agent.dyn_step(z_src, a_b @ agent.B.T)
            L_koop = ((z_pred - z_dst_tgt.detach()).pow(2)
                      .sum(dim=-1) * (1.0 - d_b)).mean()

            # Reconstruction anchor
            L_recon = (agent.decoder(z_src) - s_b).pow(2).mean()

            # 1-step TD — mask only true terminals, not time-limit truncations
            with torch.no_grad():
                V_next = target.v_net(z_dst_tgt)
                y_td   = r_b / reward_scale + gamma * V_next * (1.0 - t_b)
            L_v = (agent.v_net(z_src) - y_td).pow(2).mean()

            # Soft ortho penalty (MPS/CPU fallback; never applied on CUDA)
            L_ortho = (agent.ortho_penalty()
                       if (agent._ortho_a and not agent._use_hard_ortho)
                       else torch.tensor(0.0, device=device))

            loss = lambda_koop * L_koop + lambda_recon * L_recon + lambda_v * L_v + L_ortho
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
            opt.step()
            agent.invalidate_toeplitz_cache()   # A may have changed
            target.update(agent, tau=ema_tau)
        agent.eval()

        recent_koop.append(L_koop.item())
        recent_v.append(L_v.item())

        if env_step % 1_000 == 0:
            elapsed = time.time() - t0; t0 = time.time()
            mk = np.mean(recent_koop); mv = np.mean(recent_v)
            koop_log.append(mk); v_log.append(mv)
            recent20 = episode_returns[-20:] if episode_returns else []
            ret20    = np.mean(recent20) if recent20 else float("nan")
            ortho_err = agent.ortho_error()
            print(f"  step {env_step:5d}  noise={noise:.3f}  L_koop={mk:.4f}  L_v={mv:.4f}"
                  f"  ret/20={ret20:7.1f}"
                  f"  ‖AᵀA-I‖²={ortho_err:.1e}  sps={1000/elapsed:.0f}", flush=True)
            recent_koop.clear(); recent_v.clear()

            if ret20 > best_ret:
                best_ret = ret20
                _history = _make_history(cfg, episode_returns, koop_log, v_log, best_ret)
                save_checkpoint_async(agent, target, _history, path=best_ckpt)
                print(f"  [best] ret/20={best_ret:.1f} → {best_ckpt}", flush=True)

        if env_step % viz_every == 0 and on_viz is not None:
            on_viz(episode_returns=episode_returns, koop_log=koop_log, v_log=v_log,
                   step=env_step, ortho_err=agent.ortho_error(), agent=agent, buf=buf)

    env.close()

    # ── Planner benchmark ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  Planner benchmark — {_N_EVAL_PLAN} episodes each")
    print(f"  H={plan_horizon}  iters={plan_iters}  γ={gamma}")
    print("=" * 64)

    eval_env = gym.make_vec(env_id, num_envs=_N_EVAL_PLAN)
    planners = [
        ("sequential (baseline)",
         lambda ss: agent.act_plan_continuous_batch(
             ss, plan_horizon, plan_iters, action_scale=action_scale)),
        ("toeplitz  (GEMM)",
         lambda ss: agent.act_plan_toeplitz_continuous_batch(
             ss, plan_horizon, plan_iters,
             gamma=gamma, action_scale=action_scale, cumulative=False)),
    ]
    for name, fn in planners:
        t_plan = time.time()
        ss, _ = eval_env.reset()
        ep_rets = np.zeros(_N_EVAL_PLAN, dtype=np.float32)
        for _ in range(200):
            actions = fn(ss).astype(np.float32)
            ss, rewards, _, _, _ = eval_env.step(actions)
            ep_rets += rewards
        elapsed = time.time() - t_plan
        print(f"  {name:30s}  mean={ep_rets.mean():8.1f}  std={ep_rets.std():6.1f}"
              f"  wall={elapsed:.1f}s  ({elapsed/_N_EVAL_PLAN:.2f}s/ep)")
    eval_env.close()

    # ── Final checkpoint + viz ────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, f"kgp_{cfg.run_name}.pt")
    mean_ret   = np.mean(episode_returns[-20:]) if episode_returns else float("nan")
    history    = _make_history(cfg, episode_returns, koop_log, v_log, mean_ret)
    save_checkpoint(agent, target, history, path=final_path)

    if on_viz is not None:
        on_viz(episode_returns=episode_returns, koop_log=koop_log, v_log=v_log,
               step=n_steps, ortho_err=agent.ortho_error(), agent=agent, buf=buf)

    return dict(agent=agent, target=target, episode_returns=episode_returns,
                koop_log=koop_log, v_log=v_log, buf=buf)


def _make_history(cfg: Config, episode_returns, koop_log, v_log, mean_eval_return) -> dict:
    """Build history dict for checkpoint (caller merges into saved file)."""
    return dict(
        episode_returns=list(episode_returns),
        koop_log=list(koop_log),
        v_log=list(v_log),
        config=dict(
            state_dim=cfg.env.state_dim,
            action_dim=cfg.env.n_actions,
            d=cfg.model.d,
            n_steps=cfg.train.n_steps,
            gamma=cfg.algo.gamma,
            reward_scale=cfg.algo.reward_scale,
            mean_eval_return=mean_eval_return,
        ),
    )
