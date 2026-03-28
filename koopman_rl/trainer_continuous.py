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
from koopman_rl.losses import SIGReg

# Number of evaluation episodes in the end-of-training planner benchmark
_N_EVAL_PLAN = 20


def train_continuous(
    cfg: Config,
    env_id: str,
    device: torch.device,
    on_viz=None,
    extra_eval_fns=None,
) -> dict:
    """
    Train KoopmanGradientPlanner on a continuous-action gymnasium environment.

    Parameters
    ----------
    cfg            : Config with all hyperparameters (see config.py).
    env_id         : gymnasium env id, e.g. "Pendulum-v1".
    device         : torch device to train on.
    on_viz         : optional callable invoked every cfg.train.viz_every steps.
                     Called with keyword args:
                       episode_returns, koop_log, q_log, step, ortho_err, agent, buf
    extra_eval_fns : optional list of (name, fn_factory) pairs added to the
                     end-of-training benchmark.  fn_factory(agent) must return a
                     batched action fn  ss → actions[N, action_dim].

    Returns
    -------
    dict with keys: agent, target, episode_returns, koop_log, q_log, buf
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
    lambda_koop   = cfg.algo.lambda_koop
    lambda_recon  = cfg.algo.lambda_recon
    lambda_sigreg = cfg.algo.lambda_sigreg
    koop_lr_scale = cfg.algo.koop_lr_scale
    reward_scale = cfg.algo.reward_scale
    n_envs       = cfg.algo.n_envs
    utd_ratio    = cfg.algo.utd_ratio
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
    frozen_b     = cfg.train.frozen_b
    ou_noise     = cfg.train.ou_noise

    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Model + buffer ────────────────────────────────────────────────────────
    env    = gym.make_vec(env_id, num_envs=n_envs)
    agent  = KoopmanGradientPlanner.from_cfg(cfg, device=device)
    target = TargetNetwork(agent)
    buf    = ContinuousReplayBuffer(capacity, state_dim, action_dim)

    # World model: encoder + decoder + r_net + q_net  (NOT pi_net)
    world_neural = (list(agent.encoder.parameters()) +
                    list(agent.decoder.parameters()) +
                    list(agent.r_net.parameters()) +
                    list(agent.q_net.parameters()))
    koop_params  = agent.koop_parameters()
    opt_world = optim.Adam([
        {"params": world_neural, "lr": lr},
        {"params": koop_params,  "lr": lr * koop_lr_scale},
    ])
    # Actor: only pi_net — separate backward avoids double-counting Q gradients
    opt_pi = optim.Adam(agent.pi_net.parameters(), lr=lr)

    agent.to(device)
    target.encoder.to(device)
    target.q_net.to(device)
    target.pi_net.to(device)

    # Target networks are inference-only — stay in eval permanently.
    target.encoder.eval()
    target.q_net.eval()
    target.pi_net.eval()
    # Agent defaults to eval; switched to train() only for gradient steps.
    agent.eval()

    # Pre-allocated noise buffer — refilled in-place each step to avoid
    # repeated cuRAND state updates + allocations inside the hot loop.
    _z_noise  = (torch.empty(batch_size, d, device=device)
                 if cfg.algo.noise_z_std > 0 else None)
    _sigreg   = SIGReg().to(device) if lambda_sigreg > 0 else None
    use_target_encoder = cfg.algo.use_target_encoder

    best_ckpt = os.path.join(ckpt_dir, f"kgp_{cfg.run_name}_best.pt")

    print("=" * 64)
    print(f"  {env_id} — Continuous Koopman training (actor-critic)")
    print(f"  device={device}  hard_ortho={agent._use_hard_ortho}")
    print(f"  action_dim={action_dim}  d={d}  steps={n_steps:,}  envs={n_envs}")
    print(f"  data_collection=policy  frozen_b={frozen_b}  ou_noise={ou_noise}")
    print(f"  run_name={cfg.run_name}")
    print("=" * 64)
    print(f"\n[Warmup: {warmup} random steps...]\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    states, _ = env.reset()
    ep_returns      = np.zeros(n_envs, dtype=np.float32)
    episode_returns = []
    koop_log, q_log = [], []
    _koop_acc  = torch.zeros(1, device=device)
    _q_acc     = torch.zeros(1, device=device)
    _r_acc     = torch.zeros(1, device=device)
    _pi_acc    = torch.zeros(1, device=device)
    _recon_acc = torch.zeros(1, device=device)
    _n_acc = 0
    ou_list  = [OUNoise(action_dim) for _ in range(n_envs)] if ou_noise else None
    best_koop = float("inf")
    t0        = time.time()

    for env_step in range(n_envs, n_steps + 1, n_envs):
        noise = max(noise_end,
                    noise_start - (noise_start - noise_end)
                    * max(0, env_step - warmup) / noise_decay)

        # Data collection: random during warmup, policy + noise afterwards.
        # MPC is eval-only — using the planner here costs ~200x per step.
        if env_step <= warmup:
            actions = np.random.uniform(-action_scale, action_scale,
                                        (n_envs, action_dim)).astype(np.float32)
        else:
            actions = agent.act_policy_continuous_batch(states, action_scale)
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

        agent.train()
        for _ in range(n_envs * utd_ratio):
            batch = buf.sample(batch_size)
            s_b, ns_b, a_b, r_b, d_b, t_b = (
                torch.as_tensor(batch[k], device=device)
                for k in ("states", "next_s", "actions", "rewards", "dones", "terminals")
            )

            z_src  = agent.encode(s_b)
            a_norm = a_b / action_scale           # normalise to [-1, 1]; consistent with planner

            if use_target_encoder:
                with torch.no_grad():
                    z_dst_tgt = target.encoder(ns_b)
            else:
                z_dst_tgt = agent.encode(ns_b)   # online encoder; SIGReg prevents collapse

            # ── Koopman + reconstruction (gradient flows to encoder/A/B/decoder) ─
            z_src_dyn = z_src
            if _z_noise is not None:
                z_src_dyn = z_src + _z_noise.normal_(0, cfg.algo.noise_z_std)
            z_pred  = agent.dyn_step(z_src_dyn, a_norm @ agent.B.T)   # B trained in normalised units
            L_koop  = ((z_pred - z_dst_tgt.detach()).pow(2)
                       .sum(dim=-1) * (1.0 - d_b)).mean()
            L_recon = (agent.decoder(z_src) - s_b).pow(2).mean()

            # SIGReg: push z distribution toward N(0, I)
            L_sigreg = (_sigreg(z_src.unsqueeze(0))
                        if _sigreg is not None
                        else torch.tensor(0.0, device=device))

            # Stop-grad on z_src: RL losses must not reshape the Koopman encoder.
            z_det     = z_src.detach()
            z_dst_det = z_dst_tgt                # target encoder already has no grad

            # ── Reward predictor (no bootstrap, no target needed) ─────────────
            za     = torch.cat([z_det, a_norm], dim=-1)
            L_r    = (agent.r_net(za).squeeze(-1) - r_b / reward_scale).pow(2).mean()

            # ── Critic / Q-network ────────────────────────────────────────────
            with torch.no_grad():
                a_next = target.pi_net(z_dst_det)
                q_tgt  = (r_b / reward_scale
                          + gamma * target.q_net(
                              torch.cat([z_dst_det, a_next], -1)).squeeze(-1)
                          * (1.0 - t_b))
            L_q = (agent.q_net(za).squeeze(-1) - q_tgt).pow(2).mean()

            # ── World update (encoder + dynamics + r_net + q_net) ────────────
            recon_term = 0.0 if _sigreg is not None else lambda_recon * L_recon
            loss_world = lambda_koop * L_koop + recon_term + L_r + L_q + lambda_sigreg * L_sigreg
            opt_world.zero_grad()
            loss_world.backward()
            opt_world.step()
            agent.invalidate_toeplitz_cache()
            target.update(agent, tau=ema_tau)

            # ── Actor update (fresh forward; q_net is a fixed scoring fn here) ─
            a_curr = agent.pi_net(z_det)
            L_pi   = -agent.q_net(torch.cat([z_det, a_curr], -1)).squeeze(-1).mean()
            opt_pi.zero_grad()
            L_pi.backward()
            opt_pi.step()

            # Accumulate losses as plain floats — .item() deferred to log time
            # to avoid per-step GPU→CPU sync.
            _koop_acc  += L_koop.detach()
            _q_acc     += L_q.detach()
            _r_acc     += L_r.detach()
            _pi_acc    += L_pi.detach()
            _recon_acc += L_recon.detach()
            _n_acc     += 1

        agent.eval()

        if env_step % 1_000 == 0:
            elapsed = time.time() - t0; t0 = time.time()
            if _n_acc > 0:
                mk     = (_koop_acc  / _n_acc).item()
                mq     = (_q_acc     / _n_acc).item()
                mr     = (_r_acc     / _n_acc).item()
                mpi    = (_pi_acc    / _n_acc).item()
                mrecon = (_recon_acc / _n_acc).item()
            else:
                mk = mq = mr = mpi = mrecon = float("nan")
            _koop_acc  = torch.zeros(1, device=device)
            _q_acc     = torch.zeros(1, device=device)
            _r_acc     = torch.zeros(1, device=device)
            _pi_acc    = torch.zeros(1, device=device)
            _recon_acc = torch.zeros(1, device=device)
            _n_acc = 0
            koop_log.append(mk); q_log.append(mq)
            recent20 = episode_returns[-20:] if episode_returns else []
            ret20    = np.mean(recent20) if recent20 else float("nan")
            ortho_err = agent.ortho_error()
            print(f"  step {env_step:5d}  noise={noise:.3f}"
                  f"  L_koop={mk:.4f}  L_recon={mrecon:.4f}  L_r={mr:.4f}  L_q={mq:.4f}  L_pi={mpi:.4f}"
                  f"  ret/20={ret20:7.1f}"
                  f"  ‖AᵀA-I‖²={ortho_err:.1e}  sps={1000/elapsed:.0f}", flush=True)

            if mk < best_koop and env_step > warmup:
                best_koop = mk
                _history = _make_history(cfg, episode_returns, koop_log, q_log, ret20)
                save_checkpoint_async(agent, target, _history, path=best_ckpt)
                print(f"  [best koop] L_koop={best_koop:.4f}  ret/20={ret20:.1f} → {best_ckpt}", flush=True)

        if env_step % viz_every == 0 and on_viz is not None:
            on_viz(episode_returns=episode_returns, koop_log=koop_log, q_log=q_log,
                   step=env_step, ortho_err=agent.ortho_error(), agent=agent, buf=buf)

    env.close()

    # ── Restore best checkpoint for benchmark ────────────────────────────────
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["agent_state_dict"])
        target.encoder.load_state_dict(ckpt["target_encoder"])
        if "target_q_net" in ckpt:
            target.q_net.load_state_dict(ckpt["target_q_net"])
            target.pi_net.load_state_dict(ckpt["target_pi_net"])
        agent.eval()
        print(f"\n  [benchmark] restored best checkpoint (L_koop={best_koop:.4f}): {best_ckpt}")

    # ── Planner benchmark ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  Planner benchmark — {_N_EVAL_PLAN} episodes each")
    print(f"  H={plan_horizon}  iters={plan_iters}  γ={gamma}")
    print("=" * 64)

    eval_env = gym.make_vec(env_id, num_envs=_N_EVAL_PLAN)
    planners = [
        ("direct policy         ",
         lambda ss: agent.act_policy_continuous_batch(ss, action_scale)),
        ("toeplitz MPC (r_net+Q)",
         lambda ss: agent.act_plan_continuous(
             ss, plan_horizon, plan_iters, gamma=gamma, action_scale=action_scale)),
    ]
    if extra_eval_fns:
        for name, fn_factory in extra_eval_fns:
            planners.append((name, fn_factory(agent)))
    for name, fn in planners:
        t_plan = time.time()
        if hasattr(fn, "reset"):
            fn.reset()   # discard any stale warm-start state before the episode
        ss, _ = eval_env.reset()
        ep_rets = np.zeros(_N_EVAL_PLAN, dtype=np.float32)
        for _ in range(200):
            actions = fn(ss).astype(np.float32)
            ss, rewards, _, _, _ = eval_env.step(actions)
            ep_rets += rewards
        elapsed = time.time() - t_plan
        print(f"  {name}  mean={ep_rets.mean():8.1f}  std={ep_rets.std():6.1f}"
              f"  wall={elapsed:.1f}s  ({elapsed/_N_EVAL_PLAN:.2f}s/ep)")
    eval_env.close()

    # ── Final checkpoint + viz ────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, f"kgp_{cfg.run_name}.pt")
    mean_ret   = np.mean(episode_returns[-20:]) if episode_returns else float("nan")
    history    = _make_history(cfg, episode_returns, koop_log, q_log, mean_ret)
    save_checkpoint(agent, target, history, path=final_path)

    if on_viz is not None:
        on_viz(episode_returns=episode_returns, koop_log=koop_log, q_log=q_log,
               step=n_steps, ortho_err=agent.ortho_error(), agent=agent, buf=buf)

    return dict(agent=agent, target=target, episode_returns=episode_returns,
                koop_log=koop_log, q_log=q_log, buf=buf)


def _make_history(cfg: Config, episode_returns, koop_log, q_log, mean_eval_return) -> dict:
    """Build history dict for checkpoint (caller merges into saved file)."""
    return dict(
        episode_returns=list(episode_returns),
        koop_log=list(koop_log),
        q_log=list(q_log),
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
