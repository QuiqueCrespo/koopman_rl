"""
MiniGrid Benchmark: Neural Q-Stalk Koopman-RL vs DQN
====================================================
Tests both agents on classic MiniGrid environments.

Architecture (shared):
  Observation  (7, 7, 3) uint8 partial view  →  /10.0  →  CNN encoder
  CNN          Conv(3→16,k=2) → Conv(16→32,k=2) → Conv(32→64,k=2) → Flatten → Linear(d)
  QNetwork     d → 128 → ReLU → n_actions (7)
  Koopman      K_a ∈ R^{d×d}, one per action  [Sheaf only]

Koopman-RL     T-step backward recursion + Koopman consistency loss
DQN          Single-step Bellman target, same network (no K_a)

Environments (tested in order of complexity):
  MiniGrid-Empty-8x8-v0        pure navigation  (sanity check)
  MiniGrid-FourRooms-v0        long-horizon navigation through 4 rooms
  MiniGrid-DoorKey-5x5-v0      must pick up key, open door, reach goal
  MiniGrid-MultiRoom-N2-S4-v0  navigate through two connected rooms

Performance notes:
  - All tensors live on DEVICE (MPS on Apple Silicon, else CPU)
  - src+dst observations are encoded in a single forward pass (halves CNN work)
  - train_freq: one gradient update every N environment steps (default 4)
"""

import sys
import time
import json
import pickle
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)
print(f"[device] {DEVICE}")

# ---------------------------------------------------------------------------
# Per-environment configs
# ---------------------------------------------------------------------------

N_ACTIONS = 7      # MiniGrid always exposes 7 actions
OBS_NORM  = 10.0   # max object-type index across all channels

ENV_CONFIGS = {
    "Empty-8x8": {
        "env_id":     "MiniGrid-Empty-8x8-v0",
        "max_steps":  128,
        "n_steps":    100_000,
        "d":          64,
        "T":          16,
        "B":          16,
        "lr":         3e-4,
        "eps_decay":  30_000,
        "warmup":     2_000,
        "train_freq": 4,
    },
    "FourRooms": {
        "env_id":     "MiniGrid-FourRooms-v0",
        "max_steps":  500,
        "n_steps":    500_000,
        "d":          128,
        "T":          20,
        "B":          16,
        "lr":         1e-4,
        "eps_decay":  150_000,
        "warmup":     5_000,
        "train_freq": 4,
    },
    "DoorKey-5x5": {
        "env_id":     "MiniGrid-DoorKey-5x5-v0",
        "max_steps":  300,
        "n_steps":    500_000,
        "d":          128,
        "T":          20,
        "B":          16,
        "lr":         1e-4,
        "eps_decay":  150_000,
        "warmup":     5_000,
        "train_freq": 4,
    },
    "MultiRoom-N2": {
        "env_id":     "MiniGrid-MultiRoom-N2-S4-v0",
        "max_steps":  128,
        "n_steps":    500_000,
        "d":          128,
        "T":          20,
        "B":          16,
        "lr":         1e-4,
        "eps_decay":  150_000,
        "warmup":     5_000,
        "train_freq": 4,
    },
}

# Shared training constants
GAMMA       = 0.99
EMA_TAU     = 0.005
EPS_START   = 1.0
EPS_END     = 0.05
LAMBDA_KOOP  = 0.5
LAMBDA_BISIM = 1.0   # bisimulation metric loss (anti-collapse + latent topology)
TAU_START    = 1.0   # AWR temperature — high = include all edges (warm-up)
TAU_END      = 0.1   # AWR temperature — low = prune suboptimal edges aggressively
LOG_EVERY    = 5_000
BUFFER_CAP     = 200_000


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------

def preprocess(obs: np.ndarray) -> torch.Tensor:
    """(7, 7, 3) uint8  →  (3, 7, 7) float32 on DEVICE."""
    t = torch.from_numpy(obs.transpose(2, 0, 1).astype(np.float32)) / OBS_NORM
    return t.to(DEVICE)

def preprocess_batch_np(obs: np.ndarray) -> torch.Tensor:
    """(B, 7, 7, 3) uint8  →  (B, 3, 7, 7) float32 on DEVICE."""
    t = torch.from_numpy(obs.transpose(0, 3, 1, 2).astype(np.float32)) / OBS_NORM
    return t.to(DEVICE)


# ---------------------------------------------------------------------------
# Neural components
# ---------------------------------------------------------------------------

class CNNEncoder(nn.Module):
    """
    f_θ: (3, 7, 7) → R^d
    Three 2×2 conv layers shrink 7→6→5→4; 64×4×4=1024 → Linear(d).
    """
    def __init__(self, d: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16,  kernel_size=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 4 * 4, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.fc(self.conv(x).flatten(1))
        return F.normalize(z, p=2, dim=-1)   # project to unit hypersphere


class QNetwork(nn.Module):
    """Q_ψ: z ∈ R^d → q ∈ R^{n_actions}."""
    def __init__(self, d: int, n_actions: int = N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def value(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z).max(dim=-1).values


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class KoopmanAgent(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d       = d
        self.encoder = CNNEncoder(d)
        self.q_net   = QNetwork(d)
        # Orthogonal init: K_a maps unit vectors to unit vectors at step 0,
        # so the initial Koopman loss is purely the small perturbation from
        # the encoder, not a 0.2x norm collapse.
        def _ortho(d):
            m = torch.empty(d, d)
            nn.init.orthogonal_(m)
            return m
        self.K = nn.ParameterList([
            nn.Parameter(_ortho(d)) for _ in range(N_ACTIONS)
        ])

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, N_ACTIONS - 1)
        x = preprocess(obs).unsqueeze(0)
        return self.q_net(self.encoder(x)).argmax(dim=-1).item()


class DQNAgent(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.encoder = CNNEncoder(d)
        self.q_net   = QNetwork(d)

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, N_ACTIONS - 1)
        x = preprocess(obs).unsqueeze(0)
        return self.q_net(self.encoder(x)).argmax(dim=-1).item()


# ---------------------------------------------------------------------------
# Target network (EMA, plain Python class — excluded from optimizer)
# ---------------------------------------------------------------------------

class TargetNet:
    def __init__(self, agent):
        self.encoder = copy.deepcopy(agent.encoder).eval().to(DEVICE)
        self.q_net   = copy.deepcopy(agent.q_net).eval().to(DEVICE)
        for p in self.encoder.parameters(): p.requires_grad_(False)
        for p in self.q_net.parameters():   p.requires_grad_(False)

    @torch.no_grad()
    def update(self, agent, tau: float = EMA_TAU):
        for pt, po in zip(self.encoder.parameters(), agent.encoder.parameters()):
            pt.data.mul_(1 - tau).add_(po.data, alpha=tau)
        for pt, po in zip(self.q_net.parameters(), agent.q_net.parameters()):
            pt.data.mul_(1 - tau).add_(po.data, alpha=tau)

    @torch.no_grad()
    def v_target(self, obs_t: torch.Tensor) -> torch.Tensor:
        """obs_t: (B, 3, 7, 7) on DEVICE → V_target [B]."""
        return self.q_net(self.encoder(obs_t)).max(dim=-1).values


# ---------------------------------------------------------------------------
# Replay buffer (chunk-safe, terminal-safe)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Circular buffer storing uint8 observations.
    Separate obs/next_obs arrays (Bug A fix).
    Chunk sampling rejects windows straddling the write pointer (Bug B fix).
    """
    def __init__(self, capacity: int = BUFFER_CAP):
        self.capacity = capacity
        self.obs      = np.zeros((capacity, 7, 7, 3), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, 7, 7, 3), dtype=np.uint8)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.ptr      = 0
        self.size     = 0

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_chunks(self, n_chunks: int, chunk_len: int) -> dict:
        valid = []
        while len(valid) < n_chunks:
            start = np.random.randint(0, self.size - chunk_len)
            if not (start < self.ptr <= start + chunk_len):
                valid.append(start)

        idx     = np.array(valid)[:, None] + np.arange(chunk_len)   # [B, T]
        obs_c   = self.obs[idx]                                       # [B, T, 7, 7, 3]
        obs_f   = self.next_obs[idx[:, -1:]]                          # [B, 1, 7, 7, 3]
        obs_all = np.concatenate([obs_c, obs_f], axis=1)              # [B, T+1, 7, 7, 3]

        return {
            "obs":     obs_all,                                             # numpy uint8
            "actions": torch.from_numpy(self.actions[idx]).to(DEVICE),
            "rewards": torch.from_numpy(self.rewards[idx]).to(DEVICE),
            "dones":   torch.from_numpy(self.dones[idx]).to(DEVICE),
        }

    def sample_flat(self, n: int) -> dict:
        idx = np.random.randint(0, self.size, size=n)
        return {
            "obs":      preprocess_batch_np(self.obs[idx]),
            "next_obs": preprocess_batch_np(self.next_obs[idx]),
            "actions":  torch.from_numpy(self.actions[idx]).to(DEVICE),
            "rewards":  torch.from_numpy(self.rewards[idx]).to(DEVICE),
            "dones":    torch.from_numpy(self.dones[idx]).to(DEVICE),
        }

    def ready(self, min_size: int) -> bool:
        return self.size >= min_size


# ---------------------------------------------------------------------------
# Sheaf backward recursion (block-diagonal M solve)
# ---------------------------------------------------------------------------

def compute_q_targets(
    rewards:     torch.Tensor,   # [B, T]
    dones:       torch.Tensor,   # [B, T]
    v_bootstrap: torch.Tensor,   # [B]
    gamma:       float = GAMMA,
) -> torch.Tensor:
    B, T     = rewards.shape
    Q        = torch.zeros(B, T, device=rewards.device)
    next_val = v_bootstrap.clone()
    for t in reversed(range(T)):
        Q[:, t]  = rewards[:, t] + gamma * next_val * (1.0 - dones[:, t])
        next_val = Q[:, t]
    return Q


# ---------------------------------------------------------------------------
# Deep Bisimulation Metric loss (Zhang et al. 2020)
# ---------------------------------------------------------------------------

def compute_bisimulation_loss(
    z_src:        torch.Tensor,   # [N, d]  online encoder latents (gradients flow)
    z_dst_target: torch.Tensor,   # [N, d]  target encoder next-state latents (frozen)
    rewards:      torch.Tensor,   # [N]     immediate rewards
    gamma:        float = GAMMA,
) -> torch.Tensor:
    """
    Forces the L2 distance in latent space to equal the bootstrapped
    bisimulation distance:

        d(z_i, z_j)  =  |r_i - r_j|  +  γ · d(z'_i_tgt, z'_j_tgt)

    This organises the hypersphere by *causal future*: states with the same
    behavioural consequences cluster together, states with different futures
    are pushed apart.  It is strictly more informative than plain uniformity
    loss, which pushes all pairs apart regardless of reward structure, and
    can conflict with bisimulation's pull on similar pairs.

    Note on the hypersphere bound: z_dist ∈ [0, 2] but target_dist can reach
    ~(1 + γ·2) ≈ 3.  For dissimilar pairs the MSE target exceeds the geometric
    maximum, creating a permanent non-zero floor.  Gradients still push those
    pairs toward antipodal positions, which is the correct behaviour.
    """
    N    = z_src.size(0)
    perm = torch.randperm(N, device=z_src.device)

    z_i, z_j   = z_src,        z_src[perm]
    zt_i, zt_j = z_dst_target, z_dst_target[perm]
    r_i, r_j   = rewards,      rewards[perm]

    # Current latent geometry — gradients flow back through online encoder
    z_dist = F.pairwise_distance(z_i, z_j, p=2, eps=1e-6)   # [N]

    # Target bisimulation distance (Bellman backup, one step)
    r_dist      = (r_i - r_j).abs()
    next_z_dist = F.pairwise_distance(zt_i, zt_j, p=2, eps=1e-6)
    target_dist = (r_dist + gamma * next_z_dist).detach()   # [N]  no grad

    return F.mse_loss(z_dist, target_dist)


# ---------------------------------------------------------------------------
# Koopman-RL training
# ---------------------------------------------------------------------------

def train_koopman(cfg: dict, label: str, results_dir: pathlib.Path = None) -> dict:
    T          = cfg["T"]
    B          = cfg["B"]
    d          = cfg["d"]
    n_steps    = cfg["n_steps"]
    warmup     = cfg["warmup"]
    eps_decay  = cfg["eps_decay"]
    lr         = cfg["lr"]
    max_steps  = cfg["max_steps"]
    train_freq = cfg["train_freq"]

    env    = ImgObsWrapper(gym.make(cfg["env_id"], max_steps=max_steps))
    buf    = ReplayBuffer()
    agent  = KoopmanAgent(d).to(DEVICE)
    target = TargetNet(agent)
    # K_a gets weight_decay as a soft spectral penalty; encoder+Q-net do not.
    opt = optim.Adam([
        {"params": list(agent.encoder.parameters()) + list(agent.q_net.parameters()),
         "lr": lr,         "weight_decay": 0.0},
        {"params": list(agent.K.parameters()),
         "lr": lr * 0.5,   "weight_decay": 1e-3},
    ])

    min_buf = B * T + 2
    obs, _    = env.reset()
    ep_return = 0.0
    ep_steps  = 0
    episode_returns   = []
    episode_end_steps = []   # env step at which each episode terminated
    koop_losses, q_losses, bisim_losses = [], [], []   # LOG_EVERY averages
    all_koop_losses   = []   # per-gradient-update (fine-grained)
    all_q_losses      = []
    all_bisim_losses  = []
    all_update_steps  = []   # env step at each gradient update
    recent_koop, recent_q, recent_bisim = [], [], []
    t0 = time.time()

    print(f"\n  [Koopman-RL] {label}  d={d}  T={T}  B={B}  "
          f"train_freq={train_freq}  n_steps={n_steps:,}  device={DEVICE}")

    for step in range(1, n_steps + 1):
        eps    = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / eps_decay)
        action = agent.act(obs, epsilon=eps if step > warmup else 1.0)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buf.push(obs, action, reward, next_obs, terminated)
        ep_return += reward
        ep_steps  += 1

        if done:
            episode_returns.append(ep_return)
            episode_end_steps.append(step)
            ep_return, ep_steps = 0.0, 0
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step <= warmup or not buf.ready(min_buf) or step % train_freq != 0:
            continue

        # ---- forward pass ----
        batch   = buf.sample_chunks(B, T)
        obs_all = batch["obs"]        # [B, T+1, 7, 7, 3] numpy uint8
        a_bt    = batch["actions"]    # [B, T]  on DEVICE
        r_bt    = batch["rewards"]    # [B, T]
        d_bt    = batch["dones"]      # [B, T]

        # Encode all T+1 observations in one CNN forward pass (halves encoder work)
        obs_t  = preprocess_batch_np(
            obs_all.reshape(-1, 7, 7, 3)
        ).reshape(B, T + 1, 3, 7, 7)              # [B, T+1, 3, 7, 7]  on DEVICE

        with torch.no_grad():
            v_boot = target.v_target(obs_t[:, -1].contiguous())     # [B]
            # Target-network embeddings of next states for bisimulation RHS
            z_dst_target = target.encoder(
                obs_t[:, 1:].contiguous().reshape(N, 3, 7, 7)
            )                                                        # [N, d]

        with torch.no_grad():
            Q_diff = compute_q_targets(r_bt, d_bt, v_boot)          # [B, T]

        N       = B * T
        a_f     = a_bt.reshape(N)
        d_f     = d_bt.reshape(N)
        r_f     = r_bt.reshape(N)    # [N]  immediate rewards for bisimulation
        Q_tgt_f = Q_diff.reshape(N)

        # Single combined encoder call for all T+1 obs → split into src/dst
        z_all  = agent.encoder(obs_t.contiguous().reshape(B * (T + 1), 3, 7, 7))
        z_all  = z_all.reshape(B, T + 1, d)
        z_src  = z_all[:, :-1].contiguous().reshape(N, d)   # [N, d]
        z_dst  = z_all[:,  1:].contiguous().reshape(N, d)   # [N, d]

        # Koopman loss: ||K_a z_src - z_dst||²  (masked at episode boundaries)
        z_pred = torch.zeros_like(z_src)
        for a in range(N_ACTIONS):
            mask = (a_f == a)
            if mask.any():
                z_pred[mask] = z_src[mask] @ agent.K[a].T
        koop_mask = 1.0 - d_f
        L_koop    = ((z_pred - z_dst.detach()).pow(2)
                      .sum(-1).mul(koop_mask)).mean()

        # Q loss with advantage-weighted edge pruning (AWR / retrospective filtering)
        # W_t = exp(A_t / τ) where A_t = Q(z_t, a_t) - max_a' Q(z_t, a') ≤ 0
        # τ anneals TAU_START → TAU_END over training: early = soft (all edges),
        # late = hard (only near-greedy transitions carry the diffused reward).
        # Using Q-network advantage directly (not Koopman model) so the weights
        # are correct even before K_a has converged.
        tau     = max(TAU_END, TAU_START - (TAU_START - TAU_END) * step / n_steps)
        Q_pred  = agent.q_net(z_src)                                  # [N, A]
        Q_taken = Q_pred.gather(1, a_f.unsqueeze(1)).squeeze(1)       # [N]
        with torch.no_grad():
            A_t = Q_taken - Q_pred.max(dim=-1).values                 # [N] ≤ 0
            W   = torch.exp(A_t / tau)                                 # [N] ∈ (0,1]
        L_q = (W * (Q_taken - Q_tgt_f).pow(2)).mean()

        # Bisimulation metric loss: d(z_i, z_j) = |r_i - r_j| + γ·d(z'_i, z'_j)
        # Uses target encoder for the RHS to avoid chasing moving targets.
        # Supersedes the uniformity loss: it both pushes different-future states
        # apart AND pulls same-future states together.
        L_bisim = compute_bisimulation_loss(z_src, z_dst_target, r_f)

        loss = LAMBDA_KOOP * L_koop + L_q + LAMBDA_BISIM * L_bisim
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
        opt.step()
        target.update(agent)

        lk, lq, lb = L_koop.item(), L_q.item(), L_bisim.item()
        recent_koop.append(lk);   all_koop_losses.append(lk)
        recent_q.append(lq);      all_q_losses.append(lq)
        recent_bisim.append(lb);  all_bisim_losses.append(lb)
        all_update_steps.append(step)

        if step % LOG_EVERY == 0:
            ret20   = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            success = sum(r > 0 for r in episode_returns[-20:])
            mk, mq, mb = np.mean(recent_koop), np.mean(recent_q), np.mean(recent_bisim)
            elapsed = time.time() - t0
            sps     = step / elapsed
            print(f"    step {step:7d}  ε={eps:.3f}  "
                  f"L_koop={mk:.4f}  L_q={mq:.4f}  L_bisim={mb:.4f}  "
                  f"ret={ret20:.3f}  succ/20={success}  "
                  f"[{sps:.0f} sps]")
            koop_losses.append(mk)
            q_losses.append(mq)
            bisim_losses.append(mb)
            recent_koop.clear(); recent_q.clear(); recent_bisim.clear()

    env.close()

    result = {
        "agent":             agent,
        "cfg":               cfg,
        "episode_returns":   episode_returns,
        "episode_end_steps": episode_end_steps,
        "koop_losses":       koop_losses,       # LOG_EVERY averages
        "q_losses":          q_losses,
        "bisim_losses":      bisim_losses,
        "all_koop_losses":   all_koop_losses,   # per-update
        "all_q_losses":      all_q_losses,
        "all_bisim_losses":  all_bisim_losses,
        "all_update_steps":  all_update_steps,
    }

    if results_dir is not None:
        model_path = results_dir / f"{label}_sheaf_model.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"    [saved model] {model_path}")

        data_path = results_dir / f"{label}_sheaf_results.pt"
        torch.save({k: v for k, v in result.items() if k != "agent"}, data_path)
        print(f"    [saved data]  {data_path}")

    return result


# ---------------------------------------------------------------------------
# DQN training
# ---------------------------------------------------------------------------

def train_dqn(cfg: dict, label: str, results_dir: pathlib.Path = None) -> dict:
    d          = cfg["d"]
    n_steps    = cfg["n_steps"]
    warmup     = cfg["warmup"]
    eps_decay  = cfg["eps_decay"]
    lr         = cfg["lr"]
    max_steps  = cfg["max_steps"]
    train_freq = cfg["train_freq"]
    batch_sz   = cfg["B"] * cfg["T"]

    env    = ImgObsWrapper(gym.make(cfg["env_id"], max_steps=max_steps))
    buf    = ReplayBuffer()
    agent  = DQNAgent(d).to(DEVICE)
    opt    = optim.Adam(agent.parameters(), lr=lr)

    tgt_enc = copy.deepcopy(agent.encoder).eval().to(DEVICE)
    tgt_q   = copy.deepcopy(agent.q_net).eval().to(DEVICE)
    for p in tgt_enc.parameters(): p.requires_grad_(False)
    for p in tgt_q.parameters():   p.requires_grad_(False)

    obs, _    = env.reset()
    ep_return = 0.0
    ep_steps  = 0
    episode_returns   = []
    episode_end_steps = []   # env step at which each episode terminated
    q_losses, recent_q = [], []   # LOG_EVERY averages
    all_q_losses      = []   # per-gradient-update (fine-grained)
    all_update_steps  = []
    t0 = time.time()

    print(f"\n  [DQN]      {label}  d={d}  batch={batch_sz}  "
          f"train_freq={train_freq}  n_steps={n_steps:,}  device={DEVICE}")

    for step in range(1, n_steps + 1):
        eps    = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / eps_decay)
        action = agent.act(obs, epsilon=eps if step > warmup else 1.0)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buf.push(obs, action, reward, next_obs, terminated)
        ep_return += reward
        ep_steps  += 1

        if done:
            episode_returns.append(ep_return)
            episode_end_steps.append(step)
            ep_return, ep_steps = 0.0, 0
            obs, _ = env.reset()
        else:
            obs = next_obs

        if step <= warmup or not buf.ready(batch_sz) or step % train_freq != 0:
            continue

        batch   = buf.sample_flat(batch_sz)
        s_b     = batch["obs"]
        ns_b    = batch["next_obs"]
        a_b     = batch["actions"]
        r_b     = batch["rewards"]
        d_b     = batch["dones"]

        with torch.no_grad():
            v_next = tgt_q(tgt_enc(ns_b)).max(dim=-1).values
            Q_tgt  = r_b + GAMMA * v_next * (1.0 - d_b)

        Q_pred  = agent.q_net(agent.encoder(s_b))
        Q_taken = Q_pred.gather(1, a_b.unsqueeze(1)).squeeze(1)
        loss    = (Q_taken - Q_tgt).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
        opt.step()

        for pt, po in zip(tgt_enc.parameters(), agent.encoder.parameters()):
            pt.data.mul_(1 - EMA_TAU).add_(po.data, alpha=EMA_TAU)
        for pt, po in zip(tgt_q.parameters(), agent.q_net.parameters()):
            pt.data.mul_(1 - EMA_TAU).add_(po.data, alpha=EMA_TAU)

        lq = loss.item()
        recent_q.append(lq)
        all_q_losses.append(lq)
        all_update_steps.append(step)

        if step % LOG_EVERY == 0:
            ret20   = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            success = sum(r > 0 for r in episode_returns[-20:])
            mq      = np.mean(recent_q)
            elapsed = time.time() - t0
            sps     = step / elapsed
            print(f"    step {step:7d}  ε={eps:.3f}  "
                  f"L_q={mq:.4f}  ret={ret20:.3f}  succ/20={success}  "
                  f"[{sps:.0f} sps]")
            q_losses.append(mq)
            recent_q.clear()

    env.close()

    result = {
        "agent":             agent,
        "cfg":               cfg,
        "episode_returns":   episode_returns,
        "episode_end_steps": episode_end_steps,
        "q_losses":          q_losses,         # LOG_EVERY averages
        "all_q_losses":      all_q_losses,     # per-update
        "all_update_steps":  all_update_steps,
    }

    if results_dir is not None:
        model_path = results_dir / f"{label}_dqn_model.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"    [saved model] {model_path}")

        data_path = results_dir / f"{label}_dqn_results.pt"
        torch.save({k: v for k, v in result.items() if k != "agent"}, data_path)
        print(f"    [saved data]  {data_path}")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _rolling(x, w=50):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")

def _success_curve(returns, window=50):
    hits = [float(r > 0) for r in returns]
    if len(hits) < window:
        return np.arange(len(hits)), np.array(hits)
    return np.arange(window - 1, len(hits)), np.convolve(hits, np.ones(window) / window, mode="valid")

def _first_reach(returns, threshold=0.8, window=50):
    hits = [float(r > 0) for r in returns]
    for i in range(window, len(hits)):
        if np.mean(hits[i - window:i]) >= threshold:
            return i
    return None


def plot_all(results: dict, env_names: list) -> None:
    n   = len(env_names)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    if n == 1:
        axes = axes[:, None]
    fig.suptitle("Koopman-RL vs DQN — MiniGrid Benchmark\n"
                 "Same CNN encoder, optimizer, ε-schedule, step budget",
                 fontsize=13)

    for col, name in enumerate(env_names):
        sh  = results[name]["sheaf"]
        dq  = results[name]["dqn"]
        cfg = ENV_CONFIGS[name]

        ax = axes[0, col]
        for hist, lbl, color in [
            (sh, f"Koopman-RL (T={cfg['T']})", "royalblue"),
            (dq, "DQN (1-step)",              "tomato"),
        ]:
            ax.plot(_rolling(hist["episode_returns"], 50),
                    color=color, linewidth=1.5, label=lbl, alpha=0.9)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (50-ep mean)" if col == 0 else "")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        for hist, lbl, color in [
            (sh, "Koopman-RL", "royalblue"),
            (dq, "DQN",      "tomato"),
        ]:
            xs, ys = _success_curve(hist["episode_returns"], 50)
            ax.plot(xs, ys * 100, color=color, linewidth=1.5, label=lbl)
        ax.axhline(100, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success % (50-ep)" if col == 0 else "")
        ax.set_ylim(-5, 110); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("minigrid_benchmark.png", dpi=130)
    print("\nSaved → minigrid_benchmark.png")


def print_summary(results: dict, env_names: list) -> None:
    hdr = f"  {'Env':<18} {'Agent':<10} {'#Episodes':>10} {'Last100 Succ%':>14} {'Ep@80%':>8}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name in env_names:
        for key, lbl in [("sheaf", "Koopman-RL"), ("dqn", "DQN")]:
            hist = results[name][key]
            ret  = hist["episode_returns"]
            sr   = np.mean([r > 0 for r in ret[-100:]]) * 100 if len(ret) >= 100 else 0.0
            fe   = _first_reach(ret)
            fe_s = str(fe) if fe is not None else "never"
            print(f"  {name:<18} {lbl:<10} {len(ret):>10} {sr:>13.1f}%  {fe_s:>8}")
    print("=" * len(hdr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        run_envs = [a for a in sys.argv[1:] if a in ENV_CONFIGS]
        if not run_envs:
            print(f"Available: {list(ENV_CONFIGS)}")
            return
    else:
        run_envs = list(ENV_CONFIGS)

    # Create a timestamped results directory
    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    results_dir = pathlib.Path("results") / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Results dir: {results_dir}")

    print("=" * 62)
    print("  MiniGrid Benchmark: Koopman-RL vs DQN")
    print(f"  Environments: {run_envs}")
    print(f"  Device: {DEVICE}")
    print("=" * 62)

    results = {}
    for name in run_envs:
        cfg = ENV_CONFIGS[name]
        print(f"\n{'='*62}")
        print(f"  Environment: {name}  ({cfg['env_id']})")
        print(f"{'='*62}")

        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        sheaf_hist = train_koopman(cfg, name, results_dir=results_dir)

        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        dqn_hist = train_dqn(cfg, name, results_dir=results_dir)

        results[name] = {"sheaf": sheaf_hist, "dqn": dqn_hist}

    print_summary(results, run_envs)
    plot_all(results, run_envs)

    # Save run summary JSON (no tensors — pure Python types)
    summary = {}
    for name in run_envs:
        sh  = results[name]["sheaf"]
        dq  = results[name]["dqn"]
        ret_s = sh["episode_returns"]
        ret_d = dq["episode_returns"]
        summary[name] = {
            "sheaf": {
                "n_episodes":    len(ret_s),
                "last100_succ":  float(np.mean([r > 0 for r in ret_s[-100:]])) if len(ret_s) >= 100 else None,
                "ep_at_80pct":   _first_reach(ret_s),
                "final_ret_mean": float(np.mean(ret_s[-50:])) if ret_s else None,
            },
            "dqn": {
                "n_episodes":    len(ret_d),
                "last100_succ":  float(np.mean([r > 0 for r in ret_d[-100:]])) if len(ret_d) >= 100 else None,
                "ep_at_80pct":   _first_reach(ret_d),
                "final_ret_mean": float(np.mean(ret_d[-50:])) if ret_d else None,
            },
        }
    summary_path = results_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"timestamp": timestamp, "envs": run_envs, "results": summary}, f, indent=2)
    print(f"\n  [saved summary] {summary_path}")


if __name__ == "__main__":
    main()
