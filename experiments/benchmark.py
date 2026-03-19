"""
DQN vs Neural Q-Stalk Koopman-RL: 2D Gravity Basin Benchmark
===========================================================
Both agents share everything except the TD target and the Koopman loss.

Identical across both:
  Environment   GravityBasin (same seeds)
  Network       Encoder [2→64→tanh→64→tanh→32] + QNet [32→64→ReLU→4]
  Optimizer     Adam, lr=3e-4
  Target net    EMA, τ=0.005
  ε-schedule    1.0 → 0.05 over 40k steps, warmup=3000
  Budget        100k environment steps
  Batch size    256 transitions per gradient step (= B×T = 16×16 from Koopman-RL)

Differs:
  DQN         target = r + γ·max_{a'} Q_tgt(s',a')           (single-step Bellman)
  Koopman-RL  target = T=16 backward recursion + Koopman loss (multi-step diffusion)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse environment and network components from the Koopman-RL implementation
from gravity_basin import (
    GravityBasin, Encoder, QNetwork, train as train_koopman, evaluate,
    N_ACTIONS, STATE_DIM, D, GAMMA, EMA_TAU, LR,
    WARMUP, N_STEPS, EPS_START, EPS_END, EPS_DECAY, LOG_EVERY, MAX_EP_STEPS,
    BUFFER_SIZE, B, T_CHUNK,
)

BATCH_SIZE = B * T_CHUNK   # 256 — matches Koopman-RL total transitions per update

# ---------------------------------------------------------------------------
# DQN replay buffer (standard flat buffer, no chunk sampling needed)
# ---------------------------------------------------------------------------

class DQNBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE):
        self.capacity = capacity
        self.states   = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.next_s   = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.ptr      = 0
        self.size     = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr]  = state
        self.next_s[self.ptr]  = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr]   = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) -> dict:
        idx = np.random.randint(0, self.size, size=n)
        return {
            "states":  torch.from_numpy(self.states[idx]),
            "next_s":  torch.from_numpy(self.next_s[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "dones":   torch.from_numpy(self.dones[idx]),
        }

    def ready(self, min_size: int) -> bool:
        return self.size > min_size


# ---------------------------------------------------------------------------
# DQN agent (no Koopman matrices)
# ---------------------------------------------------------------------------

class DQNAgent(nn.Module):
    """Same capacity as KoopmanAgent minus the K_a ParameterList."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(STATE_DIM, D)
        self.q_net   = QNetwork(D, N_ACTIONS)

    @torch.no_grad()
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randint(0, N_ACTIONS - 1)
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return self.q_net(self.encoder(s)).argmax(dim=-1).item()


# ---------------------------------------------------------------------------
# DQN training loop (mirrors train() in gravity_basin.py)
# ---------------------------------------------------------------------------

def train_dqn() -> dict:
    env   = GravityBasin()
    buf   = DQNBuffer()
    agent = DQNAgent()
    opt   = optim.Adam(agent.parameters(), lr=LR)

    # EMA target network (plain copies, not nn.Module)
    tgt_enc = copy.deepcopy(agent.encoder).eval()
    tgt_q   = copy.deepcopy(agent.q_net).eval()
    for p in tgt_enc.parameters(): p.requires_grad_(False)
    for p in tgt_q.parameters():   p.requires_grad_(False)

    state     = env.reset()
    ep_return = 0.0
    ep_steps  = 0
    episode_returns = []
    q_losses, recent_q = [], []

    print("=" * 62)
    print("  DQN Baseline — 2D Gravity Basin")
    print(f"  Network: Encoder [2→64→tanh→64→tanh→{D}] + QNet [{D}→64→ReLU→4]")
    print(f"  Target:  r + γ·max_a' Q_tgt(s',a')   [single-step Bellman]")
    print(f"  Batch:   {BATCH_SIZE} transitions (= {B}×{T_CHUNK} Koopman-RL batch)")
    print("=" * 62)
    print(f"\n[Warmup: collecting {WARMUP} random transitions...]\n")

    for step in range(1, N_STEPS + 1):
        eps    = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY)
        action = agent.act(state, epsilon=eps if step > WARMUP else 1.0)

        next_state, reward, done = env.step(state, action)
        buf.push(state, action, reward, next_state, done)
        ep_return += reward
        ep_steps  += 1

        if done or ep_steps >= MAX_EP_STEPS:
            episode_returns.append(ep_return)
            ep_return, ep_steps = 0.0, 0
            state = env.reset()
        else:
            state = next_state

        if step <= WARMUP or not buf.ready(BATCH_SIZE):
            continue

        # --- Single-step Bellman target ---
        batch  = buf.sample(BATCH_SIZE)
        s_b    = batch["states"]
        ns_b   = batch["next_s"]
        a_b    = batch["actions"]
        r_b    = batch["rewards"]
        d_b    = batch["dones"]

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

        # EMA update
        for p_t, p_o in zip(tgt_enc.parameters(), agent.encoder.parameters()):
            p_t.data.mul_(1 - EMA_TAU).add_(p_o.data, alpha=EMA_TAU)
        for p_t, p_o in zip(tgt_q.parameters(), agent.q_net.parameters()):
            p_t.data.mul_(1 - EMA_TAU).add_(p_o.data, alpha=EMA_TAU)

        recent_q.append(loss.item())

        if step % LOG_EVERY == 0:
            ret20   = np.mean(episode_returns[-20:]) if episode_returns else 0.0
            success = sum(r == 1.0 for r in episode_returns[-20:])
            mq      = np.mean(recent_q)
            print(f"  step {step:6d}  ε={eps:.3f}  "
                  f"L_q={mq:.4f}  ret={ret20:.3f}  succ/20={success}")
            q_losses.append(mq)
            recent_q.clear()

    return {
        "agent":           agent,
        "q_losses":        q_losses,
        "episode_returns": episode_returns,
    }


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _rolling(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _success_curve(returns, window=50):
    """Rolling success fraction (return == 1.0) over episodes."""
    hits = [float(r == 1.0) for r in returns]
    if len(hits) < window:
        return np.arange(len(hits)), np.array(hits)
    rolled = np.convolve(hits, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, len(hits)), rolled


def plot_comparison(koopman_hist: dict, dqn_hist: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Koopman-RL vs DQN — 2D Gravity Basin\n"
                 f"Identical network, optimizer, ε-schedule, budget ({N_STEPS:,} steps)",
                 fontsize=13)

    # 1. Episode returns (rolling mean)
    ax = axes[0]
    w = 50
    for hist, label, color in [
        (koopman_hist, f"Koopman-RL  (T={T_CHUNK} multi-step + Koopman)", "royalblue"),
        (dqn_hist,   "DQN  (single-step Bellman)",                     "tomato"),
    ]:
        r = hist["episode_returns"]
        ax.plot(_rolling(r, w), color=color, linewidth=1.5, label=label, alpha=0.9)
        ax.plot(np.array(r), color=color, linewidth=0.4, alpha=0.2)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8, label="Max return")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Return  (rolling {w}-ep mean)")
    ax.set_title("Episode Returns")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Rolling success rate
    ax = axes[1]
    for hist, label, color in [
        (koopman_hist, "Koopman-RL", "royalblue"),
        (dqn_hist,   "DQN",      "tomato"),
    ]:
        xs, ys = _success_curve(hist["episode_returns"], window=50)
        ax.plot(xs, ys * 100, color=color, linewidth=1.5, label=label)
    ax.axhline(100, color="green", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate  (%,  50-ep window)")
    ax.set_title("Rolling Success Rate")
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Q-loss curves
    ax = axes[2]
    steps_koopman = np.arange(1, len(koopman_hist["q_losses"]) + 1) * LOG_EVERY
    steps_dqn   = np.arange(1, len(dqn_hist["q_losses"])   + 1) * LOG_EVERY
    ax.semilogy(steps_koopman, koopman_hist["q_losses"],
                color="royalblue", linewidth=1.5, label="Koopman-RL  $\\mathcal{L}_Q$")
    ax.semilogy(steps_dqn,   dqn_hist["q_losses"],
                color="tomato",    linewidth=1.5, label="DQN  $\\mathcal{L}_Q$")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Q-loss (log scale)")
    ax.set_title("Q-Loss vs Environment Steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150)
    print("\nSaved → benchmark_results.png")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 62)
    print("  BENCHMARK: Koopman-RL vs DQN — 2D Gravity Basin")
    print("=" * 62 + "\n")

    # --- Koopman-RL ---
    print("\n>>> Running Koopman-RL ...\n")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    koopman_hist = train_koopman()

    koopman_agent = koopman_hist["agent"]
    sr_s, ms_s  = evaluate(koopman_agent, n_episodes=200)

    # --- DQN ---
    print("\n>>> Running DQN baseline ...\n")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    dqn_hist = train_dqn()

    dqn_agent  = dqn_hist["agent"]
    sr_d, ms_d = evaluate(dqn_agent, n_episodes=200)

    # --- Summary table ---
    print("\n" + "=" * 62)
    print(f"  {'Metric':<35} {'Koopman-RL':>10}  {'DQN':>10}")
    print("  " + "-" * 58)

    def first_ep_above(returns, threshold=0.8, window=10):
        """First episode where rolling success rate exceeds threshold."""
        hits = [float(r == 1.0) for r in returns]
        for i in range(window, len(hits)):
            if np.mean(hits[i - window:i]) >= threshold:
                return i
        return None

    fe_s = first_ep_above(koopman_hist["episode_returns"])
    fe_d = first_ep_above(dqn_hist["episode_returns"])

    print(f"  {'Final success rate (200 ep greedy)':<35} {sr_s*100:>9.1f}%  {sr_d*100:>9.1f}%")
    print(f"  {'Mean steps to goal (successes only)':<35} {ms_s:>10.1f}  {ms_d:>10.1f}")
    print(f"  {'Training episodes':<35} {len(koopman_hist['episode_returns']):>10}  {len(dqn_hist['episode_returns']):>10}")
    s_succ = sum(r == 1.0 for r in koopman_hist['episode_returns'])
    d_succ = sum(r == 1.0 for r in dqn_hist['episode_returns'])
    print(f"  {'Training successes':<35} {s_succ:>10}  {d_succ:>10}")
    pct_s = s_succ / len(koopman_hist['episode_returns']) * 100
    pct_d = d_succ / len(dqn_hist['episode_returns'])   * 100
    print(f"  {'Training success rate':<35} {pct_s:>9.1f}%  {pct_d:>9.1f}%")
    fe_s_str = str(fe_s) if fe_s else "never"
    fe_d_str = str(fe_d) if fe_d else "never"
    print(f"  {'Episode reaching 80% success (10-ep)':<35} {fe_s_str:>10}  {fe_d_str:>10}")
    print("=" * 62)

    plot_comparison(koopman_hist, dqn_hist)


if __name__ == "__main__":
    main()
