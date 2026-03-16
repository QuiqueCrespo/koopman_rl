"""
Replay buffer for the gravity basin experiments.

Two sampling modes serve two distinct architectural needs:
  sample_transitions — flat random batches for stable NN gradient updates
  sample_chunks      — contiguous trajectories for Sheaf graph temporal topology
"""

import numpy as np

from sheaf_rl.config import Config, BufferConfig, EnvConfig

# Module-level defaults (backward compat)
BUFFER_SIZE = 100_000
B           = 16     # trajectory chunks per gradient step
T_CHUNK     = 16     # steps per chunk


class ReplayBuffer:
    """
    Circular buffer storing (s, a, r, s', done) tuples.

    sample_transitions() draws independent random transitions for TD learning.
    sample_chunks() draws contiguous windows that preserve temporal ordering,
    which is required for the Sheaf graph builder so that consecutive nodes
    share src/dst states and form unbroken causal pipes for Richardson diffusion.
    """

    def __init__(self, capacity: int = BUFFER_SIZE, state_dim: int = 2):
        self.capacity = capacity
        self.states   = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_s   = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)
        self.ptr      = 0
        self.size     = 0

    @classmethod
    def from_cfg(cls, cfg: Config) -> "ReplayBuffer":
        return cls(capacity=cfg.buffer.capacity, state_dim=cfg.env.state_dim)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.states[self.ptr]  = state
        self.next_s[self.ptr]  = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr]   = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def ready(self, min_size: int) -> bool:
        return self.size >= min_size

    def sample_transitions(self, batch_size: int) -> dict:
        """Flat random sampling for neural network TD updates."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "states":  self.states[idx],
            "next_s":  self.next_s[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "dones":   self.dones[idx],
        }

    def sample_chunks(self, n_chunks: int, chunk_len: int,
                      force_goal: bool = True,
                      stratified: bool = True) -> dict | None:
        """
        Sample n_chunks contiguous windows of length chunk_len using
        temporal stratification.

        The buffer timeline is divided into n_chunks equal segments and
        exactly one chunk is drawn from each segment.  This guarantees the
        graph always contains early (random) + mid (learning) + late
        (exploitation) trajectories regardless of how biased the current
        policy is — preventing the value-surface from collapsing to a
        uniform high value at low epsilon.

        Within each segment a chunk is rejected if it:
          - crosses the circular write-pointer (temporal discontinuity), or
          - spans an episode boundary (done=True mid-chunk).
        Up to 20 retries are attempted; if none succeed the boundary check
        is relaxed so sampling always terminates.
        """
        if self.size < chunk_len + 1:
            return None

        max_start = self.size - chunk_len - 1
        starts    = []

        # ------------------------------------------------------------------
        # Temporal stratification  (disabled → pure uniform random starts)
        # ------------------------------------------------------------------
        n_seg = n_chunks - (1 if force_goal else 0)

        if not stratified:
            # Uniform: draw n_seg starts independently from [0, max_start]
            for _ in range(n_seg):
                starts.append(int(np.random.randint(0, max_start)))
        else:
            seg_len = max(1, max_start // n_seg)

            for i in range(n_seg):
                seg_lo = i * seg_len
                seg_hi = min((i + 1) * seg_len, max_start)
                if seg_lo >= seg_hi:
                    seg_lo = 0

                chosen = None
                # Try to find a chunk with no wrap-around and no episode boundary
                for _ in range(20):
                    s = int(np.random.randint(seg_lo, seg_hi + 1))
                    if (not (s < self.ptr <= s + chunk_len) and
                            not self.dones[s : s + chunk_len - 1].any()):
                        chosen = s
                        break
                # Fallback: relax episode-boundary check, keep wrap-around check
                if chosen is None:
                    for _ in range(50):
                        s = int(np.random.randint(seg_lo, seg_hi + 1))
                        if not (s < self.ptr <= s + chunk_len):
                            chosen = s
                            break
                starts.append(chosen if chosen is not None
                              else int(np.random.randint(0, max_start)))

        # ------------------------------------------------------------------
        # Force goal: one chunk anchored to a successful terminal transition
        # ------------------------------------------------------------------
        if force_goal:
            goal_idx = np.where(
                (self.dones[:self.size] == 1.0) & (self.rewards[:self.size] > 0.0)
            )[0]
            if len(goal_idx) > 0:
                g = int(np.random.choice(goal_idx))
                starts.append(int(np.clip(g - chunk_len + 1, 0, max_start)))
            else:
                starts.append(int(np.random.randint(0, max_start)))

        idx_flat = (np.array(starts)[:, None] + np.arange(chunk_len)).reshape(-1)
        return {
            "states":  self.states[idx_flat],
            "next_s":  self.next_s[idx_flat],
            "actions": self.actions[idx_flat],
            "rewards": self.rewards[idx_flat],
            "dones":   self.dones[idx_flat],
        }
