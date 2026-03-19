"""
Loss functions for the gravity basin Koopman-RL agent.
"""

import torch
import torch.nn.functional as F

# Module-level defaults (backward compat)
GAMMA               = 0.95
LAMBDA_KOOP         = 1.0    # Koopman linearity loss weight
LAMBDA_BISIM        = 0.5    # bisimulation metric loss weight
LAMBDA_CONTRASTIVE  = 0.1    # repulsion hinge on unit sphere
LAMBDA_ISOMETRIC    = 0.05   # latent ↔ physical distance alignment
TAU_START           = 1.0    # AWR temperature start (soft pruning)
TAU_END             = 0.1    # AWR temperature end   (hard pruning)


def compute_v_targets(
    rewards:       torch.Tensor,   # [B, T]
    dones:         torch.Tensor,   # [B, T]  float (0 or 1)
    v_targets_all: torch.Tensor,   # [B, T+1]  target-net V for every state
    W:             torch.Tensor,   # [B, T]  Koopman advantage weights ∈ (0, 1]
    gamma:         float = GAMMA,
) -> torch.Tensor:                 # [B, T]
    """
    Tree-Backup backward recursion.

    At each step t the "next value" blends:
      - next_val          : empirical T-step return from t+1 onward
      - v_targets_all[:,t+1] : clean 1-step bootstrap from target network

    Blend weight W[:,t] = exp(A_t / τ), where A_t is the Koopman advantage.
    W ≈ 1 → keep empirical chain (good action, diffuse reward signal).
    W ≈ 0 → sever chain, fall back to 1-step TD (bad action, firewall).
    """
    B, T     = rewards.shape
    V        = torch.zeros(B, T, device=rewards.device)
    next_val = v_targets_all[:, -1]

    for t in reversed(range(T)):
        blended_next = (W[:, t] * next_val
                        + (1.0 - W[:, t]) * v_targets_all[:, t + 1])
        V[:, t]  = rewards[:, t] + gamma * blended_next * (1.0 - dones[:, t])
        next_val = V[:, t]

    return V


def compute_contrastive_loss(z_src: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Pushes random states apart to prevent hypersphere collapse,
    but allows the encoder to freely warp the geometry to satisfy Koopman linearity.
    """
    N    = z_src.size(0)
    perm = torch.randperm(N, device=z_src.device)

    z_i = z_src
    z_j = z_src[perm]

    dists    = F.pairwise_distance(z_i, z_j, p=2, eps=1e-6)
    repulsion = F.relu(margin - dists).pow(2).mean()
    return repulsion


def compute_isometric_loss(z_src: torch.Tensor, s_src: torch.Tensor) -> torch.Tensor:
    """
    Forces the latent geometry to preserve the physical state geometry.
    Prevents hypersphere collapse without relying on sparse rewards.
    """
    N    = z_src.size(0)
    perm = torch.randperm(N, device=z_src.device)

    z_i, z_j = z_src, z_src[perm]
    s_i, s_j = s_src, s_src[perm]

    z_dist = F.pairwise_distance(z_i, z_j, p=2, eps=1e-6)
    s_dist = F.pairwise_distance(s_i, s_j, p=2, eps=1e-6)
    return F.mse_loss(z_dist, s_dist)


def compute_bisimulation_loss(
    z_src:        torch.Tensor,   # [N, d]  online encoder latents
    z_dst_target: torch.Tensor,   # [N, d]  target encoder next-state latents (frozen)
    rewards:      torch.Tensor,   # [N]     immediate rewards
    dones:        torch.Tensor,   # [N]     float (0 or 1) — terminal flags
    gamma:        float = GAMMA,
) -> torch.Tensor:
    """
    Deep bisimulation metric loss (Zhang et al. 2020).

    Forces ||z_i - z_j|| → |r_i - r_j| + γ · ||z'_i - z'_j||_target

    Terminal transitions (done=1) have no valid next state — z_dst_target
    at those positions holds the reset state of the next episode, which is
    meaningless for bisimulation. The γ·next_z_dist term is zeroed out for
    any pair where either transition was terminal.
    """
    N    = z_src.size(0)
    perm = torch.randperm(N, device=z_src.device)

    z_i,  z_j  = z_src,        z_src[perm]
    zt_i, zt_j = z_dst_target, z_dst_target[perm]
    r_i,  r_j  = rewards,      rewards[perm]
    d_i,  d_j  = dones,        dones[perm]

    z_dist      = F.pairwise_distance(z_i,  z_j,  p=2, eps=1e-6)
    r_dist      = (r_i - r_j).abs()
    next_z_dist = F.pairwise_distance(zt_i, zt_j, p=2, eps=1e-6)
    # Zero out next-state contribution if either transition was terminal
    next_weight = (1.0 - d_i) * (1.0 - d_j)
    target_dist = (r_dist + gamma * next_z_dist * next_weight).detach()

    return F.mse_loss(z_dist, target_dist)
