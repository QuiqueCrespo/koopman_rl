"""Checkpoint save/load helpers for KoopmanGradientPlanner."""

import threading

import torch

from koopman_rl.model import KoopmanGradientPlanner


def save_checkpoint(agent, target, history: dict, path: str) -> None:
    """
    Save agent weights + caller-supplied history dict to a .pt file.

    history should contain at minimum a 'config' sub-dict with keys
    state_dim, d, action_dim so that load_checkpoint can reconstruct the agent.
    """
    target_extras = (
        {"target_q_net": target.q_net.state_dict(),
         "target_pi_net": target.pi_net.state_dict()}
        if hasattr(target, 'q_net') else
        {"target_v_net": target.v_net.state_dict()}
    )
    torch.save({
        "agent_state_dict": agent.state_dict(),
        "target_encoder":   target.encoder.state_dict(),
        **target_extras,
        **history,
    }, path)
    print(f"  [ckpt] {path}")


def save_checkpoint_async(agent, target, history: dict, path: str) -> None:
    """Non-blocking checkpoint: snapshot state dicts then write in a daemon thread."""
    agent_sd = {k: v.cpu().clone() for k, v in agent.state_dict().items()}
    enc_sd   = {k: v.cpu().clone() for k, v in target.encoder.state_dict().items()}
    target_extras = (
        {"target_q_net":  {k: v.cpu().clone() for k, v in target.q_net.state_dict().items()},
         "target_pi_net": {k: v.cpu().clone() for k, v in target.pi_net.state_dict().items()}}
        if hasattr(target, 'q_net') else
        {"target_v_net": {k: v.cpu().clone() for k, v in target.v_net.state_dict().items()}}
    )
    payload  = {
        "agent_state_dict": agent_sd,
        "target_encoder":   enc_sd,
        **target_extras,
        **{k: (list(v) if hasattr(v, '__iter__') and not isinstance(v, dict) else v)
           for k, v in history.items()},
    }
    def _write():
        torch.save(payload, path)
        print(f"  [ckpt] {path}")
    threading.Thread(target=_write, daemon=True).start()


def load_checkpoint(path: str, device=None) -> tuple:
    """
    Reload a saved agent from disk.

    Returns (agent, ckpt_dict).  ckpt_dict contains all saved keys including
    episode_returns, koop_log, v_log, config, etc.

    Usage:
        agent, ckpt = load_checkpoint("output/checkpoints/kgp_pendulum_best.pt")
    """
    ckpt  = torch.load(path, map_location=device or "cpu")
    cfg   = ckpt["config"]
    agent = KoopmanGradientPlanner(
        state_dim=cfg["state_dim"], d=cfg["d"], n_actions=cfg["action_dim"]
    )
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()
    return agent, ckpt
