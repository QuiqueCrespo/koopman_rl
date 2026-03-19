"""
Diagnostic hook for Koopman-RL gravity basin training.

Records per-step health metrics covering all five identified failure modes:
  1. Train/inference normalization mismatch (F.normalize in act but not in train)
  2. Bisimulation self-reinforcing collapse (target_dist drives z together)
  3. K_a orthogonality loss (K_a drifts from SO(d) under gradient steps)
  4. AWR tau-driven chain severing (W→0 when value function is noisy)
  5. Latent space geometric collapse (contrastive/isometric losses absent)

Usage:
    hook = DiagnosticHook(agent, target, n_probe=256)
    history = train(hook=hook)
    hook.plot("collapse_diagnostics.png")
    hook.leading_indicators()
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional

from env    import N_ACTIONS, STATE_DIM, GOAL_X, GOAL_Y, ACTION_NAMES, ACTION_COLORS, GravityBasin
from model  import KoopmanAgent, TargetNetwork, D
from losses import GAMMA


# ---------------------------------------------------------------------------
# Probe set — fixed reference grid, built once
# ---------------------------------------------------------------------------

@dataclass
class ProbeSet:
    """
    256 fixed states spanning [-1,1]^2, split into goal-zone / non-goal.
    Never changes after construction so all metrics are comparable across steps.
    """
    states_t  : torch.Tensor   # [N, 2]
    goal_mask : torch.Tensor   # [N] bool

    @classmethod
    def build(cls, n: int = 256, seed: int = 0) -> "ProbeSet":
        rng = np.random.RandomState(seed)
        pts = rng.uniform(-1.0, 1.0, (n, STATE_DIM)).astype(np.float32)
        goal_mask = torch.tensor(
            (pts[:, 0] > GOAL_X) & (pts[:, 1] > GOAL_Y)
        )
        return cls(states_t=torch.from_numpy(pts), goal_mask=goal_mask)


# ---------------------------------------------------------------------------
# Step record — one snapshot per LOG_EVERY boundary
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    step              : int
    success_last20    : int

    # Latent collapse
    z_dist_median     : float
    z_dist_min        : float

    # K_a health (list of N_ACTIONS floats)
    ka_ortho_dev      : list
    ka_output_norm    : list

    # Value function
    v_variance        : float
    v_goal_gap        : float

    # AWR weights
    w_mean            : float
    w_min             : float

    # Train/inference mismatch (list of N_ACTIONS floats)
    norm_discrepancy  : list

    # Bisimulation target statistics
    bisim_target_mean : float
    bisim_target_max  : float


# ---------------------------------------------------------------------------
# DiagnosticHook
# ---------------------------------------------------------------------------

class DiagnosticHook:
    """
    Attaches to train() via the hook= parameter.  Call record() at every
    LOG_EVERY boundary; the required tensors are already in scope there.
    """

    def __init__(
        self,
        agent      : KoopmanAgent,
        target     : TargetNetwork,
        n_probe    : int = 256,
        probe_seed : int = 0,
    ):
        self.agent   = agent
        self.target  = target
        self.probes  = ProbeSet.build(n_probe, seed=probe_seed)
        self.history : list[StepRecord] = []

    # ------------------------------------------------------------------
    # Main callback — called once per LOG_EVERY step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def record(
        self,
        step          : int,
        success_last20: int,
        z_src         : torch.Tensor,   # [N_flat, D]
        W             : torch.Tensor,   # [B, T]
        z_dst_target  : torch.Tensor,   # [N_flat, D]
        r_f           : torch.Tensor,   # [N_flat]
    ) -> StepRecord:
        agent  = self.agent
        probes = self.probes
        dev    = z_src.device

        # Move probe states to same device as agent
        states_dev = probes.states_t.to(dev)
        goal_mask  = probes.goal_mask.to(dev)

        # ---- 1. LATENT COLLAPSE ----------------------------------------
        # Use random-permutation pairs (same as loss functions) — avoids O(N^2)
        n_sub  = min(z_src.size(0), 512)
        z_sub  = z_src[:n_sub]
        perm   = torch.randperm(n_sub, device=dev)
        dists  = F.pairwise_distance(z_sub, z_sub[perm], p=2, eps=1e-6)
        z_dist_median = dists.median().item()
        z_dist_min    = dists.min().item()

        # ---- 2. K_a ORTHOGONALITY AND OUTPUT NORM ----------------------
        # Encode probes once; reuse for K_a norms and V metrics below
        z_probe = agent.encode(states_dev)   # [N, D]

        ka_ortho_dev   = []
        ka_output_norm = []
        for a in range(N_ACTIONS):
            K   = agent.K[a]
            KtK = K.T @ K                                     # [D, D]
            dev_frob = (KtK - torch.eye(D, device=dev)).norm(p="fro").item()
            ka_ortho_dev.append(dev_frob)

            Kz = z_probe @ K.T                                # [N, D]
            ka_output_norm.append(Kz.norm(dim=-1).mean().item())

        # ---- 3. VALUE FUNCTION HEALTH ----------------------------------
        v_probe    = agent.v_net(z_probe)                     # [N]
        v_variance = v_probe.var().item()

        v_goal   = v_probe[goal_mask]
        v_random = v_probe[~goal_mask]
        if v_goal.numel() > 0 and v_random.numel() > 0:
            v_goal_gap = (v_goal.mean() - v_random.mean()).item()
        else:
            v_goal_gap = float("nan")

        # ---- 4. AWR WEIGHTS --------------------------------------------
        w_mean = W.mean().item()
        w_min  = W.min().item()

        # ---- 5. TRAIN / INFERENCE NORMALIZATION DISCREPANCY ------------
        # act() applies F.normalize to K_a z; training does not.
        # Discrepancy = how much the policy rank-order could change.
        norm_discrepancy = []
        for a in range(N_ACTIONS):
            K  = agent.K[a]
            Kz = z_probe @ K.T                                # [N, D] — as in training
            v_train     = agent.v_net(Kz)
            v_inference = agent.v_net(F.normalize(Kz, p=2, dim=-1))
            norm_discrepancy.append((v_train - v_inference).abs().max().item())

        # ---- 6. BISIMULATION TARGET STATISTICS -------------------------
        perm2       = torch.randperm(r_f.size(0), device=dev)
        zt_i, zt_j  = z_dst_target, z_dst_target[perm2]
        r_i, r_j    = r_f, r_f[perm2]
        next_z_dist = F.pairwise_distance(zt_i, zt_j, p=2, eps=1e-6)
        target_dist = (r_i - r_j).abs() + GAMMA * next_z_dist
        bisim_target_mean = target_dist.mean().item()
        bisim_target_max  = target_dist.max().item()

        # ---- Pack and store --------------------------------------------
        rec = StepRecord(
            step=step,
            success_last20=success_last20,
            z_dist_median=z_dist_median,
            z_dist_min=z_dist_min,
            ka_ortho_dev=ka_ortho_dev,
            ka_output_norm=ka_output_norm,
            v_variance=v_variance,
            v_goal_gap=v_goal_gap,
            w_mean=w_mean,
            w_min=w_min,
            norm_discrepancy=norm_discrepancy,
            bisim_target_mean=bisim_target_mean,
            bisim_target_max=bisim_target_max,
        )
        self.history.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Internal: flatten history to numpy arrays
    # ------------------------------------------------------------------

    def _series(self) -> dict:
        h = self.history
        if not h:
            return {}

        def arr(fn):
            return np.array([fn(r) for r in h])

        s = {
            "steps":          arr(lambda r: r.step),
            "success":        arr(lambda r: r.success_last20),
            "z_dist_median":  arr(lambda r: r.z_dist_median),
            "z_dist_min":     arr(lambda r: r.z_dist_min),
            "ka_ortho_mean":  arr(lambda r: float(np.mean(r.ka_ortho_dev))),
            "ka_norm_mean":   arr(lambda r: float(np.mean(r.ka_output_norm))),
            "v_variance":     arr(lambda r: r.v_variance),
            "v_goal_gap":     arr(lambda r: r.v_goal_gap),
            "w_mean":         arr(lambda r: r.w_mean),
            "w_min":          arr(lambda r: r.w_min),
            "norm_disc_mean": arr(lambda r: float(np.mean(r.norm_discrepancy))),
            "bisim_mean":     arr(lambda r: r.bisim_target_mean),
            "bisim_max":      arr(lambda r: r.bisim_target_max),
        }
        # Per-action series
        for a in range(N_ACTIONS):
            s[f"ka_ortho_a{a}"] = arr(lambda r, a=a: r.ka_ortho_dev[a])
            s[f"ka_norm_a{a}"]  = arr(lambda r, a=a: r.ka_output_norm[a])
            s[f"norm_disc_a{a}"]= arr(lambda r, a=a: r.norm_discrepancy[a])
        return s

    # ------------------------------------------------------------------
    # Collapse window detection
    # ------------------------------------------------------------------

    def _collapse_window(
        self, success: np.ndarray, steps: np.ndarray
    ) -> tuple[Optional[int], Optional[int]]:
        peak_val = success.max()
        if peak_val < 3:
            return None, None   # never really learned

        peak_idx  = int(success.argmax())
        threshold = peak_val * 0.5
        T         = len(success)

        after_peak  = np.arange(T) > peak_idx
        below_thresh = success < threshold
        candidates   = np.where(after_peak & below_thresh)[0]

        col_start = int(steps[peak_idx])
        col_end   = int(steps[candidates[0]]) if len(candidates) > 0 else None
        return col_start, col_end

    # ------------------------------------------------------------------
    # Diagnostic plot
    # ------------------------------------------------------------------

    def plot(
        self,
        save_path : str = "collapse_diagnostics.png",
        dpi       : int = 150,
    ) -> None:
        s = self._series()
        if not s:
            print("No recorded data — run training first.")
            return

        steps   = s["steps"]
        success = s["success"]
        col_start, col_end = self._collapse_window(success, steps)

        fig = plt.figure(figsize=(20, 24))
        gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

        def make_ax(row, col):
            return fig.add_subplot(gs[row, col])

        def shade(ax):
            if col_start is None:
                return
            end = col_end if col_end is not None else int(steps[-1])
            ax.axvspan(col_start, end, alpha=0.12, color="red", zorder=0,
                       label="Collapse window" if col_end else "Post-peak")

        def fmt(ax, title, ylabel, xlabel="Step"):
            ax.set_title(title, fontsize=9, pad=4)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.25)

        # ---- ROW 0 ---------------------------------------------------

        # (0,0) Success
        ax = make_ax(0, 0)
        ax.plot(steps, success, color="green", linewidth=2)
        ax.axhline(20, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-0.5, 22)
        shade(ax)
        if col_start:
            ax.legend(fontsize=7)
        fmt(ax, "Policy Performance", "Successes / last 20 ep")

        # (0,1) Latent collapse
        ax = make_ax(0, 1)
        ax.plot(steps, s["z_dist_median"], label="median", color="royalblue", lw=1.5)
        ax.plot(steps, s["z_dist_min"],    label="min",    color="steelblue", lw=1.0, ls="--")
        shade(ax)
        ax.legend(fontsize=7)
        fmt(ax, "Latent Pairwise Distance  [FM 2, 5]",
            r"$\|z_i - z_j\|$")

        # (0,2) Bisimulation target stats
        ax = make_ax(0, 2)
        ax.plot(steps, s["bisim_mean"], label="mean", color="darkorange", lw=1.5)
        ax.plot(steps, s["bisim_max"],  label="max",  color="orange",     lw=1.0, ls="--")
        shade(ax)
        ax.legend(fontsize=7)
        fmt(ax, "Bisimulation Target Magnitude  [FM 2]",
            r"$|r_i - r_j| + \gamma \|z'_i - z'_j\|$")

        # ---- ROW 1 ---------------------------------------------------

        # (1,0) K_a orthogonality per action
        ax = make_ax(1, 0)
        for a in range(N_ACTIONS):
            ax.plot(steps, s[f"ka_ortho_a{a}"],
                    color=ACTION_COLORS[a], label=ACTION_NAMES[a], lw=1.2)
        ax.axhline(0.0, color="black", ls=":", lw=0.8)
        shade(ax)
        ax.legend(fontsize=7)
        fmt(ax, r"$K_a$ Orthogonality Loss  [FM 3]",
            r"$\|K_a^T K_a - I\|_F$")

        # (1,1) K_a output norm per action
        ax = make_ax(1, 1)
        for a in range(N_ACTIONS):
            ax.plot(steps, s[f"ka_norm_a{a}"],
                    color=ACTION_COLORS[a], label=ACTION_NAMES[a], lw=1.2)
        ax.axhline(1.0, color="black", ls=":", lw=0.8, label="Expected (orthogonal)")
        shade(ax)
        ax.legend(fontsize=7)
        fmt(ax, r"$K_a$ Output Norm  [FM 1+3]",
            r"mean $\|K_a z\|$")

        # (1,2) AWR weights
        ax = make_ax(1, 2)
        ax.plot(steps, s["w_mean"], label="mean W", color="purple",  lw=1.5)
        ax.plot(steps, s["w_min"],  label="min W",  color="violet",  lw=1.0, ls="--")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.0, color="black", ls=":", lw=0.8)
        shade(ax)
        ax.legend(fontsize=7)
        fmt(ax, "AWR Weight Distribution  [FM 4]", "W = exp(A/τ)")

        # ---- ROW 2 ---------------------------------------------------

        # (2,0) Value variance
        ax = make_ax(2, 0)
        ax.plot(steps, s["v_variance"], color="crimson", lw=1.5)
        shade(ax)
        fmt(ax, "Value Function Variance  (→0 = collapse)",
            r"Var$(V_\psi(z))$ over probe set")

        # (2,1) Value discrimination gap
        ax = make_ax(2, 1)
        ax.plot(steps, s["v_goal_gap"], color="teal", lw=1.5)
        ax.axhline(0.0, color="black", ls=":", lw=0.8)
        shade(ax)
        fmt(ax, "Value Discrimination: Goal vs Random",
            r"$\bar{V}(\text{goal}) - \bar{V}(\text{random})$")

        # (2,2) Train/inference mismatch per action
        ax = make_ax(2, 2)
        for a in range(N_ACTIONS):
            ax.plot(steps, s[f"norm_disc_a{a}"],
                    color=ACTION_COLORS[a], label=ACTION_NAMES[a], lw=1.2)
        shade(ax)
        ax.legend(fontsize=7)
        fmt(ax, "Train vs Inference Mismatch  [FM 1]",
            r"max $|V(K_a z) - V(\hat{K_a z})|$")

        # ---- ROW 3: Full-width correlation matrix --------------------

        ax_corr = fig.add_subplot(gs[3, :])

        MNAMES = [
            "success", "z_dist_med", "z_dist_min", "ka_ortho",
            "ka_norm", "V_var", "V_gap", "W_mean", "W_min",
            "norm_disc", "bisim_mean", "bisim_max",
        ]
        cols = [
            success,
            s["z_dist_median"], s["z_dist_min"],
            s["ka_ortho_mean"], s["ka_norm_mean"],
            s["v_variance"],    s["v_goal_gap"],
            s["w_mean"],        s["w_min"],
            s["norm_disc_mean"],
            s["bisim_mean"],    s["bisim_max"],
        ]
        # Replace NaN in v_goal_gap with column mean so corrcoef doesn't break
        for i, c in enumerate(cols):
            if np.any(np.isnan(c)):
                mean_val = np.nanmean(c)
                cols[i] = np.where(np.isnan(c), mean_val, c)

        matrix = np.column_stack(cols)
        corr   = np.corrcoef(matrix.T)

        im = ax_corr.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax_corr, fraction=0.015, pad=0.01)
        n = len(MNAMES)
        ax_corr.set_xticks(range(n))
        ax_corr.set_xticklabels(MNAMES, rotation=40, ha="right", fontsize=9)
        ax_corr.set_yticks(range(n))
        ax_corr.set_yticklabels(MNAMES, fontsize=9)
        ax_corr.set_title(
            "Pearson Correlation Between All Diagnostic Metrics\n"
            "(row 0 = correlation with success — look for early predictors)",
            fontsize=10,
        )
        for i in range(n):
            for j in range(n):
                r_val = corr[i, j]
                if abs(r_val) > 0.35 and i != j:
                    color = "white" if abs(r_val) > 0.75 else "black"
                    ax_corr.text(j, i, f"{r_val:.2f}",
                                 ha="center", va="center",
                                 fontsize=7, color=color)

        fig.suptitle(
            "Koopman-RL Collapse Diagnostics\n"
            "Red shading = collapse window (success peak → 50% drop)\n"
            "FM = Failure Mode (1=norm mismatch, 2=bisim, 3=K_a ortho, 4=AWR, 5=collapse)",
            fontsize=11,
        )

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"\nSaved → {save_path}")
        plt.close()

    # ------------------------------------------------------------------
    # Leading indicator analysis
    # ------------------------------------------------------------------

    def leading_indicators(self, lags: list = [1, 2, 3]) -> None:
        """
        Prints Pearson r between each metric (at t-lag) and success (at t).
        A metric with high |r| at lag>0 is a LEADING indicator of collapse.
        """
        s = self._series()
        if not s:
            print("No data.")
            return

        success = s["success"]
        T       = len(success)

        MNAMES = [
            "z_dist_median", "z_dist_min", "ka_ortho_mean", "ka_norm_mean",
            "v_variance", "v_goal_gap", "w_mean", "w_min",
            "norm_disc_mean", "bisim_mean", "bisim_max",
        ]

        print("\n" + "=" * 68)
        print("  Leading Indicator Analysis")
        print("  Pearson r(metric[t-lag], success[t])  — positive r means")
        print("  metric falls BEFORE success falls (early warning signal)")
        print("=" * 68)

        for lag in lags:
            if lag >= T:
                continue
            print(f"\n  Lag = {lag} log-intervals ({lag * 2000} steps):")
            rows = []
            for name in MNAMES:
                col = s[name]
                # Replace NaN
                col = np.where(np.isnan(col), np.nanmean(col), col)
                x   = col[:T - lag]
                y   = success[lag:]
                if x.std() < 1e-9:
                    r = float("nan")
                else:
                    r = np.corrcoef(x, y)[0, 1]
                rows.append((abs(r) if not np.isnan(r) else -1, r, name))
            rows.sort(reverse=True)
            for _, r, name in rows:
                bar = "█" * int(abs(r) * 20) if not np.isnan(r) else ""
                print(f"    {name:<20}  r={r:+.3f}  {bar}")

        print()

    # ------------------------------------------------------------------
    # Quick console summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a concise table of the last few recorded steps."""
        if not self.history:
            print("No data recorded.")
            return
        print(f"\n{'step':>8}  {'succ':>5}  {'z_med':>6}  {'ka_orth':>7}  "
              f"{'V_var':>7}  {'V_gap':>6}  {'W_mean':>6}  {'norm_d':>6}  {'bisim':>6}")
        print("-" * 75)
        for r in self.history[-10:]:
            print(f"  {r.step:6d}  {r.success_last20:5d}  "
                  f"{r.z_dist_median:6.3f}  {np.mean(r.ka_ortho_dev):7.3f}  "
                  f"{r.v_variance:7.4f}  {r.v_goal_gap:6.3f}  "
                  f"{r.w_mean:6.3f}  {np.mean(r.norm_discrepancy):6.4f}  "
                  f"{r.bisim_target_mean:6.4f}")


# ---------------------------------------------------------------------------
# Smoke-test / standalone entry point
# ---------------------------------------------------------------------------

def main():
    import random
    import sys
    sys.path.insert(0, ".")

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    from train import train

    print("Running 10k-step smoke test with DiagnosticHook...")
    history = train(n_steps_override=10_000, hook=None)  # hook wired in train.py

    # Standalone: build a fresh agent and test hook construction
    from model import KoopmanAgent, TargetNetwork
    agent  = KoopmanAgent()
    target = TargetNetwork(agent)
    hook   = DiagnosticHook(agent, target, n_probe=64)

    # Fake a record call with random tensors to validate shapes
    z_src        = F.normalize(torch.randn(256, D), dim=-1)
    W            = torch.rand(16, 16)
    z_dst_target = F.normalize(torch.randn(256, D), dim=-1)
    r_f          = torch.zeros(256)

    rec = hook.record(
        step=2000, success_last20=5,
        z_src=z_src, W=W, z_dst_target=z_dst_target, r_f=r_f,
    )
    print(f"\nSmoke test record:")
    print(f"  z_dist_median    = {rec.z_dist_median:.4f}")
    print(f"  ka_ortho_dev     = {[f'{x:.4f}' for x in rec.ka_ortho_dev]}")
    print(f"  ka_output_norm   = {[f'{x:.4f}' for x in rec.ka_output_norm]}")
    print(f"  v_variance       = {rec.v_variance:.6f}")
    print(f"  v_goal_gap       = {rec.v_goal_gap:.4f}")
    print(f"  norm_discrepancy = {[f'{x:.6f}' for x in rec.norm_discrepancy]}")
    print(f"  bisim_mean/max   = {rec.bisim_target_mean:.4f} / {rec.bisim_target_max:.4f}")
    print("\nSmoke test PASSED")


if __name__ == "__main__":
    main()
