"""
collapse_monitor.py
===================
Detects representation collapse during JEPA training.

Collapse = the model outputs nearly identical latent vectors for all inputs,
reducing loss to ~0 without learning anything useful.

Two signals are tracked per step:
  variance    – mean per-dimension variance across the batch (healthy > 0.1)
  cosine_sim  – mean pairwise cosine similarity in the batch  (healthy < 0.9)
"""

import torch
import torch.nn.functional as F


class CollapseMonitor:
    """
    Feed it a batch of latent vectors after each training step.
    It tracks a rolling window of the last `window` steps and flags collapse.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        cosine_threshold:   float = 0.98,
        window:             int   = 20,
    ):
        self.variance_threshold = variance_threshold
        self.cosine_threshold   = cosine_threshold
        self.window             = window

        self._variance_history:  list[float] = []
        self._cosine_history:    list[float] = []

    # ------------------------------------------------------------------
    def update(self, latents: torch.Tensor) -> dict:
        """
        Args:
            latents: [B, d_model] — context_latent or predicted_latent from a step

        Returns:
            dict with keys: variance, cosine_sim, collapsed
        """
        with torch.no_grad():
            # Mean per-dimension variance across the batch
            # Move to CPU first — MPS has a known bug with .var() on some layouts
            variance = latents.cpu().float().var(dim=0).mean().item()

            # Mean pairwise cosine similarity (upper triangle only, for efficiency)
            normed = F.normalize(latents, dim=1)          # [B, d_model]
            sim_matrix = normed @ normed.T                 # [B, B]
            B = latents.size(0)
            if B > 1:
                # Extract upper triangle (exclude diagonal)
                idx = torch.triu_indices(B, B, offset=1)
                cosine_sim = sim_matrix[idx[0], idx[1]].mean().item()
            else:
                cosine_sim = 1.0   # single sample — can't measure

        # Keep rolling window
        self._variance_history.append(variance)
        self._cosine_history.append(cosine_sim)
        if len(self._variance_history) > self.window:
            self._variance_history.pop(0)
            self._cosine_history.pop(0)

        collapsed = (
            variance   < self.variance_threshold or
            cosine_sim > self.cosine_threshold
        )

        return {
            "variance":   variance,
            "cosine_sim": cosine_sim,
            "collapsed":  collapsed,
        }

    # ------------------------------------------------------------------
    def is_collapsed(self) -> bool:
        """True if the *rolling average* over the last `window` steps signals collapse."""
        if not self._variance_history:
            return False
        avg_var = sum(self._variance_history) / len(self._variance_history)
        avg_cos = sum(self._cosine_history)   / len(self._cosine_history)
        return avg_var < self.variance_threshold or avg_cos > self.cosine_threshold

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """One-line human-readable status."""
        if not self._variance_history:
            return "CollapseMonitor: no data yet"
        avg_var = sum(self._variance_history) / len(self._variance_history)
        avg_cos = sum(self._cosine_history)   / len(self._cosine_history)
        status  = "COLLAPSED" if self.is_collapsed() else "healthy"
        return (
            f"CollapseMonitor [{status}] "
            f"variance={avg_var:.4f} (>{self.variance_threshold}) | "
            f"cosine_sim={avg_cos:.4f} (<{self.cosine_threshold})"
        )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    monitor = CollapseMonitor()

    print("--- Healthy latents (random) ---")
    for _ in range(5):
        latents = torch.randn(32, 128)
        info = monitor.update(latents)
        print(f"  variance={info['variance']:.4f}  cosine_sim={info['cosine_sim']:.4f}  collapsed={info['collapsed']}")
    print(monitor.summary())
    print()

    monitor = CollapseMonitor()
    print("--- Collapsed latents (all identical) ---")
    for _ in range(5):
        latents = torch.ones(32, 128)   # same vector repeated
        info = monitor.update(latents)
        print(f"  variance={info['variance']:.4f}  cosine_sim={info['cosine_sim']:.4f}  collapsed={info['collapsed']}")
    print(monitor.summary())
    print()
    print("All checks passed.")

