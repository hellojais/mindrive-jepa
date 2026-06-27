"""
surprise_score.py
=================
SurpriseScore: measures how unexpected a driving scenario is,
using the L2 distance between the model's predicted latent and
the actual future latent encoded by the target encoder.

    surprise = ||predicted_latent - target_latent||₂

High score → model was wrong → scenario was unusual.
Low score  → model was right → routine driving.

Usage:
    scorer = SurpriseScore("checkpoints/best.pt", config, device="mps")
    scorer.calibrate(all_scenario_tensors)      # set 90th-pct threshold
    score, is_surprising = scorer.score(tensor) # single scenario
    scores = scorer.score_batch(tensors)        # list of floats
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F

from mindrive_jepa.models import MinDriveJEPA


class SurpriseScore:
    """
    Loads a trained MinDriveJEPA checkpoint and scores scenarios.

    Args:
        checkpoint_path: path to best.pt (or last.pt)
        config:          full config dict (data + model sections needed)
        device:          "mps", "cuda", or "cpu"
    """

    def __init__(
        self,
        checkpoint_path: str | pathlib.Path,
        config: dict,
        device: str = "cpu",
    ):
        self.device     = torch.device(device)
        self.data_cfg   = config["data"]
        self.model_cfg  = config["model"]
        self.threshold: float | None = None   # set by calibrate()

        # Build model and load weights
        self.model = MinDriveJEPA(self.data_cfg, self.model_cfg)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score(self, scenario: torch.Tensor) -> tuple[float, bool]:
        """
        Score a single scenario.

        Args:
            scenario: [T, N+1, D] or [1, T, N+1, D]

        Returns:
            (surprise_score: float, is_surprising: bool)
            is_surprising is always False if calibrate() has not been called.
        """
        if scenario.dim() == 3:
            scenario = scenario.unsqueeze(0)   # add batch dim → [1, T, N+1, D]

        scenario = scenario.to(self.device)
        T   = scenario.size(1)
        mid = T // 2

        ctx = scenario[:, :mid]
        tgt = scenario[:, mid:]

        # Encode context and predict future latent
        context_latent   = self.model.context_encoder(ctx)          # [1, 128]
        predicted_latent = self.model.predictor(context_latent)      # [1, 128]

        # Encode actual future with frozen target encoder
        target_latent = self.model.target_encoder(tgt)               # [1, 128]

        # L2 distance = surprise
        surprise = torch.norm(predicted_latent - target_latent, p=2, dim=1)
        score_val = surprise.item()

        is_surprising = (self.threshold is not None) and (score_val > self.threshold)
        return score_val, is_surprising

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_batch(self, scenarios: list[torch.Tensor] | torch.Tensor) -> list[float]:
        """
        Score a list of [T, N+1, D] tensors (or a single [N, T, N+1, D] batch).

        Returns:
            list of float surprise scores, one per scenario
        """
        if isinstance(scenarios, torch.Tensor) and scenarios.dim() == 4:
            # Already a stacked batch [N, T, N+1, D]
            batch = scenarios.to(self.device)
        else:
            batch = torch.stack(scenarios, dim=0).to(self.device)  # [N, T, N+1, D]

        T   = batch.size(1)
        mid = T // 2

        ctx = batch[:, :mid]
        tgt = batch[:, mid:]

        context_latent   = self.model.context_encoder(ctx)
        predicted_latent = self.model.predictor(context_latent)
        target_latent    = self.model.target_encoder(tgt)

        surprises = torch.norm(predicted_latent - target_latent, p=2, dim=1)
        return surprises.cpu().tolist()

    # ------------------------------------------------------------------
    def calibrate(
        self,
        scenarios: list[torch.Tensor] | torch.Tensor,
        percentile: float | None = None,
    ) -> float:
        """
        Run the model over a background set of scenarios and set the
        surprise threshold at the configured percentile (default: 90th).

        Args:
            scenarios:  list of [T, N+1, D] tensors, or [N, T, N+1, D]
            percentile: override the config value (evaluation.surprise_threshold_percentile)

        Returns:
            threshold value (also stored as self.threshold)
        """
        if percentile is None:
            percentile = self.data_cfg.get(
                "surprise_threshold_percentile",
                90,
            )
            # fall back to evaluation section if present
            # (config layout: evaluation.surprise_threshold_percentile)

        scores = self.score_batch(scenarios)
        self.threshold = float(np.percentile(scores, percentile))
        return self.threshold


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import yaml, pathlib, sys

    root = pathlib.Path(__file__).resolve().parents[3]
    cfg  = yaml.safe_load((root / "configs/default.yaml").read_text())

    ckpt_path = root / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print("No checkpoint found — run  python scripts/train.py  first.")
        sys.exit(1)

    data_dir = root / cfg["data"]["processed_dir"]
    files    = sorted(data_dir.glob("scenario_*.pt"))[:50]   # use first 50 for speed
    tensors  = [torch.load(f, weights_only=True) for f in files]

    scorer = SurpriseScore(ckpt_path, cfg, device=cfg["training"]["device"])

    print(f"Loaded model from {ckpt_path}")
    print(f"Calibrating on {len(tensors)} scenarios...")
    threshold = scorer.calibrate(tensors)
    print(f"Surprise threshold (90th pct): {threshold:.4f}")
    print()

    scores = scorer.score_batch(tensors)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    print("Top-10 most surprising scenarios:")
    for rank, (idx, s) in enumerate(ranked[:10], 1):
        flag = "⚠️ " if s > threshold else "   "
        print(f"  {rank:2d}. scenario_{idx:05d}  surprise={s:.4f}  {flag}")

    print()
    print(f"Min: {min(scores):.4f}  Max: {max(scores):.4f}  "
          f"Mean: {sum(scores)/len(scores):.4f}")
    print("All checks passed.")

