"""
mindrive_jepa.models
====================
MinDriveJEPA: full JEPA world model assembled from four components.

    context_window  [B, T_ctx, N+1, D]  ──► context_encoder ──► context_latent   [B, 128]
                                                                       │
                                                                   predictor  (horizon h)
                                                                       │
                                                               predicted_latent   [B, 128]
                                                                       │
                                               ┌───────────────────────┴──────────────┐
                                       jepa_loss (MSE)            position_decoder
                                               │                       │
    target_window  [B, T_tgt, N+1, D]  ──► target_encoder ──► target_latent  predicted_positions
                                         (EMA, no grad)                │         [B, 21, 2]
                                                               position_loss (MSE vs actual)

    total_loss = jepa_loss + lambda_aux * position_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mindrive_jepa.models.context_encoder import ContextEncoder
from mindrive_jepa.models.target_encoder import TargetEncoder
from mindrive_jepa.models.predictor import Predictor
from mindrive_jepa.models.position_decoder import PositionDecoder


class MinDriveJEPA(nn.Module):
    """Full JEPA latent world model for autonomous driving scenes."""

    def __init__(self, data_config: dict, model_config: dict):
        super().__init__()
        self.context_encoder  = ContextEncoder(data_config, model_config)
        self.target_encoder   = TargetEncoder(self.context_encoder)
        self.predictor        = Predictor(data_config, model_config)
        self.position_decoder = PositionDecoder(
            d_model    = model_config['d_model'],
            max_agents = data_config['max_agents'] + 1,   # +1 for ego
        )
        self.lambda_aux = model_config['lambda_aux']

    # ------------------------------------------------------------------
    def forward(
        self,
        context_window: torch.Tensor,   # [B, T_ctx, N+1, D]
        target_window:  torch.Tensor,   # [B, T_tgt, N+1, D]
        horizon:        int = 20,
    ) -> dict:
        """
        Returns a dict with:
            loss               – total scalar loss (jepa + lambda*position)
            jepa_loss          – latent MSE
            position_loss      – position MSE
            context_latent     – [B, d_model]
            predicted_latent   – [B, d_model]
            target_latent      – [B, d_model]
            predicted_positions– [B, max_agents+1, 2]
        """
        # 1. Encode context window
        context_latent = self.context_encoder(context_window)          # [B, 128]

        # 2. Predict future latent
        predicted_latent = self.predictor(context_latent, horizon)     # [B, 128]

        # 3. Encode target window — target encoder is frozen (EMA copy)
        with torch.no_grad():
            target_latent = self.target_encoder(target_window)         # [B, 128]

        # 4. JEPA loss: predicted latent vs actual future latent
        jepa_loss = F.mse_loss(predicted_latent, target_latent)

        # 5. Decode predicted positions from predicted latent
        predicted_positions = self.position_decoder(predicted_latent)  # [B, 21, 2]

        # 6. Actual positions at the last frame of the target window (x, y only)
        actual_positions = target_window[:, -1, :, :2].detach()        # [B, 21, 2]

        # 7. Auxiliary position loss
        position_loss = F.mse_loss(predicted_positions, actual_positions)

        # 8. Combined loss
        total_loss = jepa_loss + self.lambda_aux * position_loss

        return {
            "loss":                total_loss,
            "jepa_loss":           jepa_loss,
            "position_loss":       position_loss,
            "context_latent":      context_latent,
            "predicted_latent":    predicted_latent,
            "target_latent":       target_latent,
            "predicted_positions": predicted_positions,
        }

    # ------------------------------------------------------------------
    def update_target_encoder(self, decay: float = 0.996) -> None:
        """Call once per training step after optimizer.step()."""
        self.target_encoder.update_ema(self.context_encoder, decay)


# ----------------------------------------------------------------------
# Sanity check — run with:  python src/mindrive_jepa/models/__init__.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import yaml, sys, pathlib

    root = pathlib.Path(__file__).resolve().parents[3]   # project root
    cfg  = yaml.safe_load((root / "configs/default.yaml").read_text())

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]

    # Build the full model
    model = MinDriveJEPA(data_cfg, model_cfg)
    model.eval()

    # Fake data that matches the real tensor shape [B, T, N+1, D]
    B      = 4
    T_ctx  = 25    # first 25 of 50 frames  (context)
    T_tgt  = 25    # last  25 of 50 frames  (target)
    N1     = data_cfg["max_agents"] + 1    # 21
    D      = data_cfg["agent_feat_dim"]    # 6

    ctx = torch.randn(B, T_ctx, N1, D)
    tgt = torch.randn(B, T_tgt, N1, D)

    out = model(ctx, tgt, horizon=model_cfg.get("prediction_horizon", 20))

    print("=== MinDriveJEPA sanity check ===")
    print(f"context_window shape:       {tuple(ctx.shape)}")
    print(f"target_window  shape:       {tuple(tgt.shape)}")
    print(f"context_latent shape:       {tuple(out['context_latent'].shape)}       (expected [4, 128])")
    print(f"predicted_latent shape:     {tuple(out['predicted_latent'].shape)}       (expected [4, 128])")
    print(f"target_latent shape:        {tuple(out['target_latent'].shape)}       (expected [4, 128])")
    print(f"predicted_positions shape:  {tuple(out['predicted_positions'].shape)}    (expected [4, 21, 2])")
    print(f"total loss:                 {out['loss'].item():.6f}")
    print(f"  jepa_loss:                {out['jepa_loss'].item():.6f}")
    print(f"  position_loss:            {out['position_loss'].item():.6f}")
    print(f"  lambda_aux:               {model.lambda_aux}")
    print()

    # Verify no NaN anywhere
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any(), f"NaN in {k}!"

    # Verify target encoder has NO gradients
    target_grads = [p.requires_grad for p in model.target_encoder.parameters()]
    assert not any(target_grads), "Target encoder should have requires_grad=False!"
    print("Target encoder: all parameters frozen ✓")

    # Parameter counts
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print()
    print("All checks passed.")

