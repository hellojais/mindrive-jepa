import copy

import torch
import torch.nn as nn

from mindrive_jepa.models.context_encoder import ContextEncoder


class TargetEncoder(nn.Module):
    """
    EMA copy of the ContextEncoder. Never updated by backpropagation.

    Encodes the TARGET (future) window into a latent vector.
    Its weights are initialised as a deep copy of the ContextEncoder and then
    slowly updated via EMA after each training step.

    This slow-moving target prevents representation collapse — the two encoders
    cannot co-adapt to a trivial constant-output solution because the target
    always lags behind by ~250 steps (at decay=0.996).
    """

    def __init__(self, context_encoder: ContextEncoder):
        super().__init__()

        # Deep copy: same architecture, same initial weights as context encoder
        self.encoder = copy.deepcopy(context_encoder)

        # Freeze all parameters — backprop never touches this encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Identical forward pass to ContextEncoder.
        x: [B, T, N+1, D]
        returns: [B, d_model]
        """
        return self.encoder(x)

    @torch.no_grad()
    def update_ema(self, context_encoder: ContextEncoder, decay: float = 0.996):
        """
        Exponential moving average update. Call once per training step, after optimizer.step().

        For each parameter pair:
            target = decay * target + (1 - decay) * context

        At decay=0.996: each step moves 0.4% toward the context encoder.
        The target encoder lags ~250 steps behind — stable enough to prevent collapse.
        """
        for target_param, context_param in zip(
            self.encoder.parameters(), context_encoder.parameters()
        ):
            target_param.data.mul_(decay).add_(context_param.data, alpha=1.0 - decay)
