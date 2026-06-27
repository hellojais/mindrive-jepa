import torch
import torch.nn as nn


class PositionDecoder(nn.Module):
    """
    Decodes a predicted latent vector into predicted (x, y) positions for all agents.

    Added by SPEC_PATCH to enable bird's-eye visualization (Phase 5).
    The JEPA loss operates in latent space — this decoder provides an auxiliary
    supervision signal that also makes predictions interpretable as map positions.

    Architecture: 2-layer MLP (no transformer needed — latent → positions is simple)
      Linear(d_model, 256) → ReLU → Dropout(0.1) → Linear(256, max_agents * 2)

    Trained with auxiliary loss weighted by lambda_aux=0.1 (see MinDriveJEPA.forward).
    The JEPA latent loss remains the primary objective.

    Input:  predicted_latent [B, d_model]
    Output: predicted_positions [B, max_agents, 2]  — (x, y) per agent, normalized coords
    """

    def __init__(self, d_model: int, max_agents: int):
        """
        d_model:    latent dimension (128)
        max_agents: total agents including ego (21 = data.max_agents + 1)
        """
        super().__init__()

        self.max_agents = max_agents
        output_dim = max_agents * 2  # x,y for each agent = 42

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: [B, d_model]
        returns: [B, max_agents, 2]  — predicted (x, y) for all agents in normalized coords
        """
        B = latent.size(0)
        out = self.mlp(latent)                      # [B, max_agents * 2]
        return out.reshape(B, self.max_agents, 2)   # [B, 21, 2]
