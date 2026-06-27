import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (no learnable params)."""

    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return self.dropout(x + self.pe[:, :x.size(1), :])


class ContextEncoder(nn.Module):
    """
    Compresses a scene window into a fixed-size latent vector.

    Input:  [B, T, N+1, D]  — T timesteps of agent features (typically T = sequence_len // 2)
    Output: [B, d_model]    — latent summarising the scene dynamics

    Steps:
      1. Flatten per-timestep features: [B, T, (N+1)*D]
      2. Linear projection → [B, T, d_model]
      3. Sinusoidal positional encoding
      4. TransformerEncoder (n_encoder_layers, n_heads)
      5. Mean pool over T → [B, d_model]
      6. Output projection → [B, d_model]
    """

    def __init__(self, data_config: dict, model_config: dict):
        super().__init__()

        n_agents  = data_config['max_agents'] + 1   # 21 (ego + 20 others)
        feat_dim  = data_config['agent_feat_dim']   # 6
        input_dim = n_agents * feat_dim             # 126

        d_model  = model_config['d_model']          # 128
        n_heads  = model_config['n_heads']          # 4
        n_layers = model_config['n_encoder_layers'] # 4
        dropout  = model_config['dropout']          # 0.1
        max_len  = data_config['sequence_len']      # 50

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,  # 512
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, N+1, D]
        returns: [B, d_model]
        """
        B, T, N1, D = x.shape

        # Step 1: flatten agents+features per timestep
        x = x.reshape(B, T, N1 * D)          # [B, T, 126]

        # Step 2: project to d_model
        x = self.input_proj(x)                # [B, T, 128]

        # Step 3: positional encoding
        x = self.pos_enc(x)                   # [B, T, 128]

        # Step 4: transformer encoder
        x = self.transformer(x)               # [B, T, 128]

        # Step 5: mean pool over time
        x = x.mean(dim=1)                     # [B, 128]

        # Step 6: output projection
        x = self.output_proj(x)               # [B, 128]

        return x
