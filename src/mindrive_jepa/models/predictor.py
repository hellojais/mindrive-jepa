import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    Predicts what the future latent will look like, given the context latent.

    Input:  context_latent [B, d_model]  — output of the ContextEncoder
    Output: predicted_latent [B, d_model] — the model's guess at the future latent

    The prediction_horizon (how many steps ahead) is encoded as a learnable embedding
    so the model can learn different prediction styles for "2 seconds ahead" vs "5 seconds ahead".

    Architecture: narrow transformer (2 layers vs 4 in the context encoder).
    Operating on a sequence of length 1 — the transformer's feedforward layers
    do the heavy lifting here, with the horizon embedding steering the prediction.
    """

    def __init__(self, data_config: dict, model_config: dict):
        super().__init__()

        d_model  = model_config['d_model']              # 128
        n_heads  = model_config['n_heads']              # 4
        n_layers = model_config['n_predictor_layers']   # 2
        dropout  = model_config['dropout']              # 0.1
        max_horizon = data_config['sequence_len']       # 50 — one embedding per possible horizon

        # Learnable horizon embedding: model learns different "prediction modes"
        # for short vs long prediction horizons
        self.horizon_embed = nn.Embedding(max_horizon, d_model)

        # Narrow transformer — 2 layers (half the depth of context encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,  # 512
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, context_latent: torch.Tensor, horizon: int = 20) -> torch.Tensor:
        """
        context_latent: [B, d_model]
        horizon:        how many timesteps ahead to predict (default: prediction_horizon=20)
        returns:        [B, d_model] predicted future latent
        """
        # Expand to a sequence of length 1 so the transformer can process it
        x = context_latent.unsqueeze(1)             # [B, 1, d_model]

        # Add the horizon embedding — tells the model HOW FAR ahead to predict
        horizon_idx = torch.tensor([horizon], device=context_latent.device)
        horizon_emb = self.horizon_embed(horizon_idx)  # [1, d_model]
        x = x + horizon_emb.unsqueeze(0)            # [B, 1, d_model]

        # Narrow transformer
        x = self.transformer(x)                     # [B, 1, d_model]

        # Remove the sequence dimension
        x = x.squeeze(1)                            # [B, d_model]

        # Output projection
        x = self.output_proj(x)                     # [B, d_model]

        return x
