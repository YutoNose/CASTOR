"""Inverse Prediction Model for Spatial Anomaly Detection.

Key insight
-----------
- Forward (neighbors -> expression): "Does this expression fit here?"
- Inverse (expression -> position): "Where does this expression belong?"

For **contextual** anomalies (ectopic):
  Both forward and inverse errors are high -- the model predicts the
  expression belongs at its *donor* location, not its current position.

For **intrinsic** anomalies:
  Forward error is high but inverse error is only moderate -- the
  expression doesn't map to any specific location.

The inverse prediction therefore provides *selective* detection of
contextual anomalies.
"""

import torch
import torch.nn as nn

from castor.model.graph import aggregate_neighbors


class InversePredictionModel(nn.Module):
    """Multi-task model with inverse spatial prediction.

    Architecture
    ------------
    1. Encoder: expression -> embedding
    2. Position predictor: embedding -> (x, y) **(contextual axis)**
    3. Self decoder: embedding -> reconstructed expression (auxiliary)
    4. Neighbor predictor: aggregated neighbor embedding -> expression (forward baseline)

    Parameters
    ----------
    in_dim : int
        Number of input features (genes).
    hid_dim : int
        Hidden dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(self, in_dim: int, hid_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # LayerNorm is used instead of BatchNorm1d because the model trains
        # on the full dataset as a single batch (no mini-batching), making
        # BatchNorm's running statistics unreliable during eval mode.
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
        )

        self.pos_predictor = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 2),
        )

        self.self_decoder = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim),
        )

        self.expr_predictor = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns ``(h, pos_pred, x_self, x_neighbor)``.
        """
        h = self.encoder(x)
        pos_pred = self.pos_predictor(h)
        x_self = self.self_decoder(h)
        h_agg = aggregate_neighbors(h, edge_index)
        x_neighbor = self.expr_predictor(h_agg)
        return h, pos_pred, x_self, x_neighbor
