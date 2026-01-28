"""
Inverse Prediction Model for Spatial Anomaly Detection.

Key insight:
- Forward (neighbors -> expression): measures "Does this expression fit here?"
- Inverse (expression -> position): measures "Where does this expression belong?"

For Ectopic anomalies:
- Forward: High error (doesn't fit here)
- Inverse: High error (predicts wrong location, because expression belongs elsewhere)

For Intrinsic anomalies:
- Forward: High error (doesn't fit here)
- Inverse: Low-to-medium error (expression doesn't belong anywhere specific)

The inverse prediction provides SELECTIVE detection of Ectopic anomalies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional

from .utils import aggregate_neighbors


class InversePredictionModel(nn.Module):
    """
    Inverse Prediction Model with Multi-task Learning.

    Architecture:
    1. Encoder: Expression -> Embedding
    2. Position Predictor: Embedding -> Predicted Position (for Ectopic)
    3. Self Decoder: Embedding -> Reconstructed Expression (auxiliary)
    4. Neighbor Predictor: Neighbor Embedding -> Predicted Expression (forward baseline)

    Scores:
    - s_pos: Position prediction error (Ectopic-specific)
    - s_self: Self-reconstruction error (auxiliary)
    - s_neighbor: Neighbor reconstruction error (forward baseline)
    """

    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 64,
        dropout: float = 0.3,
    ):
        """
        Initialize the model.

        Parameters
        ----------
        in_dim : int
            Number of input features (genes)
        hid_dim : int
            Hidden dimension
        dropout : float
            Dropout rate
        """
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # Encoder: Expression -> Embedding
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        )

        # Position predictor: Embedding -> (x, y)
        # This is the KEY component for Ectopic detection
        self.pos_predictor = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 2),
        )

        # Self decoder: Embedding -> Expression (auxiliary task)
        self.self_decoder = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim),
        )

        # Neighbor expression predictor: Aggregated embedding -> Expression
        # This provides the forward baseline
        self.expr_predictor = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, in_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Expression matrix [n_spots, n_genes]
        edge_index : torch.Tensor
            Edge index [2, n_edges]

        Returns
        -------
        h : torch.Tensor
            Embedding [n_spots, hid_dim]
        pos_pred : torch.Tensor
            Predicted position [n_spots, 2]
        x_self : torch.Tensor
            Self-reconstructed expression [n_spots, n_genes]
        x_neighbor : torch.Tensor
            Predicted expression from neighbors [n_spots, n_genes]
        """
        # Encode expression to embedding
        h = self.encoder(x)

        # Inverse prediction: expression -> position (KEY for Ectopic)
        pos_pred = self.pos_predictor(h)

        # Self reconstruction: embedding -> expression (auxiliary)
        x_self = self.self_decoder(h)

        # Forward prediction: neighbors -> expression (baseline)
        h_agg = aggregate_neighbors(h, edge_index)
        x_neighbor = self.expr_predictor(h_agg)

        return h, pos_pred, x_self, x_neighbor


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-3,
    lambda_pos: float = 0.5,
    lambda_self: float = 1.0,
    verbose: bool = True,
) -> nn.Module:
    """
    Train the inverse prediction model.

    Loss = lambda_self * MSE(x_self, x) + lambda_pos * MSE(pos_pred, coords) + MSE(x_neighbor, x)

    Parameters
    ----------
    model : nn.Module
        InversePredictionModel instance
    x : torch.Tensor
        Expression tensor [n_spots, n_genes]
    coords : torch.Tensor
        Normalized coordinates [n_spots, 2] in [0, 1]
    edge_index : torch.Tensor
        Edge index for spatial graph
    n_epochs : int
        Number of training epochs
    lr : float
        Learning rate
    lambda_pos : float
        Weight for position prediction loss
    lambda_self : float
        Weight for self-reconstruction loss
    verbose : bool
        Print training progress

    Returns
    -------
    model : nn.Module
        Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(n_epochs):
        h, pos_pred, x_self, x_neighbor = model(x, edge_index)

        # Self-reconstruction loss
        loss_self = F.mse_loss(x_self, x)

        # Position prediction loss (KEY for Ectopic detection)
        loss_pos = F.mse_loss(pos_pred, coords)

        # Neighbor reconstruction loss (forward baseline)
        loss_neighbor = F.mse_loss(x_neighbor, x)

        # Combined loss
        loss = lambda_self * loss_self + lambda_pos * loss_pos + loss_neighbor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(
                f"  Epoch {epoch + 1}: "
                f"loss_self={loss_self.item():.4f}, "
                f"loss_pos={loss_pos.item():.4f}, "
                f"loss_neighbor={loss_neighbor.item():.4f}"
            )

    return model


def compute_scores(
    model: nn.Module,
    x: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    if_contamination: float = 0.1,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Compute anomaly scores from trained model.

    Parameters
    ----------
    model : nn.Module
        Trained InversePredictionModel
    x : torch.Tensor
        Expression tensor
    coords : torch.Tensor
        Normalized coordinates
    edge_index : torch.Tensor
        Edge index
    if_contamination : float
        Contamination parameter for Isolation Forest
    random_state : int
        Random state for reproducibility

    Returns
    -------
    scores : dict
        Dictionary with keys:
        - s_pos: Position prediction error (Ectopic-specific) [n_spots]
        - s_self: Self-reconstruction error [n_spots]
        - s_neighbor: Neighbor reconstruction error [n_spots]
        - s_if: Isolation Forest score on embedding [n_spots]
        - embedding: Learned embedding [n_spots, hid_dim]
        - pos_pred: Predicted positions [n_spots, 2]
    """
    model.eval()
    with torch.no_grad():
        h, pos_pred, x_self, x_neighbor = model(x, edge_index)

        # Ectopic score: position prediction error (squared L2 distance)
        s_pos = ((pos_pred - coords) ** 2).sum(dim=1).cpu().numpy()

        # Self-reconstruction error (mean squared error per spot)
        s_self = ((x - x_self) ** 2).mean(dim=1).cpu().numpy()

        # Neighbor reconstruction error (forward baseline)
        s_neighbor = ((x - x_neighbor) ** 2).mean(dim=1).cpu().numpy()

        # Embedding and predicted positions for analysis
        h_np = h.cpu().numpy()
        pos_pred_np = pos_pred.cpu().numpy()

    # Isolation Forest on embedding (additional intrinsic detection)
    iso = IsolationForest(
        n_estimators=100,
        contamination=if_contamination,
        random_state=random_state,
    )
    iso.fit(h_np)
    s_if = -iso.decision_function(h_np)

    return {
        "s_pos": s_pos,
        "s_self": s_self,
        "s_neighbor": s_neighbor,
        "s_if": s_if,
        "embedding": h_np,
        "pos_pred": pos_pred_np,
    }
