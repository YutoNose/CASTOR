"""Training and score computation for InversePredictionModel."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


def train_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    n_epochs: int = 100,
    lr: float = 1e-3,
    lambda_pos: float = 0.5,
    lambda_self: float = 1.0,
    verbose: bool = True,
) -> torch.nn.Module:
    """Train the inverse prediction model.

    Loss = lambda_self * MSE(x_self, x)
         + lambda_pos * MSE(pos_pred, coords)
         + MSE(x_neighbor, x)

    Parameters
    ----------
    model : nn.Module
        ``InversePredictionModel`` instance.
    x : torch.Tensor
        Expression tensor ``[n_spots, n_genes]``.
    coords : torch.Tensor
        Normalised coordinates ``[n_spots, 2]`` in [0, 1].
    edge_index : torch.Tensor
        Edge index for the spatial graph.
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.
    lambda_pos : float
        Weight for position prediction loss.
    lambda_self : float
        Weight for self-reconstruction loss.
    verbose : bool
        Log training progress.

    Returns
    -------
    nn.Module
        Trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(n_epochs):
        h, pos_pred, x_self, x_neighbor = model(x, edge_index)

        loss_self = F.mse_loss(x_self, x)
        loss_pos = F.mse_loss(pos_pred, coords)
        loss_neighbor = F.mse_loss(x_neighbor, x)

        loss = lambda_self * loss_self + lambda_pos * loss_pos + loss_neighbor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            logger.info(
                "Epoch %d: loss_self=%.4f, loss_pos=%.4f, loss_neighbor=%.4f",
                epoch + 1,
                loss_self.item(),
                loss_pos.item(),
                loss_neighbor.item(),
            )

    return model


def compute_scores(
    model: torch.nn.Module,
    x: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    if_contamination: float = 0.1,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute anomaly scores from a trained model.

    Returns
    -------
    dict
        Keys: ``s_pos``, ``s_self``, ``s_neighbor``, ``s_if``,
        ``embedding``, ``pos_pred``.
    """
    model.eval()
    with torch.no_grad():
        h, pos_pred, x_self, x_neighbor = model(x, edge_index)

        s_pos = ((pos_pred - coords) ** 2).sum(dim=1).cpu().numpy()
        s_self = ((x - x_self) ** 2).mean(dim=1).cpu().numpy()
        s_neighbor = ((x - x_neighbor) ** 2).mean(dim=1).cpu().numpy()

        h_np = h.cpu().numpy()
        pos_pred_np = pos_pred.cpu().numpy()

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
