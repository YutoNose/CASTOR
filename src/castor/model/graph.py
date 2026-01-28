"""Spatial graph construction and neighbor aggregation."""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_spatial_graph(coords: np.ndarray, k: int = 15) -> torch.Tensor:
    """Build k-NN spatial graph from coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates ``[n_spots, 2]``.
    k : int
        Number of neighbors (excluding self).

    Returns
    -------
    torch.Tensor
        Edge index ``[2, n_edges]``.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)

    row, col = [], []
    for i in range(len(coords)):
        for j in indices[i, 1:]:
            row.append(i)
            col.append(j)

    return torch.tensor([row, col], dtype=torch.long)


def aggregate_neighbors(h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Mean aggregation over neighbors.

    Parameters
    ----------
    h : torch.Tensor
        Node features ``[n_nodes, hidden_dim]``.
    edge_index : torch.Tensor
        Edge index ``[2, n_edges]``.

    Returns
    -------
    torch.Tensor
        Aggregated features ``[n_nodes, hidden_dim]``.
    """
    row, col = edge_index
    out = torch.zeros_like(h)
    out.index_add_(0, row, h[col])

    deg = torch.zeros(h.size(0), device=h.device)
    deg.index_add_(0, row, torch.ones(col.size(0), device=h.device))
    deg = deg.clamp(min=1).unsqueeze(1)

    return out / deg
