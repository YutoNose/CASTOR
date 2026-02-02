"""
Shared utility functions for inverse prediction experiments.

This module contains all shared utilities used across the codebase,
eliminating code duplication.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from typing import Optional


def build_spatial_graph(coords: np.ndarray, k: int = 15) -> torch.Tensor:
    """
    Build k-NN spatial graph from coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    k : int
        Number of neighbors (excluding self)

    Returns
    -------
    edge_index : torch.Tensor
        Edge index [2, n_edges] for message passing
    """
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    row, col = [], []
    for i in range(len(coords)):
        for j in indices[i, 1:]:  # Skip self (index 0)
            row.append(i)
            col.append(j)

    return torch.tensor([row, col], dtype=torch.long)


def aggregate_neighbors(
    h: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Mean aggregation over neighbors.

    This is the SINGLE implementation used throughout the codebase
    to avoid inconsistencies.

    Parameters
    ----------
    h : torch.Tensor
        Node features [n_nodes, hidden_dim]
    edge_index : torch.Tensor
        Edge index [2, n_edges]

    Returns
    -------
    h_agg : torch.Tensor
        Aggregated features [n_nodes, hidden_dim]
    """
    row, col = edge_index
    out = torch.zeros_like(h)
    out.index_add_(0, row, h[col])

    # Compute degree for normalization
    deg = torch.zeros(h.size(0), device=h.device)
    deg.index_add_(0, row, torch.ones(col.size(0), device=h.device))
    deg = deg.clamp(min=1).unsqueeze(1)

    return out / deg


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score using median and MAD.

    More robust to outliers than standard z-score.

    Parameters
    ----------
    x : np.ndarray
        Input array

    Returns
    -------
    z : np.ndarray
        Robust z-scores
    """
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))

    # MAD to standard deviation conversion factor
    if mad < 1e-10:
        mad = np.std(x)
    if mad < 1e-10:
        return np.zeros_like(x)

    return (x - med) / (mad * 1.4826)


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """
    Normalize coordinates to [0, 1] range.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]

    Returns
    -------
    coords_norm : np.ndarray
        Normalized coordinates in [0, 1]
    """
    c_min = coords.min(axis=0)
    c_max = coords.max(axis=0)
    c_range = c_max - c_min  # np.ptp is deprecated in NumPy 2.0
    c_range = np.maximum(c_range, 1e-8)  # Avoid division by zero
    return (coords - c_min) / c_range


def normalize_expression(
    X: np.ndarray,
    log_transform: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """
    Normalize expression matrix.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    log_transform : bool
        Apply log1p transformation
    scale : bool
        Standardize to zero mean and unit variance per gene

    Returns
    -------
    X_norm : np.ndarray
        Normalized expression matrix
    """
    X = np.asarray(X, dtype=np.float64)

    if log_transform:
        X = np.log1p(X)

    if scale:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.maximum(std, 1e-8)  # Avoid division by zero
        X = (X - mean) / std

    return X


def compute_neighbor_means(
    X: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """
    Compute mean expression of neighbors for each spot.

    This is the CORRECT vectorized implementation.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    indices : np.ndarray
        Neighbor indices [n_spots, k] (excluding self)

    Returns
    -------
    neighbor_means : np.ndarray
        Mean neighbor expression [n_spots, n_genes]
    """
    # X[indices] has shape [n_spots, k, n_genes]
    # Mean over axis=1 (the k neighbors) gives [n_spots, n_genes]
    return X[indices].mean(axis=1)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
