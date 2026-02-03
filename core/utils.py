"""
Shared utility functions for inverse prediction experiments.

This module contains all shared utilities used across the codebase,
eliminating code duplication.
"""

import numpy as np
import torch
from scipy.spatial import Delaunay
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


def build_delaunay_graph(coords: np.ndarray) -> torch.Tensor:
    """
    Build Delaunay triangulation graph from coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]

    Returns
    -------
    edge_index : torch.Tensor
        Edge index [2, n_edges] for message passing
    """
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                edges.add((a, b))
                edges.add((b, a))

    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edges = sorted(edges)
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    return torch.tensor([row, col], dtype=torch.long)


def build_radius_graph(coords: np.ndarray, radius: float) -> torch.Tensor:
    """
    Build radius-based spatial graph from coordinates.

    Each spot is connected to all other spots within a given radius.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    radius : float
        Connection radius (in same units as coords)

    Returns
    -------
    edge_index : torch.Tensor
        Edge index [2, n_edges] for message passing
    """
    nn = NearestNeighbors(radius=radius)
    nn.fit(coords)
    distances, indices = nn.radius_neighbors(coords)

    row, col = [], []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:  # Skip self-loops
                row.append(i)
                col.append(j)

    if len(row) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

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
    assert edge_index.shape[0] == 2, (
        f"edge_index must be [2, n_edges], got {edge_index.shape}"
    )
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

    # 1.4826 is the MAD-to-SD conversion factor for normal distributions:
    # 1 / Φ^(-1)(3/4) ≈ 1.4826, where Φ is the standard normal CDF
    return (x - med) / (mad * 1.4826)


def normalize_coordinates(
    coords: np.ndarray,
    ref_min: Optional[np.ndarray] = None,
    ref_range: Optional[np.ndarray] = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Normalize coordinates to [0, 1] range.

    When ``ref_min`` and ``ref_range`` are provided (e.g. from training data),
    the same transform is applied to new coordinates, ensuring consistent
    normalization between train and test sets.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    ref_min : np.ndarray, optional
        Reference minimum (e.g. from training coordinates)
    ref_range : np.ndarray, optional
        Reference range (e.g. from training coordinates)

    Returns
    -------
    coords_norm : np.ndarray
        Normalized coordinates in [0, 1]
    c_min : np.ndarray
        Minimum used for normalization (pass as ``ref_min`` for test data)
    c_range : np.ndarray
        Range used for normalization (pass as ``ref_range`` for test data)
    """
    if ref_min is not None and ref_range is not None:
        c_min = ref_min
        c_range = ref_range
    else:
        c_min = coords.min(axis=0)
        c_max = coords.max(axis=0)
        c_range = c_max - c_min  # np.ptp is deprecated in NumPy 2.0
    c_range = np.maximum(c_range, 1e-8)  # Avoid division by zero
    return (coords - c_min) / c_range, c_min, c_range


def normalize_expression(
    X: np.ndarray,
    log_transform: bool = True,
    scale: bool = True,
    ref_mean: Optional[np.ndarray] = None,
    ref_std: Optional[np.ndarray] = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """
    Normalize expression matrix.

    When ``ref_mean`` and ``ref_std`` are provided (e.g. from training data),
    the same transform is applied to new data, ensuring consistent
    normalization between train and test sets.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    log_transform : bool
        Apply log1p transformation
    scale : bool
        Standardize to zero mean and unit variance per gene
    ref_mean : np.ndarray, optional
        Reference mean (e.g. from training expression after log1p)
    ref_std : np.ndarray, optional
        Reference std (e.g. from training expression after log1p)

    Returns
    -------
    X_norm : np.ndarray
        Normalized expression matrix
    mean : np.ndarray
        Mean used for normalization (pass as ``ref_mean`` for test data)
    std : np.ndarray
        Std used for normalization (pass as ``ref_std`` for test data)
    """
    X = np.asarray(X, dtype=np.float64)

    if log_transform:
        X = np.log1p(X)

    if scale:
        if ref_mean is not None and ref_std is not None:
            mean = ref_mean
            std = ref_std
        else:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
        std = np.maximum(std, 1e-8)  # Avoid division by zero
        X = (X - mean) / std
    else:
        mean = np.zeros(X.shape[1])
        std = np.ones(X.shape[1])

    return X, mean, std


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
