"""Spatial (contextual) anomaly baseline methods.

The primary contextual anomaly score in CASTOR comes from the inverse
prediction model's position-prediction error (``s_pos``).  The functions
here are supplementary baselines.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors


def compute_neighbor_diff(X: np.ndarray, coords: np.ndarray, k: int = 15) -> np.ndarray:
    """L2 distance between each spot and the mean of its spatial neighbors.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix ``[n_spots, n_genes]``.
    coords : np.ndarray
        Spatial coordinates ``[n_spots, 2]``.
    k : int
        Number of neighbors.

    Returns
    -------
    np.ndarray
        Score per spot ``[n_spots]``.
    """
    if sparse.issparse(X):
        X = X.toarray()  # type: ignore[union-attr]

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    indices = indices[:, 1:]

    neighbor_means = X[indices].mean(axis=1)
    return np.linalg.norm(X - neighbor_means, axis=1)


def compute_lisa(X: np.ndarray, coords: np.ndarray, k: int = 15) -> np.ndarray:
    """Local Moran's I (LISA) -- classical spatial autocorrelation score.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix ``[n_spots, n_genes]``.
    coords : np.ndarray
        Spatial coordinates ``[n_spots, 2]``.
    k : int
        Number of neighbors.

    Returns
    -------
    np.ndarray
        LISA score ``[n_spots]``.
    """
    if sparse.issparse(X):
        X = X.toarray()  # type: ignore[union-attr]

    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    indices = indices[:, 1:]

    X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    neighbor_means = X_scaled[indices].mean(axis=1)
    return np.abs(X_scaled - neighbor_means).mean(axis=1)


def compute_spotsweeper(X: np.ndarray, coords: np.ndarray, k: int = 15) -> np.ndarray:
    """SpotSweeper-like score combining QC metrics.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix ``[n_spots, n_genes]``.
    coords : np.ndarray
        Spatial coordinates ``[n_spots, 2]``.
    k : int
        Number of neighbors.

    Returns
    -------
    np.ndarray
        Combined score ``[n_spots]``.
    """
    if sparse.issparse(X):
        X = X.toarray()  # type: ignore[union-attr]

    def _robust_z(x: np.ndarray) -> np.ndarray:
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < 1e-10:
            mad = np.std(x)
        if mad < 1e-10:
            return np.zeros_like(x)
        return (x - med) / (mad * 1.4826)

    local_var = X.var(axis=1)
    neighbor_diff = compute_neighbor_diff(X, coords, k)
    lib_size = X.sum(axis=1)
    lib_z = np.abs((lib_size - lib_size.mean()) / (lib_size.std() + 1e-8))

    return _robust_z(local_var) + _robust_z(neighbor_diff) + lib_z
