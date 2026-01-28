"""Expression normalization and data preparation pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from castor.model.graph import build_spatial_graph


def normalize_expression(
    X: np.ndarray,
    log_transform: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """Normalize an expression matrix.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix ``[n_spots, n_genes]``.
    log_transform : bool
        Apply ``log1p`` transformation.
    scale : bool
        Standardize to zero mean / unit variance per gene.

    Returns
    -------
    np.ndarray
        Normalized expression matrix.
    """
    X = np.asarray(X, dtype=np.float64)

    if log_transform:
        X = np.log1p(X)

    if scale:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.maximum(std, 1e-8)
        X = (X - mean) / std

    return X


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Normalize coordinates to [0, 1].

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates ``[n_spots, 2]``.

    Returns
    -------
    np.ndarray
        Normalized coordinates.
    """
    c_min = coords.min(axis=0)
    c_range = np.ptp(coords, axis=0)
    c_range = np.maximum(c_range, 1e-8)
    return (coords - c_min) / c_range


def prepare_data(
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
    device: str = "cuda",
    log_transform: bool = True,
    scale: bool = True,
) -> dict[str, Any]:
    """Unified data preparation pipeline.

    Parameters
    ----------
    X : np.ndarray
        Raw expression matrix ``[n_spots, n_genes]``.
    coords : np.ndarray
        Spatial coordinates ``[n_spots, 2]``.
    k : int
        Number of neighbors for the spatial graph.
    device : str
        ``"cuda"`` or ``"cpu"``.
    log_transform : bool
        Apply ``log1p``.
    scale : bool
        Standardize per gene.

    Returns
    -------
    dict
        Keys: ``X_norm``, ``coords_norm``, ``x_tensor``, ``coords_tensor``,
        ``edge_index``, ``n_spots``, ``n_genes``.
    """
    X = np.asarray(X)
    coords = np.asarray(coords)

    X_norm = normalize_expression(X, log_transform=log_transform, scale=scale)
    coords_norm = normalize_coordinates(coords)
    edge_index = build_spatial_graph(coords, k=k)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    x_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    coords_tensor = torch.tensor(coords_norm, dtype=torch.float32).to(device)
    edge_index = edge_index.to(device)

    return {
        "X_norm": X_norm,
        "coords_norm": coords_norm,
        "x_tensor": x_tensor,
        "coords_tensor": coords_tensor,
        "edge_index": edge_index,
        "n_spots": X.shape[0],
        "n_genes": X.shape[1],
    }


def prepare_from_anndata(
    adata: Any,
    n_top_genes: int = 2000,
    k: int = 15,
    device: str = "cuda",
) -> dict[str, Any]:
    """Prepare data from an AnnData object.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data with spatial coordinates in ``obsm``.
    n_top_genes : int
        Number of highly variable genes to select.
    k : int
        Spatial graph neighbors.
    device : str
        Device for tensors.

    Returns
    -------
    dict
        Same keys as :func:`prepare_data`.
    """
    import scanpy as sc
    from scipy import sparse

    adata = adata.copy()

    if adata.X.max() > 100:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if n_top_genes < adata.n_vars:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var["highly_variable"]]

    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)

    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
    elif "X_spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["X_spatial"])
    else:
        raise ValueError("No spatial coordinates found in adata.obsm")

    return prepare_data(X, coords, k=k, device=device, log_transform=True, scale=True)
