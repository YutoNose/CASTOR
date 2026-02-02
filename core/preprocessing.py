"""
Unified preprocessing pipeline for inverse prediction experiments.

This module consolidates all preprocessing code that was previously
duplicated across 11+ files (16+ copies).
"""

import numpy as np
import torch
from typing import Dict, Optional, Any

from .utils import (
    build_spatial_graph,
    normalize_expression,
    normalize_coordinates,
)


def prepare_data(
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
    device: str = "cuda",
    log_transform: bool = True,
    scale: bool = True,
) -> Dict[str, Any]:
    """
    Unified data preparation pipeline.

    This replaces the 16+ duplicated preprocessing blocks across the codebase.

    Parameters
    ----------
    X : np.ndarray
        Raw expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    k : int
        Number of neighbors for spatial graph
    device : str
        Device for tensors ('cuda' or 'cpu')
    log_transform : bool
        Apply log1p transformation
    scale : bool
        Standardize expression to zero mean and unit variance

    Returns
    -------
    data : dict
        Dictionary containing:
        - X_norm: Normalized expression [n_spots, n_genes]
        - coords_norm: Normalized coordinates [n_spots, 2]
        - x_tensor: Expression tensor on device
        - coords_tensor: Coordinates tensor on device
        - edge_index: Spatial graph edges on device
        - n_spots: Number of spots
        - n_genes: Number of genes

    Example
    -------
    >>> data = prepare_data(X, coords, k=15, device="cuda")
    >>> model = InversePredictionModel(data["n_genes"])
    >>> model = train_model(model, data["x_tensor"], data["coords_tensor"],
    ...                     data["edge_index"])
    """
    # Ensure numpy arrays
    X = np.asarray(X)
    coords = np.asarray(coords)

    # Normalize expression
    X_norm = normalize_expression(X, log_transform=log_transform, scale=scale)

    # Normalize coordinates to [0, 1]
    coords_norm = normalize_coordinates(coords)

    # Build spatial graph
    edge_index = build_spatial_graph(coords, k=k)

    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Convert to tensors
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
    adata,
    n_top_genes: int = 2000,
    k: int = 15,
    device: str = "cuda",
    skip_normalization: bool = False,
) -> Dict[str, Any]:
    """
    Prepare data from AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with spatial coordinates
    n_top_genes : int
        Number of highly variable genes to select
    k : int
        Number of neighbors for spatial graph
    device : str
        Device for tensors
    skip_normalization : bool
        If True, skip all normalization (for pre-normalized data).
        If False (default), apply normalization based on data state.

    Returns
    -------
    data : dict
        Same as prepare_data output
    """
    import scanpy as sc
    from scipy import sparse

    # Copy to avoid modifying original
    adata = adata.copy()

    # Determine if data needs normalization
    X_dense = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    max_val = X_dense.max()

    # Heuristics to detect data state:
    # - Raw counts: typically max > 100, integers
    # - Library-size normalized: max typically 1-100, floats
    # - Log-transformed: typically max < 15 for log1p(1e4 normalized)
    # - Z-scored: typically max < 10, can be negative
    min_val = X_dense.min()
    has_integers = np.allclose(X_dense[:100], np.round(X_dense[:100]))

    is_likely_raw = max_val > 100 or (max_val > 20 and has_integers)
    is_likely_log = max_val < 15 and min_val >= 0 and not has_integers
    is_likely_scaled = min_val < -0.1  # Z-scored data has negative values

    if skip_normalization:
        log_transform = False
        scale = False
    elif is_likely_scaled:
        # Already z-scored, skip all normalization
        log_transform = False
        scale = False
    elif is_likely_log:
        # Already log-transformed, just scale
        log_transform = False
        scale = True
    elif is_likely_raw:
        # Raw counts, apply full pipeline
        sc.pp.normalize_total(adata, target_sum=1e4)
        log_transform = True
        scale = True
    else:
        # Ambiguous state (e.g., library-size normalized, max 15-100)
        import warnings
        warnings.warn(
            f"Ambiguous data state (max={max_val:.1f}, min={min_val:.1f}). "
            "Assuming library-size normalized data; applying log1p + scaling. "
            "Set skip_normalization=True if data is already preprocessed.",
            UserWarning,
        )
        log_transform = True
        scale = True

    # Select HVGs (only if not already subset)
    if n_top_genes < adata.n_vars:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            adata = adata[:, adata.var["highly_variable"]]
        except Exception:
            # HVG selection may fail on some data; proceed without
            pass

    # Get expression matrix
    X = adata.X.toarray() if sparse.issparse(adata.X) else adata.X

    # Get coordinates
    if "spatial" in adata.obsm:
        coords = adata.obsm["spatial"]
    elif "X_spatial" in adata.obsm:
        coords = adata.obsm["X_spatial"]
    else:
        raise ValueError("No spatial coordinates found in adata.obsm")

    return prepare_data(X, coords, k=k, device=device, log_transform=log_transform, scale=scale)
