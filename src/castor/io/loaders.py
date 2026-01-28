"""Data loading utilities for .h5ad and CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_anndata(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load expression and coordinates from an ``.h5ad`` file.

    Coordinates are read from ``adata.obsm["spatial"]`` or
    ``adata.obsm["X_spatial"]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        ``(X, coords, gene_names)``
    """
    import scanpy as sc
    from scipy import sparse

    adata = sc.read_h5ad(str(path))
    X = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    gene_names = adata.var_names.tolist()

    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
    elif "X_spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["X_spatial"])
    else:
        raise ValueError(
            "No spatial coordinates found in adata.obsm. Expected 'spatial' or 'X_spatial'."
        )

    if coords.shape[1] > 2:
        coords = coords[:, :2]

    return X, coords, gene_names


def load_csv(
    expression_path: str | Path,
    coords_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load expression and coordinates from CSV files.

    The expression CSV is expected to have spots as rows and genes as
    columns (with a row index).  The coordinates CSV must have the same
    row index and exactly two value columns.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        ``(X, coords, gene_names)``
    """
    expr_df = pd.read_csv(str(expression_path), index_col=0)
    coords_df = pd.read_csv(str(coords_path), index_col=0)

    if expr_df.shape[0] != coords_df.shape[0]:
        raise ValueError(
            f"Shape mismatch: expression has {expr_df.shape[0]} spots "
            f"but coordinates have {coords_df.shape[0]} spots"
        )

    return (
        expr_df.values.astype(np.float64),
        coords_df.values.astype(np.float64),
        expr_df.columns.tolist(),
    )


def auto_load(
    input_path: str | Path,
    coords_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Auto-detect format and load data.

    Parameters
    ----------
    input_path : str | Path
        Path to ``.h5ad`` or expression ``.csv``.
    coords_path : str | Path | None
        Path to coordinates CSV (required when *input_path* is CSV).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        ``(X, coords, gene_names)``
    """
    input_path = Path(input_path)

    if input_path.suffix == ".h5ad":
        return load_anndata(input_path)

    if input_path.suffix == ".csv":
        if coords_path is None:
            raise ValueError("--coords is required when the input is a CSV file.")
        return load_csv(input_path, coords_path)

    raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .h5ad or .csv.")
