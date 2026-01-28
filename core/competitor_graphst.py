"""
GraphST competitor wrapper for spatial transcriptomics anomaly detection.

GraphST: Spatially Informed Graph Neural Network for Spatial Transcriptomics
Paper: https://doi.org/10.1038/s41467-023-36796-3

Strategy: GraphST embedding + Isolation Forest for anomaly detection.
This is NOT a pseudo-implementation - it uses the actual GraphST package.

No fallback: If GraphST fails, we raise an exception.
"""

import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.ensemble import IsolationForest
from scipy import sparse
from typing import Optional
import warnings
import torch


def compute_graphst_score(
    X: np.ndarray,
    coords: np.ndarray,
    n_top_genes: int = 3000,
    n_pcs: int = 50,
    n_epochs: int = 600,
    contamination: float = 0.1,
    random_state: int = 42,
    device: str = "cpu",  # Default to CPU for compatibility
) -> np.ndarray:
    """
    Compute anomaly score using GraphST embedding + Isolation Forest.

    This function:
    1. Creates AnnData object with spatial coordinates
    2. Runs GraphST to get spatial-aware embedding
    3. Applies Isolation Forest on the embedding

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    n_top_genes : int
        Number of highly variable genes to use
    n_pcs : int
        Number of PCA components
    n_epochs : int
        Training epochs.
        NOTE: GraphST's Python API does not expose n_epochs in train().
        This parameter is accepted for API consistency but GraphST uses
        its internal default (typically 1000 epochs). For fair comparison,
        we rely on GraphST's default training schedule.
    contamination : float
        Expected proportion of anomalies for IF
    random_state : int
        Random seed for reproducibility
    device : str
        Device for training ('cuda' or 'cpu')

    Returns
    -------
    score : np.ndarray
        Anomaly score [n_spots]. Higher = more anomalous.

    Raises
    ------
    ImportError
        If GraphST is not installed
    RuntimeError
        If GraphST training fails
    """
    from GraphST import GraphST

    # Check if CUDA is actually usable (not just detected)
    cuda_usable = False
    if device == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            torch.cuda.synchronize()
            _ = torch.tensor([1.0], device='cuda')
            torch.cuda.synchronize()
            cuda_usable = True
        except (AssertionError, RuntimeError):
            pass

    # Fall back to CPU if CUDA isn't usable
    if device == "cuda" and not cuda_usable:
        device = "cpu"

    # Set random seeds for reproducibility
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if cuda_usable:
        torch.cuda.manual_seed(random_state)

    # Convert to sparse if dense
    if not sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X)
    else:
        X_sparse = X

    # Create AnnData object
    adata = ad.AnnData(X_sparse)
    adata.obsm["spatial"] = coords.astype(np.float32)

    # Store raw counts
    adata.raw = adata.copy()

    # HVG selection on raw counts (seurat_v3 requires counts before log1p)
    n_hvg = min(n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_hvg,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Preprocessing (standard GraphST pipeline)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # PCA
    sc.pp.scale(adata)
    n_comp = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    sc.tl.pca(adata, n_comps=n_comp)

    # Create GraphST model
    model = GraphST.GraphST(adata, device=device, random_seed=random_state)

    # Train model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = model.train()

    # Check if embedding was created
    if "emb" not in adata.obsm:
        raise RuntimeError("GraphST training failed: no embedding created")

    # Get embedding
    embedding = adata.obsm["emb"]

    # Apply Isolation Forest on embedding
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )
    clf.fit(embedding)

    # Return anomaly score (higher = more anomalous)
    score = -clf.decision_function(embedding)

    return score


def compute_graphst_embedding(
    X: np.ndarray,
    coords: np.ndarray,
    n_top_genes: int = 3000,
    n_pcs: int = 50,
    n_epochs: int = 600,
    random_state: int = 42,
    device: str = "cpu",  # Default to CPU for compatibility
) -> np.ndarray:
    """
    Get GraphST embedding without anomaly detection.

    Useful for visualization or custom downstream analysis.

    Returns
    -------
    embedding : np.ndarray
        GraphST embedding [n_spots, latent_dim]
    """
    from GraphST import GraphST

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_state)

    if not sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X)
    else:
        X_sparse = X

    adata = ad.AnnData(X_sparse)
    adata.obsm["spatial"] = coords.astype(np.float32)
    adata.raw = adata.copy()

    # HVG selection on raw counts (seurat_v3 requires counts before log1p)
    n_hvg = min(n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_hvg,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.scale(adata)
    n_comp = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    sc.tl.pca(adata, n_comps=n_comp)

    model = GraphST.GraphST(adata, device=device, random_seed=random_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = model.train()

    if "emb" not in adata.obsm:
        raise RuntimeError("GraphST training failed")

    return adata.obsm["emb"]


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_spots = 500
    n_genes = 200

    X = np.random.exponential(1, (n_spots, n_genes))
    coords = np.random.rand(n_spots, 2) * 100

    print("Testing GraphST wrapper...")
    try:
        scores = compute_graphst_score(X, coords, device="cuda")
        print(f"Score shape: {scores.shape}")
        print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print("GraphST wrapper test PASSED")
    except Exception as e:
        print(f"GraphST wrapper test FAILED: {e}")
        raise
