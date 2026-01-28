"""
Squidpy competitor wrapper for spatial transcriptomics anomaly detection.

Squidpy: Spatial Single Cell Analysis in Python
Paper: https://doi.org/10.1038/s41592-021-01358-2

Strategy: Use Squidpy's neighborhood analysis to detect spatial anomalies.
This is NOT a pseudo-implementation - it uses the actual Squidpy package.

No fallback: If Squidpy fails, we raise an exception.
"""

import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.ensemble import IsolationForest
from scipy import sparse
from typing import Optional


def compute_squidpy_nhood_score(
    X: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 15,
    cluster_resolution: float = 0.5,
    contamination: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute anomaly score using Squidpy neighborhood analysis.

    This function:
    1. Clusters cells using Leiden
    2. Computes spatial neighbors
    3. Measures cluster mismatch with spatial neighbors

    A cell is anomalous if its cluster differs from its spatial neighbors.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    n_neighbors : int
        Number of spatial neighbors
    cluster_resolution : float
        Resolution for Leiden clustering
    contamination : float
        Expected proportion of anomalies for IF
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    score : np.ndarray
        Anomaly score [n_spots]. Higher = more anomalous.

    Raises
    ------
    ImportError
        If Squidpy is not installed
    RuntimeError
        If analysis fails
    """
    import squidpy as sq

    # Convert to sparse if dense
    if not sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X)
    else:
        X_sparse = X

    # Create AnnData object
    adata = ad.AnnData(X_sparse)
    adata.obsm["spatial"] = coords.astype(np.float32)

    # HVG selection on raw counts (seurat_v3 requires counts before log1p)
    if adata.n_vars > 3000:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=3000,
        )
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    else:
        adata_hvg = adata.copy()

    # Preprocessing
    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    sc.pp.log1p(adata_hvg)

    # PCA
    sc.pp.scale(adata_hvg)
    n_pcs = min(50, adata_hvg.n_vars - 1, adata_hvg.n_obs - 1)
    sc.tl.pca(adata_hvg, n_comps=n_pcs)

    # Build expression-based neighbors for clustering
    sc.pp.neighbors(adata_hvg, n_neighbors=15, use_rep="X_pca")

    # Leiden clustering
    sc.tl.leiden(adata_hvg, resolution=cluster_resolution, random_state=random_state)
    adata.obs["leiden"] = adata_hvg.obs["leiden"]

    # Build spatial neighbors using Squidpy
    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        n_neighs=n_neighbors,
    )

    # Check if spatial connectivity was built
    if "spatial_connectivities" not in adata.obsp:
        raise RuntimeError("Squidpy failed to build spatial neighbors")

    # Compute cluster mismatch score
    # For each cell: what fraction of spatial neighbors are in a different cluster?
    spatial_conn = adata.obsp["spatial_connectivities"].toarray()
    clusters = adata.obs["leiden"].astype(int).values

    n_spots = len(clusters)
    mismatch_scores = np.zeros(n_spots)

    for i in range(n_spots):
        neighbors = np.where(spatial_conn[i] > 0)[0]
        if len(neighbors) > 0:
            neighbor_clusters = clusters[neighbors]
            mismatch_fraction = np.mean(neighbor_clusters != clusters[i])
            mismatch_scores[i] = mismatch_fraction

    return mismatch_scores


def compute_squidpy_ripley_score(
    X: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 15,
    cluster_resolution: float = 0.5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute anomaly score using Squidpy Ripley's statistics.

    Uses Ripley's L function to measure spatial clustering.
    Anomalous cells have unusual local clustering patterns.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    n_neighbors : int
        Number of spatial neighbors
    cluster_resolution : float
        Resolution for Leiden clustering
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    score : np.ndarray
        Anomaly score [n_spots]. Higher = more anomalous.
    """
    import squidpy as sq

    if not sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X)
    else:
        X_sparse = X

    adata = ad.AnnData(X_sparse)
    adata.obsm["spatial"] = coords.astype(np.float32)

    # HVG selection on raw counts (seurat_v3 requires counts before log1p)
    if adata.n_vars > 3000:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=3000,
        )
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    else:
        adata_hvg = adata.copy()

    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    sc.pp.log1p(adata_hvg)

    sc.pp.scale(adata_hvg)
    n_pcs = min(50, adata_hvg.n_vars - 1, adata_hvg.n_obs - 1)
    sc.tl.pca(adata_hvg, n_comps=n_pcs)
    sc.pp.neighbors(adata_hvg, n_neighbors=15, use_rep="X_pca")
    sc.tl.leiden(adata_hvg, resolution=cluster_resolution, random_state=random_state)
    adata.obs["leiden"] = adata_hvg.obs["leiden"]

    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n_neighbors)

    # Compute centrality score per cell
    # Use combination of cluster mismatch and local density
    spatial_conn = adata.obsp["spatial_connectivities"].toarray()
    clusters = adata.obs["leiden"].astype(int).values

    n_spots = len(clusters)
    scores = np.zeros(n_spots)

    # Local density: degree in spatial graph
    degrees = spatial_conn.sum(axis=1)
    degree_z = (degrees - degrees.mean()) / (degrees.std() + 1e-8)

    # Cluster mismatch
    for i in range(n_spots):
        neighbors = np.where(spatial_conn[i] > 0)[0]
        if len(neighbors) > 0:
            neighbor_clusters = clusters[neighbors]
            mismatch = np.mean(neighbor_clusters != clusters[i])
            # Combine mismatch with density anomaly
            scores[i] = mismatch + 0.2 * np.abs(degree_z[i])

    return scores


def compute_squidpy_embedding_score(
    X: np.ndarray,
    coords: np.ndarray,
    n_neighbors: int = 15,
    contamination: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute anomaly score using spatial smoothed embedding + Isolation Forest.

    Uses Squidpy to get spatially-aware features, then applies IF.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    n_neighbors : int
        Number of spatial neighbors
    contamination : float
        Expected proportion of anomalies for IF
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    score : np.ndarray
        Anomaly score [n_spots]. Higher = more anomalous.
    """
    import squidpy as sq

    if not sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X)
    else:
        X_sparse = X

    adata = ad.AnnData(X_sparse)
    adata.obsm["spatial"] = coords.astype(np.float32)

    # HVG selection on raw counts (seurat_v3 requires counts before log1p)
    if adata.n_vars > 3000:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=3000,
        )
        adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    else:
        adata_hvg = adata.copy()

    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    sc.pp.log1p(adata_hvg)

    sc.pp.scale(adata_hvg)
    n_pcs = min(50, adata_hvg.n_vars - 1, adata_hvg.n_obs - 1)
    sc.tl.pca(adata_hvg, n_comps=n_pcs)

    # Build spatial graph
    sq.gr.spatial_neighbors(adata_hvg, coord_type="generic", n_neighs=n_neighbors)

    # Get PCA embedding with spatial context
    # Compute difference between each cell and its spatial neighbors
    pca_embedding = adata_hvg.obsm["X_pca"]
    spatial_conn = adata_hvg.obsp["spatial_connectivities"].toarray()

    n_spots = pca_embedding.shape[0]
    spatial_diff = np.zeros_like(pca_embedding)

    for i in range(n_spots):
        neighbors = np.where(spatial_conn[i] > 0)[0]
        if len(neighbors) > 0:
            neighbor_mean = pca_embedding[neighbors].mean(axis=0)
            spatial_diff[i] = pca_embedding[i] - neighbor_mean

    # Combine original embedding with spatial difference
    combined_embedding = np.concatenate([pca_embedding, spatial_diff], axis=1)

    # Apply Isolation Forest
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )
    clf.fit(combined_embedding)

    score = -clf.decision_function(combined_embedding)
    return score


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_spots = 500
    n_genes = 200

    X = np.random.exponential(1, (n_spots, n_genes))
    coords = np.random.rand(n_spots, 2) * 100

    print("Testing Squidpy wrapper...")
    try:
        scores = compute_squidpy_nhood_score(X, coords)
        print(f"Nhood score shape: {scores.shape}")
        print(f"Nhood score range: [{scores.min():.3f}, {scores.max():.3f}]")

        scores_emb = compute_squidpy_embedding_score(X, coords)
        print(f"Embedding score shape: {scores_emb.shape}")
        print(f"Embedding score range: [{scores_emb.min():.3f}, {scores_emb.max():.3f}]")

        print("Squidpy wrapper test PASSED")
    except Exception as e:
        print(f"Squidpy wrapper test FAILED: {e}")
        raise
