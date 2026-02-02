"""
STAGATE competitor wrapper for spatial transcriptomics anomaly detection.

STAGATE: Spatial Graph Attention Autoencoder for Spatial Transcriptomics
Paper: https://doi.org/10.1038/s41467-022-34879-1

Strategy: STAGATE embedding + Isolation Forest for anomaly detection.
This is NOT a pseudo-implementation - it uses the actual STAGATE package.

No fallback: If STAGATE fails, we raise an exception.
"""

# STAGATE requires TensorFlow 1.x compatibility mode
# Force CPU due to GPU compatibility issues with newer GPUs (e.g., RTX 5090)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use -1 instead of empty string
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Programmatically hide all GPUs - more reliable than env var alone
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass  # May fail if TF already initialized, but env var should cover it

# Disable eager execution before creating any TF objects
try:
    tf.compat.v1.disable_eager_execution()
except AttributeError:
    pass  # Already disabled

# Reset TF graph state for multiple STAGATE runs
def _reset_tf_graph():
    """Reset TensorFlow graph state for clean STAGATE runs."""
    try:
        tf.compat.v1.reset_default_graph()
    except Exception:
        pass

import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.ensemble import IsolationForest
from scipy import sparse
from typing import Optional
import warnings


def compute_stagate_score(
    X: np.ndarray,
    coords: np.ndarray,
    rad_cutoff: Optional[float] = None,
    n_pcs: int = 50,
    hidden_dims: Optional[list] = None,
    n_epochs: int = 500,
    alpha: float = 0.0,
    contamination: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute anomaly score using STAGATE embedding + Isolation Forest.

    This function:
    1. Creates AnnData object with spatial coordinates
    2. Runs STAGATE to get spatial-aware embedding
    3. Applies Isolation Forest on the embedding

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    rad_cutoff : float, optional
        Radius cutoff for spatial neighbors. If None, auto-computed.
    n_pcs : int
        Number of PCA components before STAGATE
    hidden_dims : list
        Hidden layer dimensions for STAGATE encoder
    n_epochs : int
        Training epochs (default 500, same as STAGATE default)
    alpha : float
        Balance between reconstruction (0) and spatial consistency (1)
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
        If STAGATE is not installed
    RuntimeError
        If STAGATE training fails
    """
    import STAGATE

    # Default for mutable argument
    if hidden_dims is None:
        hidden_dims = [512, 30]

    # Reset TF graph for clean run
    _reset_tf_graph()

    # Convert to sparse if dense (STAGATE expects sparse)
    if not sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X)
    else:
        X_sparse = X

    # Create AnnData object
    adata = ad.AnnData(X_sparse)
    adata.obsm["spatial"] = coords.astype(np.float32)

    # HVG selection on raw counts (seurat_v3 requires counts before log1p)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=min(3000, adata.n_vars),
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    # Preprocessing (standard STAGATE pipeline)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # PCA
    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))

    # Build spatial graph
    if rad_cutoff is None:
        # Auto-compute radius based on data spread
        coord_range = coords.max(axis=0) - coords.min(axis=0)
        rad_cutoff = np.sqrt(coord_range[0] * coord_range[1]) / 20

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)

    # Check if spatial network was built
    if "Spatial_Net" not in adata.uns or adata.uns["Spatial_Net"].shape[0] == 0:
        raise RuntimeError(
            f"STAGATE failed to build spatial network. "
            f"rad_cutoff={rad_cutoff} may be too small for this dataset."
        )

    # Train STAGATE on CPU (RTX 5090 compute capability 12.0 not supported by TF CUDA kernels)
    with warnings.catch_warnings(), tf.device('/CPU:0'):
        warnings.simplefilter("ignore")
        STAGATE.train_STAGATE(
            adata,
            alpha=alpha,
            hidden_dims=hidden_dims,
            n_epochs=n_epochs,
            random_seed=random_state,
            verbose=False,
        )

    # Check if embedding was created
    if "STAGATE" not in adata.obsm:
        raise RuntimeError("STAGATE training failed: no embedding created")

    # Get embedding
    embedding = adata.obsm["STAGATE"]

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


def compute_stagate_embedding(
    X: np.ndarray,
    coords: np.ndarray,
    rad_cutoff: Optional[float] = None,
    n_pcs: int = 50,
    hidden_dims: Optional[list] = None,
    n_epochs: int = 500,
    alpha: float = 0.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Get STAGATE embedding without anomaly detection.

    Useful for visualization or custom downstream analysis.

    Returns
    -------
    embedding : np.ndarray
        STAGATE embedding [n_spots, hidden_dims[-1]]
    """
    import STAGATE

    # Default for mutable argument
    if hidden_dims is None:
        hidden_dims = [512, 30]

    # Reset TF graph for clean run
    _reset_tf_graph()

    if not sparse.issparse(X):
        X_sparse = sparse.csr_matrix(X)
    else:
        X_sparse = X

    adata = ad.AnnData(X_sparse)
    adata.obsm["spatial"] = coords.astype(np.float32)

    # HVG selection on raw counts (seurat_v3 requires counts before log1p)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=min(3000, adata.n_vars),
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.scale(adata)
    sc.tl.pca(adata, n_comps=min(n_pcs, adata.n_vars - 1, adata.n_obs - 1))

    if rad_cutoff is None:
        coord_range = coords.max(axis=0) - coords.min(axis=0)
        rad_cutoff = np.sqrt(coord_range[0] * coord_range[1]) / 20

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=rad_cutoff)

    if "Spatial_Net" not in adata.uns or adata.uns["Spatial_Net"].shape[0] == 0:
        raise RuntimeError("STAGATE failed to build spatial network")

    with warnings.catch_warnings(), tf.device('/CPU:0'):
        warnings.simplefilter("ignore")
        STAGATE.train_STAGATE(
            adata,
            alpha=alpha,
            hidden_dims=hidden_dims,
            n_epochs=n_epochs,
            random_seed=random_state,
            verbose=False,
        )

    if "STAGATE" not in adata.obsm:
        raise RuntimeError("STAGATE training failed")

    return adata.obsm["STAGATE"]


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_spots = 500
    n_genes = 200

    X = np.random.exponential(1, (n_spots, n_genes))
    coords = np.random.rand(n_spots, 2) * 100

    print("Testing STAGATE wrapper...")
    try:
        scores = compute_stagate_score(X, coords)
        print(f"Score shape: {scores.shape}")
        print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print("STAGATE wrapper test PASSED")
    except Exception as e:
        print(f"STAGATE wrapper test FAILED: {e}")
        raise
