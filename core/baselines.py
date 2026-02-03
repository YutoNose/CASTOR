"""
Baseline methods for comparison with Inverse Prediction.

This module consolidates all competing methods:
1. Spatial methods: LISA, NeighborDiff, SpotSweeper
2. Global methods: PCA, LOF, IF, Mahalanobis, OCSVM

All functions follow a consistent interface:
    score = compute_xxx(X, coords, ...)  -> np.ndarray [n_spots]

Higher score = more anomalous.

Note on hyperparameters: All baseline methods use default parameters
(e.g., k=15, n_components=50, n_neighbors=20) without task-specific
tuning. CASTOR's parameters (lambda_pos=0.5, hidden_dim=64, etc.)
were selected via ablation study (exp07), not validation set tuning.
For fair comparison, both CASTOR and baselines use fixed parameters
without access to test labels.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy import sparse
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from typing import Optional


# =============================================================================
# Spatial Methods
# =============================================================================


def compute_neighbor_diff(
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """
    Spatial neighbor difference score.

    For each spot, compute ||expression - mean(neighbor expressions)||.
    Higher score = more different from spatial neighbors.

    This is the SINGLE correct implementation using vectorized numpy.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    k : int
        Number of neighbors

    Returns
    -------
    score : np.ndarray
        Neighbor difference score [n_spots]
    """
    if sparse.issparse(X):
        X = X.toarray()

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    indices = indices[:, 1:]  # Remove self (index 0)

    # CORRECT vectorized computation:
    # X[indices] has shape [n_spots, k, n_genes]
    # mean(axis=1) averages over the k neighbors -> [n_spots, n_genes]
    neighbor_means = X[indices].mean(axis=1)

    # L2 norm of difference
    return np.linalg.norm(X - neighbor_means, axis=1)


def compute_local_spatial_deviation(
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """
    Local Spatial Deviation score.

    Computes mean absolute deviation from spatial neighbors.
    Higher score = more different from local spatial context.

    NOTE: This is NOT Local Moran's I. For true Local Moran's I, use compute_lisa().

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    k : int
        Number of neighbors

    Returns
    -------
    score : np.ndarray
        Local spatial deviation score [n_spots]
    """
    if sparse.issparse(X):
        X = X.toarray()

    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    indices = indices[:, 1:]  # Remove self

    # Standardize expression
    X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Vectorized computation
    neighbor_means = X_scaled[indices].mean(axis=1)

    # Mean absolute difference from neighbor mean
    return np.abs(X_scaled - neighbor_means).mean(axis=1)


def compute_lisa(
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """
    True Local Moran's I (LISA) - Classical spatial autocorrelation statistic.

    For each spot i, Local Moran's I is:
        I_i = z_i * sum_j(w_ij * z_j)

    where z_i is the standardized value and w_ij is the spatial weight.

    We compute this for each gene and aggregate across genes.
    Negative values indicate spatial outliers (high-low or low-high).
    We return |I_i| so higher = more anomalous (outlier).

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    k : int
        Number of neighbors

    Returns
    -------
    score : np.ndarray
        LISA-based anomaly score [n_spots] (higher = more anomalous)
    """
    if sparse.issparse(X):
        X = X.toarray()

    n_spots, n_genes = X.shape

    tree = KDTree(coords)
    distances, indices = tree.query(coords, k=k + 1)
    indices = indices[:, 1:]  # Remove self
    distances = distances[:, 1:]

    # Standardize expression per gene
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    z = (X - X_mean) / X_std

    # Row-standardized spatial weights (inverse distance)
    # w_ij = 1/d_ij, then row-normalize
    weights = 1.0 / (distances + 1e-8)
    weights = weights / weights.sum(axis=1, keepdims=True)  # Row normalize

    # Compute Local Moran's I for each gene
    # I_i = z_i * sum_j(w_ij * z_j)
    # For vectorized computation:
    # neighbor_z has shape [n_spots, k, n_genes]
    neighbor_z = z[indices]  # [n_spots, k, n_genes]
    # weighted_neighbor_z: multiply each neighbor's z by its weight
    # weights has shape [n_spots, k], expand to [n_spots, k, 1]
    weighted_sum = (neighbor_z * weights[:, :, np.newaxis]).sum(axis=1)  # [n_spots, n_genes]

    # Local Moran's I per gene
    local_moran = z * weighted_sum  # [n_spots, n_genes]

    # Aggregate across genes - use negative of mean because:
    # - Positive I: similar to neighbors (cluster)
    # - Negative I: different from neighbors (outlier)
    # We want outliers to have HIGH score, so we look for negative I values
    # Using -mean captures spots that are consistently different from neighbors
    # But for anomaly detection, we care about magnitude, so use |I| aggregated

    # For spatial outliers, I_i is typically negative (high-low or low-high pattern)
    # Spots with consistently negative I across genes are outliers
    # Use mean Local Moran's I - negative values indicate spatial outliers
    mean_local_moran = local_moran.mean(axis=1)

    # Return negative mean so that outliers (negative I) get high scores
    return -mean_local_moran


def compute_spotsweeper(
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """
    SpotSweeper-like score: Combination of QC metrics.

    SpotSweeper uses:
    1. Local variance (technical artifacts)
    2. Library size outliers
    3. Neighbor difference

    We approximate with a combination of these metrics.

    NOTE: If X is already z-scored (centered), the library size component
    will have low variance since sum of z-scored values is near zero.
    In this case, we use squared sum (L2 norm) as an alternative metric
    that captures outliers in normalized data.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    k : int
        Number of neighbors

    Returns
    -------
    score : np.ndarray
        SpotSweeper-like score [n_spots]
    """
    if sparse.issparse(X):
        X = X.toarray()

    # 1. Local variance per spot
    local_var = X.var(axis=1)

    # 2. Neighbor difference
    neighbor_diff = compute_neighbor_diff(X, coords, k)

    # 3. Library size or L2 norm based metric
    lib_size = X.sum(axis=1)
    lib_std = lib_size.std()

    # If data appears to be z-scored (low library size variance),
    # use L2 norm instead which captures expression magnitude outliers
    if lib_std < 1e-3 * np.abs(lib_size.mean() + 1e-8):
        # Data is likely z-scored - use L2 norm as alternative
        magnitude = np.linalg.norm(X, axis=1)
        lib_z = np.abs((magnitude - magnitude.mean()) / (magnitude.std() + 1e-8))
    else:
        # Normal case - use library size
        lib_z = np.abs((lib_size - lib_size.mean()) / (lib_std + 1e-8))

    # Helper for robust z-score
    def robust_z(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < 1e-10:
            mad = np.std(x)
        if mad < 1e-10:
            return np.zeros_like(x)
        return (x - med) / (mad * 1.4826)

    # Combined score
    return robust_z(local_var) + robust_z(neighbor_diff) + lib_z


# =============================================================================
# Global Methods
# =============================================================================


def compute_pca_error(
    X: np.ndarray,
    n_components: int = 50,
    X_train: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    PCA reconstruction error.

    Higher score = expression doesn't fit main variance directions.
    This is effective for Intrinsic anomaly detection.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes] to score.
    n_components : int
        Number of PCA components.
    X_train : np.ndarray, optional
        If provided, fit PCA on X_train and compute reconstruction error
        on X (inductive mode). If None, fit and score on X (transductive).

    Returns
    -------
    score : np.ndarray
        PCA reconstruction error [n_spots]
    """
    if sparse.issparse(X):
        X = X.toarray()

    fit_data = X_train if X_train is not None else X
    if sparse.issparse(fit_data):
        fit_data = fit_data.toarray()

    n_comp = min(n_components, fit_data.shape[1] - 1, fit_data.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(fit_data)
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)

    return np.linalg.norm(X - X_reconstructed, axis=1)


def compute_lof(
    X: np.ndarray,
    n_neighbors: int = 20,
    X_train: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Local Outlier Factor in expression space.

    Higher score = more anomalous.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes] to score.
    n_neighbors : int
        Number of neighbors for LOF.
    X_train : np.ndarray, optional
        If provided, fit LOF on X_train in novelty detection mode and
        score X. If None, use transductive mode (fit and score on X).

    Returns
    -------
    score : np.ndarray
        LOF score [n_spots]
    """
    if sparse.issparse(X):
        X = X.toarray()

    if X_train is not None:
        if sparse.issparse(X_train):
            X_train = X_train.toarray()
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        lof.fit(X_train)
        return -lof.decision_function(X)
    else:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
        lof.fit(X)
        return -lof.negative_outlier_factor_


def compute_isolation_forest(
    X: np.ndarray,
    n_estimators: int = 100,
    contamination: str = "auto",
    random_state: int = 42,
    X_train: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Isolation Forest score.

    Higher score = more anomalous.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes] to score.
    n_estimators : int
        Number of trees.
    contamination : str or float
        Expected proportion of anomalies. Default 'auto' lets IF estimate
        the threshold using the original paper's offset, avoiding the need
        to assume a specific contamination rate.
    random_state : int
        Random state.
    X_train : np.ndarray, optional
        If provided, fit on X_train and score X (inductive mode).
        If None, fit and score on X (transductive).

    Returns
    -------
    score : np.ndarray
        IF score [n_spots]
    """
    if sparse.issparse(X):
        X = X.toarray()

    fit_data = X_train if X_train is not None else X
    if sparse.issparse(fit_data):
        fit_data = fit_data.toarray()

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    iso.fit(fit_data)
    return -iso.decision_function(X)


def compute_mahalanobis(
    X: np.ndarray,
    n_components: int = 20,
) -> np.ndarray:
    """
    Mahalanobis distance-based outlier detection.

    Uses PCA for dimensionality reduction first.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    n_components : int
        Number of PCA components before Mahalanobis

    Returns
    -------
    score : np.ndarray
        Mahalanobis score [n_spots]
    """
    if sparse.issparse(X):
        X = X.toarray()

    try:
        n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)

        ee = EllipticEnvelope(contamination=0.1, random_state=42)
        ee.fit(X_pca)
        return -ee.decision_function(X_pca)
    except Exception as e:
        import warnings
        warnings.warn(
            f"EllipticEnvelope failed ({e}), falling back to Euclidean distance from mean. "
            "Scores will have different scale."
        )
        mean = X.mean(axis=0)
        return np.linalg.norm(X - mean, axis=1)


def compute_ocsvm(
    X: np.ndarray,
    nu: float = 0.1,
    n_components: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """
    One-Class SVM anomaly score.

    Uses PCA for dimensionality reduction first.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes]
    nu : float
        Upper bound on fraction of outliers
    n_components : int
        Number of PCA components

    Returns
    -------
    score : np.ndarray
        OCSVM score [n_spots]
    """
    if sparse.issparse(X):
        X = X.toarray()

    # Reduce dimensionality
    n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)

    # Subsample if too large
    if len(X_pca) > 2000:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_pca), 2000, replace=False)
        ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        ocsvm.fit(X_pca[idx])
    else:
        ocsvm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        ocsvm.fit(X_pca)

    return -ocsvm.decision_function(X_pca)


# =============================================================================
# Compute all baselines at once
# =============================================================================


def compute_all_baselines(
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
    random_state: int = 42,
    X_train: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all baseline scores at once.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes] to score.
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    k : int
        Number of neighbors for spatial methods
    random_state : int
        Random state for reproducibility
    X_train : np.ndarray, optional
        If provided, global methods (PCA, LOF, IF) are fit on X_train
        and score X (inductive mode for train/test experiments).

    Returns
    -------
    scores : dict
        Dictionary with all baseline scores

    Notes
    -----
    IMPORTANT: Spatial methods (neighbor_diff, lisa, local_spatial_deviation,
    spotsweeper) are always computed transductively on X, using the test set's
    spatial structure. This means they can see the spatial neighborhood of test
    spots, which may give them an advantage in train/test experiments compared
    to methods that only use training data. When comparing in inductive settings,
    this asymmetry should be acknowledged in the paper's Methods section.
    Our method (Inv_PosError) is also evaluated transductively in the standard
    setting (exp01/exp10), ensuring a fair comparison with spatial baselines.
    """
    return {
        # Spatial methods (always transductive - computed on X)
        "neighbor_diff": compute_neighbor_diff(X, coords, k),
        "lisa": compute_lisa(X, coords, k),  # True Local Moran's I
        "local_spatial_deviation": compute_local_spatial_deviation(X, coords, k),
        "spotsweeper": compute_spotsweeper(X, coords, k),
        # Global methods (inductive if X_train provided)
        "pca_error": compute_pca_error(X, X_train=X_train),
        "lof": compute_lof(X, X_train=X_train),
        "isolation_forest": compute_isolation_forest(X, random_state=random_state, X_train=X_train),
        "mahalanobis": compute_mahalanobis(X),
        "ocsvm": compute_ocsvm(X),
    }
