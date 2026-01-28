"""Built-in intrinsic anomaly detectors.

All detectors follow the interface::

    def detector(X: np.ndarray, **kwargs) -> np.ndarray:
        ...  # returns scores [n_spots], higher = more anomalous

Importing this module registers them with
:class:`~castor.detection.registry.IntrinsicDetectorRegistry`.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from castor.detection.registry import register_intrinsic_detector


def _to_dense(X: np.ndarray) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()  # type: ignore[union-attr]
    return np.asarray(X)


# ---------------------------------------------------------------------------
# PCA reconstruction error (default)
# ---------------------------------------------------------------------------
@register_intrinsic_detector("pca_error")
def compute_pca_error(X: np.ndarray, *, n_components: int = 50) -> np.ndarray:
    """PCA reconstruction error -- effective for intrinsic anomalies."""
    X = _to_dense(X)
    n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    X_trans = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_trans)
    return np.linalg.norm(X - X_recon, axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Local Outlier Factor
# ---------------------------------------------------------------------------
@register_intrinsic_detector("lof")
def compute_lof(X: np.ndarray, *, n_neighbors: int = 20) -> np.ndarray:
    """Local Outlier Factor in expression space."""
    X = _to_dense(X)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof.fit(X)
    return (-lof.negative_outlier_factor_).astype(np.float64)


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------
@register_intrinsic_detector("isolation_forest")
def compute_isolation_forest(
    X: np.ndarray,
    *,
    n_estimators: int = 100,
    contamination: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Isolation Forest anomaly score."""
    X = _to_dense(X)
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    iso.fit(X)
    return (-iso.decision_function(X)).astype(np.float64)


# ---------------------------------------------------------------------------
# Mahalanobis distance
# ---------------------------------------------------------------------------
@register_intrinsic_detector("mahalanobis")
def compute_mahalanobis(X: np.ndarray, *, n_components: int = 20) -> np.ndarray:
    """Mahalanobis distance via PCA + Elliptic Envelope."""
    X = _to_dense(X)
    try:
        n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X)
        ee = EllipticEnvelope(contamination=0.1, random_state=42)
        ee.fit(X_pca)
        return (-ee.decision_function(X_pca)).astype(np.float64)
    except Exception:
        mean = X.mean(axis=0)
        return np.linalg.norm(X - mean, axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# One-Class SVM
# ---------------------------------------------------------------------------
@register_intrinsic_detector("ocsvm")
def compute_ocsvm(
    X: np.ndarray,
    *,
    nu: float = 0.1,
    n_components: int = 20,
    random_state: int = 42,
) -> np.ndarray:
    """One-Class SVM anomaly score (with PCA pre-reduction)."""
    X = _to_dense(X)
    n_comp = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X)

    if len(X_pca) > 2000:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_pca), 2000, replace=False)
        svm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        svm.fit(X_pca[idx])
    else:
        svm = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        svm.fit(X_pca)

    return (-svm.decision_function(X_pca)).astype(np.float64)
