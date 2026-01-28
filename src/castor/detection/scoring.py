"""Z-scoring and four-category diagnosis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_mad_zscore(scores: np.ndarray) -> np.ndarray:
    """MAD-based z-score (robust to outliers).

    Parameters
    ----------
    scores : np.ndarray
        Raw anomaly scores.

    Returns
    -------
    np.ndarray
        Z-scores.
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median)) + 1e-6
    return (scores - median) / (1.4826 * mad)


def compute_diagnosis(
    z_local: np.ndarray,
    z_global: np.ndarray,
    threshold_local: float = 2.0,
    threshold_global: float = 2.0,
) -> pd.DataFrame:
    """Assign four diagnostic categories based on dual-axis z-scores.

    Categories
    ----------
    - **Normal** -- both scores below threshold
    - **Contextual Anomaly** -- only local score above threshold
    - **Intrinsic Anomaly** -- only global score above threshold
    - **Confirmed Anomaly** -- both scores above threshold

    Parameters
    ----------
    z_local : np.ndarray
        Local (contextual) z-scores.
    z_global : np.ndarray
        Global (intrinsic) z-scores.
    threshold_local : float
        Threshold for the contextual axis.
    threshold_global : float
        Threshold for the intrinsic axis.

    Returns
    -------
    pd.DataFrame
        Columns: ``Local_Z``, ``Global_Z``, ``Diagnosis``, ``Confidence``.
    """
    n = len(z_local)
    diagnosis = np.array(["Normal"] * n, dtype=object)

    contextual_mask = (z_local > threshold_local) & (z_global <= threshold_global)
    intrinsic_mask = (z_global > threshold_global) & (z_local <= threshold_local)
    confirmed_mask = (z_local > threshold_local) & (z_global > threshold_global)

    diagnosis[contextual_mask] = "Contextual Anomaly"
    diagnosis[intrinsic_mask] = "Intrinsic Anomaly"
    diagnosis[confirmed_mask] = "Confirmed Anomaly"

    # Sigmoid-based confidence
    confidence = np.zeros(n)

    confidence[contextual_mask] = 1.0 / (
        1.0 + np.exp(-(z_local[contextual_mask] - threshold_local))
    )
    confidence[intrinsic_mask] = 1.0 / (
        1.0 + np.exp(-(z_global[intrinsic_mask] - threshold_global))
    )

    min_z = np.minimum(z_local[confirmed_mask], z_global[confirmed_mask])
    base_thresh = max(threshold_local, threshold_global)
    confidence[confirmed_mask] = 1.0 / (1.0 + np.exp(-(min_z - base_thresh)))

    normal_mask = diagnosis == "Normal"
    max_z = np.maximum(z_local[normal_mask], z_global[normal_mask])
    base_thresh_normal = min(threshold_local, threshold_global)
    confidence[normal_mask] = 1.0 - 1.0 / (1.0 + np.exp(-(-max_z + base_thresh_normal)))

    return pd.DataFrame(
        {
            "Local_Z": z_local,
            "Global_Z": z_global,
            "Diagnosis": diagnosis,
            "Confidence": confidence,
        }
    )
