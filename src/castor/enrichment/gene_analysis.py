"""Gene contribution analysis for anomaly types."""

from __future__ import annotations

import numpy as np
import pandas as pd


def identify_contributing_genes(
    X_processed: np.ndarray,
    results: pd.DataFrame,
    gene_names: list[str] | None = None,
    diagnosis_type: str = "Confirmed Anomaly",
    top_n: int | None = 20,
) -> pd.DataFrame:
    """Identify genes driving a specific anomaly type.

    Computes differential expression between anomalous and normal spots.

    Parameters
    ----------
    X_processed : np.ndarray
        Processed expression matrix ``[n_spots, n_genes]``.
    results : pd.DataFrame
        CASTOR results with ``Diagnosis`` column.
    gene_names : list[str] | None
        Gene names (uses indices if *None*).
    diagnosis_type : str
        Which diagnosis category to analyse.
    top_n : int | None
        Return top *N* genes.  *None* returns all.

    Returns
    -------
    pd.DataFrame
        Columns: ``Gene``, ``Score``, ``Mean_Anomaly``, ``Mean_Normal``, ``Fold_Change``.
    """
    anomaly_mask = (results["Diagnosis"] == diagnosis_type).values
    normal_mask = (results["Diagnosis"] == "Normal").values

    if anomaly_mask.sum() == 0:
        return pd.DataFrame(columns=["Gene", "Score", "Mean_Anomaly", "Mean_Normal", "Fold_Change"])

    if normal_mask.sum() == 0:
        normal_mask = ~anomaly_mask

    anomaly_mean = np.asarray(X_processed[anomaly_mask].mean(axis=0)).flatten()
    normal_mean = np.asarray(X_processed[normal_mask].mean(axis=0)).flatten()

    contribution = np.abs(anomaly_mean - normal_mean)
    fold_change = anomaly_mean - normal_mean

    if top_n is None:
        top_idx = np.argsort(contribution)[::-1]
    else:
        top_idx = np.argsort(contribution)[-top_n:][::-1]

    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(X_processed.shape[1])]

    return pd.DataFrame(
        {
            "Gene": [gene_names[i] for i in top_idx],
            "Score": contribution[top_idx],
            "Mean_Anomaly": anomaly_mean[top_idx],
            "Mean_Normal": normal_mean[top_idx],
            "Fold_Change": fold_change[top_idx],
        }
    )
