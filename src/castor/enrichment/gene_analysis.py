"""Gene contribution analysis for anomaly types."""

from __future__ import annotations

import numpy as np
import pandas as pd


def identify_spot_contributing_genes(
    X_processed: np.ndarray,
    results: pd.DataFrame,
    gene_names: list[str] | None = None,
    top_n_spots: int = 10,
    top_n_genes: int | None = None,
) -> dict[int, pd.DataFrame]:
    """Identify contributing genes for individual high-scoring spots.

    For each top-scoring spot, computes differential expression relative to
    the mean of normal spots and returns a gene ranking.

    Parameters
    ----------
    X_processed : np.ndarray
        Processed expression matrix ``[n_spots, n_genes]``.
    results : pd.DataFrame
        CASTOR results with ``Local_Z``, ``Global_Z``, ``Diagnosis``.
    gene_names : list[str] | None
        Gene names (uses indices if *None*).
    top_n_spots : int
        Number of top-scoring spots to analyse.
    top_n_genes : int | None
        Keep top *N* genes per spot. *None* returns all.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping from spot index to a DataFrame with columns
        ``Gene``, ``Score``, ``Fold_Change``.
    """
    final_score = np.maximum(results["Local_Z"].values, results["Global_Z"].values)
    top_spot_indices = np.argsort(final_score)[-top_n_spots:][::-1]

    normal_mask = (results["Diagnosis"] == "Normal").values
    if normal_mask.sum() == 0:
        normal_mask = np.ones(len(results), dtype=bool)
    normal_mean = np.asarray(X_processed[normal_mask].mean(axis=0)).flatten()

    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(X_processed.shape[1])]

    spot_results: dict[int, pd.DataFrame] = {}
    for idx in top_spot_indices:
        spot_expr = np.asarray(X_processed[idx]).flatten()
        diff = spot_expr - normal_mean
        contribution = np.abs(diff)

        if top_n_genes is None:
            sorted_idx = np.argsort(contribution)[::-1]
        else:
            sorted_idx = np.argsort(contribution)[-top_n_genes:][::-1]

        spot_results[int(idx)] = pd.DataFrame(
            {
                "Gene": [gene_names[i] for i in sorted_idx],
                "Score": contribution[sorted_idx],
                "Fold_Change": diff[sorted_idx],
            }
        )

    return spot_results


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
