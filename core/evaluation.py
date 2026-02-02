"""
Evaluation metrics for inverse prediction experiments.

Key metrics:
1. Cross-Detection AUC: Does each score detect only its target anomaly type?
2. Selectivity: |AUC_target - AUC_non_target|
3. Score Correlation: Are the two detection axes independent?
4. Separation Power: Can we distinguish Ectopic vs Intrinsic?
5. Statistical Tests: Wilcoxon signed-rank with multiple testing correction

IMPORTANT: This implementation uses PROPER AUC evaluation without
artificial label conditioning that inflates performance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Optional, Tuple


def compute_auc_metrics(
    scores: Dict[str, np.ndarray],
    labels: np.ndarray,
    ectopic_label: int = 1,
    intrinsic_label: int = 2,
) -> pd.DataFrame:
    """
    Compute AUC metrics for each score.

    IMPORTANT: This uses PROPER evaluation where anomaly detection is:
    - Ectopic AUC: Can score distinguish Ectopic from ALL other spots?
    - Intrinsic AUC: Can score distinguish Intrinsic from ALL other spots?

    This is more realistic than label-conditioned evaluation.

    Parameters
    ----------
    scores : dict
        Dictionary of score arrays {name: array}
    labels : np.ndarray
        Anomaly labels (0=normal, 1=ectopic, 2=intrinsic)
    ectopic_label : int
        Label for ectopic anomalies
    intrinsic_label : int
        Label for intrinsic anomalies

    Returns
    -------
    df : pd.DataFrame
        DataFrame with AUC metrics for each score
    """
    ectopic_mask = labels == ectopic_label
    intrinsic_mask = labels == intrinsic_label

    results = []

    for name, score in scores.items():
        # Handle NaN/Inf
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        # Ectopic AUC: Ectopic vs ALL others (including intrinsic)
        # This is the realistic evaluation scenario
        if ectopic_mask.sum() > 0:
            y_ect = ectopic_mask.astype(int)
            auc_ectopic = roc_auc_score(y_ect, score)
            ap_ectopic = average_precision_score(y_ect, score)
        else:
            auc_ectopic = np.nan
            ap_ectopic = np.nan

        # Intrinsic AUC: Intrinsic vs ALL others (including ectopic)
        if intrinsic_mask.sum() > 0:
            y_int = intrinsic_mask.astype(int)
            auc_intrinsic = roc_auc_score(y_int, score)
            ap_intrinsic = average_precision_score(y_int, score)
        else:
            auc_intrinsic = np.nan
            ap_intrinsic = np.nan

        results.append(
            {
                "score": name,
                "auc_ectopic": auc_ectopic,
                "auc_intrinsic": auc_intrinsic,
                "ap_ectopic": ap_ectopic,
                "ap_intrinsic": ap_intrinsic,
                "selectivity_ectopic": auc_ectopic - auc_intrinsic,
                "selectivity_intrinsic": auc_intrinsic - auc_ectopic,
            }
        )

    return pd.DataFrame(results)


def compute_correlation_matrix(scores: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlations between scores.

    Low correlation between Ectopic-specific and Intrinsic-specific scores
    indicates independent detection axes.

    Parameters
    ----------
    scores : dict
        Dictionary of score arrays

    Returns
    -------
    df : pd.DataFrame
        Correlation matrix
    """
    names = list(scores.keys())
    n = len(names)

    corr_matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            score_i = np.nan_to_num(scores[name_i])
            score_j = np.nan_to_num(scores[name_j])
            # Check for zero variance to avoid pearsonr errors
            if np.std(score_i) < 1e-10 or np.std(score_j) < 1e-10:
                corr = np.nan if i != j else 1.0
            else:
                corr, _ = stats.pearsonr(score_i, score_j)
            corr_matrix[i, j] = corr

    return pd.DataFrame(corr_matrix, index=names, columns=names)


def compute_separation_auc(
    score_ectopic: np.ndarray,
    score_intrinsic: np.ndarray,
    labels: np.ndarray,
    ectopic_label: int = 1,
    intrinsic_label: int = 2,
) -> float:
    """
    Compute separation AUC between Ectopic and Intrinsic anomalies.

    Using (score_ectopic - score_intrinsic) as a discriminator:
    - Ectopic should have positive difference
    - Intrinsic should have negative difference

    Parameters
    ----------
    score_ectopic : np.ndarray
        Score designed for ectopic detection (e.g., Inv_PosError)
    score_intrinsic : np.ndarray
        Score designed for intrinsic detection (e.g., PCA_Error)
    labels : np.ndarray
        Ground truth labels
    ectopic_label, intrinsic_label : int
        Label values

    Returns
    -------
    sep_auc : float
        Separation AUC (how well can we distinguish anomaly types)
    """
    # Normalize scores to [0, 1]
    def normalize(s):
        s = np.nan_to_num(s)
        s_min, s_max = s.min(), s.max()
        if s_max - s_min < 1e-10:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)

    s_ect_norm = normalize(score_ectopic)
    s_int_norm = normalize(score_intrinsic)

    diff = s_ect_norm - s_int_norm

    # Only evaluate on anomalies
    ectopic_mask = labels == ectopic_label
    intrinsic_mask = labels == intrinsic_label
    anomaly_mask = ectopic_mask | intrinsic_mask

    if anomaly_mask.sum() < 2:
        return np.nan

    # Label: 1 = Ectopic, 0 = Intrinsic
    y = ectopic_mask[anomaly_mask].astype(int)
    diff_anomalies = diff[anomaly_mask]

    if y.sum() == 0 or y.sum() == len(y):
        return np.nan

    return roc_auc_score(y, diff_anomalies)


def compute_position_accuracy(
    pos_pred: np.ndarray,
    coords: np.ndarray,
    donor_coords: np.ndarray,
    labels: np.ndarray,
    ectopic_label: int = 1,
) -> Dict[str, float]:
    """
    Analyze position prediction accuracy for ectopic anomalies.

    For ectopic anomalies, check if predicted position is closer to:
    - True position (where the spot actually is)
    - Donor position (where the expression came from)

    This validates the inverse prediction hypothesis.

    Parameters
    ----------
    pos_pred : np.ndarray
        Predicted positions [n_spots, 2]
    coords : np.ndarray
        True positions [n_spots, 2]
    donor_coords : np.ndarray
        Donor positions [n_spots, 2]
    labels : np.ndarray
        Anomaly labels
    ectopic_label : int
        Label for ectopic anomalies

    Returns
    -------
    metrics : dict
        Position accuracy metrics
    """
    ectopic_mask = labels == ectopic_label

    if ectopic_mask.sum() == 0:
        return {
            "n_ectopic": 0,
            "fraction_closer_to_donor": np.nan,
            "mean_dist_to_true": np.nan,
            "mean_dist_to_donor": np.nan,
        }

    # For ectopic spots
    pred_ect = pos_pred[ectopic_mask]
    true_ect = coords[ectopic_mask]
    donor_ect = donor_coords[ectopic_mask]

    # Distance from predicted to true position
    dist_to_true = np.linalg.norm(pred_ect - true_ect, axis=1)
    # Distance from predicted to donor position
    dist_to_donor = np.linalg.norm(pred_ect - donor_ect, axis=1)

    # Count how many predictions are closer to donor than true
    closer_to_donor = (dist_to_donor < dist_to_true).sum()

    return {
        "n_ectopic": int(ectopic_mask.sum()),
        "fraction_closer_to_donor": closer_to_donor / ectopic_mask.sum(),
        "mean_dist_to_true": float(dist_to_true.mean()),
        "mean_dist_to_donor": float(dist_to_donor.mean()),
    }


def statistical_tests(
    results_df: pd.DataFrame,
    method1_col: str,
    method2_col: str,
    alpha: float = 0.05,
    correction: str = "bonferroni",
    n_comparisons: int = 1,
) -> Dict[str, float]:
    """
    Perform statistical significance tests with multiple testing correction.

    Uses Wilcoxon signed-rank test (paired, non-parametric).

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with results from multiple seeds
    method1_col, method2_col : str
        Column names for the two methods to compare
    alpha : float
        Significance level (before correction)
    correction : str
        Multiple testing correction method ('bonferroni' or 'none')
    n_comparisons : int
        Number of comparisons for Bonferroni correction

    Returns
    -------
    test_results : dict
        Dictionary with test statistic, p-value, and significance
    """
    values1 = results_df[method1_col].values
    values2 = results_df[method2_col].values

    # Remove NaN pairs
    mask = ~(np.isnan(values1) | np.isnan(values2))
    values1 = values1[mask]
    values2 = values2[mask]

    if len(values1) < 5:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "n_samples": len(values1),
        }

    try:
        stat, pval = stats.wilcoxon(values1, values2, alternative="two-sided")
    except Exception:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "n_samples": len(values1),
        }

    n = len(values1)

    # Effect size: matched-pairs rank-biserial correlation (Kerby 2014)
    # |r| < 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, > 0.5 large
    # stat = min(W+, W-), so magnitude = 1 - 4*stat/(n*(n+1))
    # Sign from mean difference to indicate direction
    effect_magnitude = 1 - 4 * stat / (n * (n + 1))
    effect_size = effect_magnitude * np.sign(np.mean(values1 - values2))

    # Apply Bonferroni correction
    if correction == "bonferroni" and n_comparisons > 1:
        adjusted_alpha = alpha / n_comparisons
    else:
        adjusted_alpha = alpha

    return {
        "statistic": stat,
        "p_value": pval,
        "significant": pval < adjusted_alpha,
        "effect_size": effect_size,
        "n_samples": n,
        "mean_diff": float(np.mean(values1 - values2)),
        "adjusted_alpha": adjusted_alpha,
        "n_comparisons": n_comparisons,
    }


def summarize_results(
    all_results: List[Dict],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Summarize results across multiple runs with confidence intervals.

    Parameters
    ----------
    all_results : list of dict
        Results from multiple experimental runs
    confidence : float
        Confidence level for intervals

    Returns
    -------
    df : pd.DataFrame
        Summary with mean, std, and confidence intervals
    """
    df = pd.DataFrame(all_results)

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    summary_data = []
    for col in numeric_cols:
        values = df[col].dropna().values
        if len(values) == 0:
            continue

        mean = values.mean()
        std = values.std(ddof=1)
        n = len(values)

        # Bootstrap confidence interval (distribution-free, appropriate for
        # bounded metrics like AUC where normality cannot be assumed)
        if n > 1:
            n_bootstrap = 10000
            rng = np.random.RandomState(42)
            boot_means = np.array([
                rng.choice(values, size=n, replace=True).mean()
                for _ in range(n_bootstrap)
            ])
            alpha = 1 - confidence
            ci_lower = np.percentile(boot_means, 100 * alpha / 2)
            ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
        else:
            ci_lower = np.nan
            ci_upper = np.nan

        summary_data.append(
            {
                "metric": col,
                "mean": mean,
                "std": std,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": n,
                "ci_method": "bootstrap_percentile",
            }
        )

    return pd.DataFrame(summary_data)
