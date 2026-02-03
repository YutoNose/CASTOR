"""
Experiment 24: False Positive Spatial Characterization

Characterizes the spatial properties of false positives from Inv_PosError.

Hypothesis
----------
False positives (normal spots with high position-prediction error) cluster
at spatial transition zones — boundaries between distinct expression domains
— rather than being randomly distributed.  This is expected because boundary
spots have ambiguous spatial expression profiles that are harder to predict.

Design
------
- Run the standard pipeline on synthetic data with known anomalies.
- Define FP as normal spots (label=0) with s_pos above the 95th percentile
  among normal spots.
- Characterize FP vs true-negative (TN) distributions on three axes:
  1. Expression gradient magnitude (L2 distance to neighbour mean)
  2. Neighbour expression heterogeneity (mean pairwise L2 among k-NN)
  3. Local Moran's I of s_pos (spatial autocorrelation)
- Statistical test: Mann-Whitney U (unpaired, FP and TN are different groups)
  with rank-biserial effect size.

Statistical controls
--------------------
- set_seed at the start of each run_single_seed call
- Mann-Whitney U (not Wilcoxon signed-rank) for unpaired comparison
- Rank-biserial effect size: r = 1 - 2U / (n1 * n2)
- FDR correction (BH) across the 3 tests per seed
- Skip seed if n_fp < 5 (insufficient for reliable test)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from config import ExperimentConfig, DEFAULT_CONFIG
from core import (
    generate_synthetic_data,
    prepare_data,
    InversePredictionModel,
    train_model,
    compute_scores,
    set_seed,
)
from core.evaluation import apply_fdr_correction

# FP threshold: top X% of s_pos among normal spots
FP_PERCENTILE = 95


def _compute_gradient_magnitude(
    X_norm: np.ndarray, nn_indices: np.ndarray
) -> np.ndarray:
    """
    Compute expression gradient magnitude per spot.

    Gradient is measured as the L2 distance between each spot's expression
    and the mean expression of its k-NN neighbors.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalized expression [n_spots, n_genes].
    nn_indices : np.ndarray
        Neighbor indices [n_spots, k] (excluding self).

    Returns
    -------
    gradient : np.ndarray
        Gradient magnitude per spot [n_spots].
    """
    neighbor_means = X_norm[nn_indices].mean(axis=1)  # [n_spots, n_genes]
    return np.linalg.norm(X_norm - neighbor_means, axis=1)


def _compute_neighbor_heterogeneity(
    X_norm: np.ndarray, nn_indices: np.ndarray
) -> np.ndarray:
    """
    Compute neighbor expression heterogeneity per spot.

    Measured as the mean pairwise L2 distance among a spot's k-NN neighbors.

    Parameters
    ----------
    X_norm : np.ndarray
        Normalized expression [n_spots, n_genes].
    nn_indices : np.ndarray
        Neighbor indices [n_spots, k] (excluding self).

    Returns
    -------
    heterogeneity : np.ndarray
        Heterogeneity per spot [n_spots].
    """
    n_spots = X_norm.shape[0]
    k = nn_indices.shape[1]
    het = np.zeros(n_spots)

    for i in range(n_spots):
        nbr_expr = X_norm[nn_indices[i]]  # [k, n_genes]
        # Compute variance across neighbors (faster than all pairwise distances)
        het[i] = np.sqrt(nbr_expr.var(axis=0).sum())

    return het


def _compute_local_moran(
    score: np.ndarray, coords: np.ndarray, nn_indices: np.ndarray
) -> np.ndarray:
    """
    Compute local Moran's I for each spot.

    Local Moran's I_i = z_i * sum_j(w_ij * z_j) where z are standardized
    scores and w_ij = 1/k for k-NN neighbors.

    Positive I_i indicates spatial clustering (similar neighbors).
    Negative I_i indicates spatial outlier (dissimilar neighbors).

    Parameters
    ----------
    score : np.ndarray
        Score vector [n_spots].
    coords : np.ndarray
        Spatial coordinates [n_spots, 2] (unused, kept for API clarity).
    nn_indices : np.ndarray
        Neighbor indices [n_spots, k].

    Returns
    -------
    local_moran : np.ndarray
        Local Moran's I per spot [n_spots].
    """
    z = (score - score.mean()) / (score.std() + 1e-10)
    n_spots = len(score)
    k = nn_indices.shape[1]

    local_i = np.zeros(n_spots)
    for i in range(n_spots):
        neighbor_z = z[nn_indices[i]]
        local_i[i] = z[i] * neighbor_z.mean()

    return local_i


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """
    Compute rank-biserial correlation from Mann-Whitney U statistic.

    r = 1 - 2U / (n1 * n2)
    Range: [-1, 1]. Positive means group 1 tends to have larger values.

    Benchmarks: |r| < 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, > 0.5 large.
    """
    denom = n1 * n2
    if denom == 0:
        return 0.0
    return 1.0 - 2.0 * u_stat / denom


def run_single_seed(
    seed: int, config: ExperimentConfig, verbose: bool = False
) -> dict:
    """Run FP characterization for a single seed."""
    set_seed(seed)

    # Generate data
    X, coords, labels, ectopic_idx, intrinsic_idx = generate_synthetic_data(
        n_spots=config.n_spots,
        n_genes=config.n_genes,
        n_ectopic=config.n_ectopic,
        n_intrinsic=config.n_intrinsic,
        n_modules=config.n_modules,
        random_state=seed,
    )

    # Prepare data
    data = prepare_data(X, coords, k=config.k_neighbors, device=config.device)

    # Train model
    model = InversePredictionModel(
        in_dim=data["n_genes"],
        hid_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(config.device)

    model = train_model(
        model,
        data["x_tensor"],
        data["coords_tensor"],
        data["edge_index"],
        n_epochs=config.n_epochs,
        lr=config.learning_rate,
        lambda_pos=config.lambda_pos,
        lambda_self=config.lambda_self,
        verbose=verbose,
    )

    # Compute scores
    scores_inv = compute_scores(
        model, data["x_tensor"], data["coords_tensor"],
        data["edge_index"], random_state=seed,
    )
    s_pos = scores_inv["s_pos"]

    # Identify FP and TN
    normal_mask = labels == 0
    normal_scores = s_pos[normal_mask]
    threshold = np.percentile(normal_scores, FP_PERCENTILE)

    fp_mask = normal_mask & (s_pos > threshold)
    tn_mask = normal_mask & (s_pos <= threshold)

    n_fp = int(fp_mask.sum())
    n_tn = int(tn_mask.sum())

    result = {"seed": seed, "n_fp": n_fp, "n_tn": n_tn}

    if n_fp < 5:
        # Insufficient FP for reliable testing
        import warnings
        warnings.warn(
            f"Seed {seed}: only {n_fp} FP spots (< 5 minimum), skipping."
        )
        for prefix in ["gradient", "heterogeneity", "moran"]:
            result[f"{prefix}_mean_fp"] = np.nan
            result[f"{prefix}_mean_tn"] = np.nan
            result[f"{prefix}_effect_size"] = np.nan
            result[f"{prefix}_pval"] = np.nan
            result[f"{prefix}_pval_fdr"] = np.nan
        return result

    # Compute k-NN indices for spatial features
    nn = NearestNeighbors(n_neighbors=config.k_neighbors + 1)
    nn.fit(coords)
    _, nn_indices = nn.kneighbors(coords)
    nn_indices = nn_indices[:, 1:]  # Remove self

    # 1. Expression gradient magnitude
    gradient = _compute_gradient_magnitude(data["X_norm"], nn_indices)

    # 2. Neighbor expression heterogeneity
    heterogeneity = _compute_neighbor_heterogeneity(data["X_norm"], nn_indices)

    # 3. Local Moran's I of s_pos
    local_moran = _compute_local_moran(s_pos, coords, nn_indices)

    # Compare FP vs TN for each feature
    raw_pvals = []
    features = {
        "gradient": gradient,
        "heterogeneity": heterogeneity,
        "moran": local_moran,
    }

    for feat_name, feat_values in features.items():
        fp_vals = feat_values[fp_mask]
        tn_vals = feat_values[tn_mask]

        result[f"{feat_name}_mean_fp"] = float(fp_vals.mean())
        result[f"{feat_name}_mean_tn"] = float(tn_vals.mean())

        # Mann-Whitney U test (unpaired)
        u_stat, pval = stats.mannwhitneyu(
            fp_vals, tn_vals, alternative="two-sided"
        )
        effect = _rank_biserial(u_stat, n_fp, n_tn)

        result[f"{feat_name}_effect_size"] = effect
        result[f"{feat_name}_pval"] = pval
        raw_pvals.append(pval)

    # FDR correction across 3 tests
    fdr_result = apply_fdr_correction(raw_pvals, alpha=0.05, method="fdr_bh")
    for i, feat_name in enumerate(features.keys()):
        result[f"{feat_name}_pval_fdr"] = fdr_result["p_adjusted"][i]

    return result


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run FP characterization across all seeds."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []
    iterator = tqdm(config.seeds, desc="FP Characterization") if verbose else config.seeds

    for seed in iterator:
        try:
            result = run_single_seed(seed, config, verbose=False)
            all_results.append(result)
        except Exception as e:
            if verbose:
                print(f"Seed {seed} failed: {e}")

    if not all_results:
        raise RuntimeError("All seeds failed in exp24_fp_characterization")

    results = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 60)
        print("False Positive Characterization Summary")
        print("=" * 60)

        valid = results.dropna(subset=["gradient_pval_fdr"])
        if len(valid) > 0:
            print(f"\nValid seeds: {len(valid)}/{len(results)}")
            print(f"\nFP count: {valid['n_fp'].mean():.1f} ± {valid['n_fp'].std():.1f}")

            for feat in ["gradient", "heterogeneity", "moran"]:
                mean_fp = valid[f"{feat}_mean_fp"].mean()
                mean_tn = valid[f"{feat}_mean_tn"].mean()
                effect = valid[f"{feat}_effect_size"].mean()
                pval_med = valid[f"{feat}_pval_fdr"].median()
                print(
                    f"\n  {feat}:"
                    f"\n    FP mean={mean_fp:.4f}  TN mean={mean_tn:.4f}"
                    f"\n    effect_size={effect:.3f}  median p_fdr={pval_med:.2e}"
                )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "exp24_fp_characterization.csv"),
    )
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    results = run(config, verbose=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
