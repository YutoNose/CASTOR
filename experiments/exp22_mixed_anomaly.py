"""
Experiment 22: Mixed Anomaly Overlap

Tests how the dual-axis framework behaves when ectopic and intrinsic
anomalies co-occur in the same spot.

Hypothesis
----------
Even when a spot is simultaneously ectopic (misplaced) AND intrinsic
(aberrant expression), the two detection axes (Inv_PosError and PCA_Error)
each respond to its respective anomaly component.  As overlap increases,
ectopic AUC should remain stable while intrinsic detection of mixed spots
improves.

Design
------
- Generate standard synthetic data (ectopic + intrinsic, non-overlapping).
- For each overlap_fraction in {0, 0.1, 0.25, 0.5, 0.75, 1.0}, select
  that fraction of ectopic spots and apply intrinsic-style perturbation
  on top (boosting 15-30 % of genes by 3-10x).
- Mixed spots are labelled 3 (both ectopic AND intrinsic).

Label encoding
--------------
- 0 = normal
- 1 = ectopic-only
- 2 = intrinsic-only
- 3 = mixed (ectopic + intrinsic)

Statistical controls
--------------------
- set_seed at the start of each run_single_seed call
- Perturbation applied to the raw count matrix produced by ZINB generation
  (same rng discipline as core/data_generation.py)
- FDR correction (BH) across pairwise tests within each overlap level
- Both AUC and AUPRC reported
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
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
    compute_all_baselines,
    set_seed,
)
from core.evaluation import apply_fdr_correction

# Fraction of ectopic spots that also receive intrinsic perturbation
OVERLAP_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def _apply_intrinsic_perturbation(
    X: np.ndarray,
    spot_indices: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Apply intrinsic-style perturbation to selected spots.

    Mimics the intrinsic anomaly injection in core/data_generation.py:
    15-30 % of genes are boosted 3-10x, plus ~5 % downregulated.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix [n_spots, n_genes] (will be modified in-place).
    spot_indices : np.ndarray
        Indices of spots to perturb.
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    X : np.ndarray
        Modified expression matrix.
    """
    n_genes = X.shape[1]
    for idx in spot_indices:
        n_affected = rng.randint(max(20, n_genes // 7), max(50, n_genes // 3))
        affected_genes = rng.choice(n_genes, n_affected, replace=False)
        boost_factors = rng.uniform(3.0, 10.0, n_affected)
        X[idx, affected_genes] *= boost_factors

        # Also downregulate some genes
        n_down = n_affected // 3
        down_genes = rng.choice(
            np.setdiff1d(np.arange(n_genes), affected_genes),
            min(n_down, n_genes - n_affected),
            replace=False,
        )
        X[idx, down_genes] *= rng.uniform(0.1, 0.3, len(down_genes))

    return X


def run_single_seed(
    seed: int,
    config: ExperimentConfig,
    overlap_fraction: float,
    verbose: bool = False,
) -> dict:
    """Run mixed anomaly analysis for a single seed and overlap fraction."""
    set_seed(seed)

    # Generate base data
    X, coords, labels, ectopic_idx, intrinsic_idx = generate_synthetic_data(
        n_spots=config.n_spots,
        n_genes=config.n_genes,
        n_ectopic=config.n_ectopic,
        n_intrinsic=config.n_intrinsic,
        n_modules=config.n_modules,
        random_state=seed,
    )

    # Select ectopic spots to receive additional intrinsic perturbation
    rng = np.random.RandomState(seed + 10000)  # Separate rng for overlap selection
    n_overlap = int(len(ectopic_idx) * overlap_fraction)

    if n_overlap > 0 and len(ectopic_idx) > 0:
        overlap_indices = rng.choice(ectopic_idx, n_overlap, replace=False)
        X = _apply_intrinsic_perturbation(X, overlap_indices, rng)
        labels[overlap_indices] = 3  # Mixed label
    else:
        overlap_indices = np.array([], dtype=int)

    # Prepare data and train
    data = prepare_data(X, coords, k=config.k_neighbors, device=config.device)

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
    scores_baseline = compute_all_baselines(
        data["X_norm"], coords, k=config.k_neighbors, random_state=seed,
    )

    s_pos = scores_inv["s_pos"]
    s_pca = scores_baseline["pca_error"]

    # Masks
    ectopic_only_mask = labels == 1
    intrinsic_only_mask = labels == 2
    mixed_mask = labels == 3
    normal_mask = labels == 0

    # Ectopic detection: ectopic-only + mixed vs normal + intrinsic-only
    ectopic_positive = ectopic_only_mask | mixed_mask
    # Intrinsic detection: intrinsic-only + mixed vs normal + ectopic-only
    intrinsic_positive = intrinsic_only_mask | mixed_mask
    # Any anomaly detection: all anomalies vs normal
    any_anomaly = ectopic_positive | intrinsic_only_mask

    result = {
        "seed": seed,
        "overlap_fraction": overlap_fraction,
        "n_overlap": n_overlap,
        "n_ectopic_only": int(ectopic_only_mask.sum()),
        "n_intrinsic_only": int(intrinsic_only_mask.sum()),
        "n_mixed": int(mixed_mask.sum()),
    }

    # AUC: ectopic detection (pos score)
    if ectopic_positive.sum() >= 3 and (~ectopic_positive).sum() > 0:
        y = ectopic_positive.astype(int)
        result["auc_ectopic_pos"] = roc_auc_score(y, s_pos)
        result["ap_ectopic_pos"] = average_precision_score(y, s_pos)
    else:
        result["auc_ectopic_pos"] = np.nan
        result["ap_ectopic_pos"] = np.nan

    # AUC: intrinsic detection (pca score)
    if intrinsic_positive.sum() >= 3 and (~intrinsic_positive).sum() > 0:
        y = intrinsic_positive.astype(int)
        result["auc_intrinsic_pca"] = roc_auc_score(y, s_pca)
        result["ap_intrinsic_pca"] = average_precision_score(y, s_pca)
    else:
        result["auc_intrinsic_pca"] = np.nan
        result["ap_intrinsic_pca"] = np.nan

    # AUC: any anomaly (both scores)
    if any_anomaly.sum() >= 3 and normal_mask.sum() > 0:
        y = any_anomaly.astype(int)
        result["auc_any_pos"] = roc_auc_score(y, s_pos)
        result["auc_any_pca"] = roc_auc_score(y, s_pca)
    else:
        result["auc_any_pos"] = np.nan
        result["auc_any_pca"] = np.nan

    # Dominance analysis on mixed spots
    if mixed_mask.sum() > 0:
        # Normalize scores to [0, 1] for fair comparison
        s_pos_norm = (s_pos - s_pos.min()) / (s_pos.max() - s_pos.min() + 1e-10)
        s_pca_norm = (s_pca - s_pca.min()) / (s_pca.max() - s_pca.min() + 1e-10)

        mixed_pos = s_pos_norm[mixed_mask]
        mixed_pca = s_pca_norm[mixed_mask]

        result["frac_ectopic_dominant"] = float((mixed_pos > mixed_pca).mean())
        result["frac_intrinsic_dominant"] = float((mixed_pca > mixed_pos).mean())
    else:
        result["frac_ectopic_dominant"] = np.nan
        result["frac_intrinsic_dominant"] = np.nan

    # AUC on mixed spots only (ectopic detection)
    if mixed_mask.sum() >= 3 and normal_mask.sum() > 0:
        eval_mask = mixed_mask | normal_mask
        y = mixed_mask[eval_mask].astype(int)
        result["auc_ectopic_pos_mixed_only"] = roc_auc_score(y, s_pos[eval_mask])
        result["auc_intrinsic_pca_mixed_only"] = roc_auc_score(y, s_pca[eval_mask])
    else:
        result["auc_ectopic_pos_mixed_only"] = np.nan
        result["auc_intrinsic_pca_mixed_only"] = np.nan

    return result


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run mixed anomaly analysis across all seeds and overlap fractions."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []
    total = len(config.seeds) * len(OVERLAP_FRACTIONS)
    iterator = tqdm(total=total, desc="Mixed Anomaly") if verbose else None

    for seed in config.seeds:
        for overlap_frac in OVERLAP_FRACTIONS:
            try:
                result = run_single_seed(seed, config, overlap_frac, verbose=False)
                all_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Seed {seed}, overlap {overlap_frac} failed: {e}")
            if iterator is not None:
                iterator.update(1)

    if iterator is not None:
        iterator.close()

    if not all_results:
        raise RuntimeError("All seeds failed in exp22_mixed_anomaly")

    results = pd.DataFrame(all_results)

    # FDR correction per overlap level: test AUC_ectopic_pos vs baselines
    for overlap_frac in OVERLAP_FRACTIONS:
        subset = results[results["overlap_fraction"] == overlap_frac]
        if len(subset) < 10:
            continue

        raw_pvals = []
        test_keys = []

        # Test pos vs pca for ectopic detection
        vals_pos = subset["auc_ectopic_pos"].dropna().values
        vals_pca = subset["auc_intrinsic_pca"].dropna().values
        n_valid = min(len(vals_pos), len(vals_pca))
        if n_valid >= 10:
            _, pval = stats.wilcoxon(vals_pos[:n_valid], vals_pca[:n_valid])
            raw_pvals.append(pval)
            test_keys.append(f"pval_pos_vs_pca_overlap{overlap_frac}")

        # Test any_pos vs any_pca
        vals_any_pos = subset["auc_any_pos"].dropna().values
        vals_any_pca = subset["auc_any_pca"].dropna().values
        n_valid = min(len(vals_any_pos), len(vals_any_pca))
        if n_valid >= 10:
            _, pval = stats.wilcoxon(vals_any_pos[:n_valid], vals_any_pca[:n_valid])
            raw_pvals.append(pval)
            test_keys.append(f"pval_any_pos_vs_pca_overlap{overlap_frac}")

        if raw_pvals:
            fdr_result = apply_fdr_correction(raw_pvals, alpha=0.05, method="fdr_bh")
            idx = subset.index[0]
            for key, adj_p in zip(test_keys, fdr_result["p_adjusted"]):
                results.loc[idx, key] = adj_p

    if verbose:
        print("\n" + "=" * 60)
        print("Mixed Anomaly Overlap Summary")
        print("=" * 60)
        for overlap_frac in OVERLAP_FRACTIONS:
            subset = results[results["overlap_fraction"] == overlap_frac]
            auc_ect = subset["auc_ectopic_pos"].mean()
            auc_int = subset["auc_intrinsic_pca"].mean()
            frac_ed = subset["frac_ectopic_dominant"].mean()
            print(
                f"  overlap={overlap_frac:.2f}  "
                f"AUC_ect_pos={auc_ect:.3f}  AUC_int_pca={auc_int:.3f}  "
                f"frac_ect_dom={frac_ed:.3f}"
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "exp22_mixed_anomaly.csv"),
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
