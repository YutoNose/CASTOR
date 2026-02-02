"""
Experiment 05: Independence Analysis

Tests whether Inv_PosError and PCA_Error are independent detection axes.

Key metrics:
1. Pearson correlation on all spots
2. Correlation on normal spots only
3. Mutual information
4. Separation AUC (can we distinguish Ectopic from Intrinsic?)

Low correlation validates the two-axis framework.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
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
    compute_correlation_matrix,
    compute_separation_auc,
    set_seed,
)


def run_single_seed(seed: int, config: ExperimentConfig, verbose: bool = False):
    """Run independence analysis for a single seed."""
    # Set all random seeds for reproducibility
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
    scores_baseline = compute_all_baselines(
        data["X_norm"], coords, k=config.k_neighbors, random_state=seed,
    )

    # Key scores
    s_pos = scores_inv["s_pos"]
    s_pca = scores_baseline["pca_error"]
    s_neighbor = scores_baseline["neighbor_diff"]

    # Masks
    normal_mask = labels == 0
    ectopic_mask = labels == 1
    intrinsic_mask = labels == 2

    results = {"seed": seed}

    # 1. Correlation analysis (all spots)
    # Pearson (linear) and Spearman (monotonic, robust to skewed distributions)
    corr_pos_pca, pval_pos_pca = stats.pearsonr(s_pos, s_pca)
    corr_pos_neighbor, _ = stats.pearsonr(s_pos, s_neighbor)
    corr_pca_neighbor, _ = stats.pearsonr(s_pca, s_neighbor)

    results["corr_pos_pca_all"] = corr_pos_pca
    results["corr_pos_pca_pval"] = pval_pos_pca
    results["corr_pos_neighbor_all"] = corr_pos_neighbor
    results["corr_pca_neighbor_all"] = corr_pca_neighbor

    # Spearman rank correlation (more appropriate for right-skewed anomaly scores)
    sp_pos_pca, sp_pos_pca_pval = stats.spearmanr(s_pos, s_pca)
    sp_pos_neighbor, _ = stats.spearmanr(s_pos, s_neighbor)
    sp_pca_neighbor, _ = stats.spearmanr(s_pca, s_neighbor)

    results["spearman_pos_pca_all"] = sp_pos_pca
    results["spearman_pos_pca_pval"] = sp_pos_pca_pval
    results["spearman_pos_neighbor_all"] = sp_pos_neighbor
    results["spearman_pca_neighbor_all"] = sp_pca_neighbor

    # 2. Correlation on normal spots only
    if normal_mask.sum() > 10:
        corr_normal, _ = stats.pearsonr(s_pos[normal_mask], s_pca[normal_mask])
        results["corr_pos_pca_normal"] = corr_normal
        sp_normal, _ = stats.spearmanr(s_pos[normal_mask], s_pca[normal_mask])
        results["spearman_pos_pca_normal"] = sp_normal

    # 3. Correlation on anomalies only
    anomaly_mask = ectopic_mask | intrinsic_mask
    if anomaly_mask.sum() > 10:
        corr_anomaly, _ = stats.pearsonr(s_pos[anomaly_mask], s_pca[anomaly_mask])
        results["corr_pos_pca_anomaly"] = corr_anomaly
        sp_anomaly, _ = stats.spearmanr(s_pos[anomaly_mask], s_pca[anomaly_mask])
        results["spearman_pos_pca_anomaly"] = sp_anomaly

    # 4. Separation AUC: Can we distinguish Ectopic from Intrinsic?
    sep_auc = compute_separation_auc(s_pos, s_pca, labels)
    results["separation_auc"] = sep_auc

    # 5. Score statistics by group
    for name, mask in [("normal", normal_mask), ("ectopic", ectopic_mask), ("intrinsic", intrinsic_mask)]:
        if mask.sum() > 0:
            results[f"s_pos_mean_{name}"] = float(s_pos[mask].mean())
            results[f"s_pos_std_{name}"] = float(s_pos[mask].std())
            results[f"s_pca_mean_{name}"] = float(s_pca[mask].mean())
            results[f"s_pca_std_{name}"] = float(s_pca[mask].std())

    # 6. Compute full correlation matrix for all methods
    all_scores = {
        "Inv_PosError": s_pos,
        "PCA_Error": s_pca,
        "Neighbor_Diff": s_neighbor,
        "LISA": scores_baseline["lisa"],
        "LOF": scores_baseline["lof"],
        "IF": scores_baseline["isolation_forest"],
    }
    corr_matrix = compute_correlation_matrix(all_scores)

    # Store all pairwise correlations for Figure 4a
    results["corr_pos_lisa"] = corr_matrix.loc["Inv_PosError", "LISA"]
    results["corr_pos_lof"] = corr_matrix.loc["Inv_PosError", "LOF"]
    results["corr_pos_if"] = corr_matrix.loc["Inv_PosError", "IF"]
    results["corr_pca_lisa"] = corr_matrix.loc["PCA_Error", "LISA"]
    results["corr_pca_lof"] = corr_matrix.loc["PCA_Error", "LOF"]
    results["corr_pca_if"] = corr_matrix.loc["PCA_Error", "IF"]
    results["corr_neighbor_lisa"] = corr_matrix.loc["Neighbor_Diff", "LISA"]
    results["corr_neighbor_lof"] = corr_matrix.loc["Neighbor_Diff", "LOF"]
    results["corr_neighbor_if"] = corr_matrix.loc["Neighbor_Diff", "IF"]
    results["corr_lisa_lof"] = corr_matrix.loc["LISA", "LOF"]
    results["corr_lisa_if"] = corr_matrix.loc["LISA", "IF"]
    results["corr_lof_if"] = corr_matrix.loc["LOF", "IF"]

    return results


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run independence analysis across all seeds."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []
    iterator = tqdm(config.seeds, desc="Independence Analysis") if verbose else config.seeds

    for seed in iterator:
        try:
            result = run_single_seed(seed, config, verbose=False)
            all_results.append(result)
        except Exception as e:
            if verbose:
                print(f"Seed {seed} failed: {e}")

    if not all_results:
        raise RuntimeError("All seeds failed in exp05_independence")
    results = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 60)
        print("Independence Analysis Summary")
        print("=" * 60)
        print(f"\nPearson Correlations:")
        print(f"  Inv_PosError vs PCA_Error (all):     {results['corr_pos_pca_all'].mean():.3f} ± {results['corr_pos_pca_all'].std():.3f}")
        print(f"  Inv_PosError vs PCA_Error (normal):  {results['corr_pos_pca_normal'].mean():.3f} ± {results['corr_pos_pca_normal'].std():.3f}")
        print(f"  Inv_PosError vs Neighbor_Diff:       {results['corr_pos_neighbor_all'].mean():.3f} ± {results['corr_pos_neighbor_all'].std():.3f}")

        print(f"\nSpearman Rank Correlations:")
        print(f"  Inv_PosError vs PCA_Error (all):     {results['spearman_pos_pca_all'].mean():.3f} ± {results['spearman_pos_pca_all'].std():.3f}")
        print(f"  Inv_PosError vs PCA_Error (normal):  {results['spearman_pos_pca_normal'].mean():.3f} ± {results['spearman_pos_pca_normal'].std():.3f}")
        print(f"  Inv_PosError vs Neighbor_Diff:       {results['spearman_pos_neighbor_all'].mean():.3f} ± {results['spearman_pos_neighbor_all'].std():.3f}")

        print(f"\nSeparation Power:")
        print(f"  Ectopic vs Intrinsic AUC: {results['separation_auc'].mean():.3f} ± {results['separation_auc'].std():.3f}")

        print(f"\nScore Distributions:")
        print(f"  Ectopic   - Inv_PosError: {results['s_pos_mean_ectopic'].mean():.3f}, PCA_Error: {results['s_pca_mean_ectopic'].mean():.3f}")
        print(f"  Intrinsic - Inv_PosError: {results['s_pos_mean_intrinsic'].mean():.3f}, PCA_Error: {results['s_pca_mean_intrinsic'].mean():.3f}")
        print(f"  Normal    - Inv_PosError: {results['s_pos_mean_normal'].mean():.3f}, PCA_Error: {results['s_pca_mean_normal'].mean():.3f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp05_independence_analysis.csv"))
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
