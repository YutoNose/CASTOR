"""
Experiment 01: Cross-Detection AUC Validation

Key Question: Does each anomaly score selectively detect its target anomaly type?

Expected Results:
- Inv_PosError: High AUC for Ectopic, low for Intrinsic
- PCA_Error: High AUC for Intrinsic, low for Ectopic
- Neighbor_Diff: Medium for both (non-selective)

This validates the core hypothesis of the inverse prediction approach.
"""

import numpy as np
import pandas as pd
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
    compute_auc_metrics,
    set_seed,
)


def run_single_seed(seed: int, config: ExperimentConfig, verbose: bool = False):
    """Run experiment for a single seed."""
    # Set all random seeds for reproducibility
    set_seed(seed)

    # Generate data
    X, coords, labels, ectopic_idx, intrinsic_idx = generate_synthetic_data(
        n_spots=config.n_spots,
        n_genes=config.n_genes,
        n_ectopic=config.n_ectopic,
        n_intrinsic=config.n_intrinsic,
        n_modules=config.n_modules,
        min_distance_factor=config.min_distance_factor,
        random_state=seed,
    )

    # Prepare data
    data = prepare_data(
        X, coords,
        k=config.k_neighbors,
        device=config.device,
    )

    # Train inverse prediction model
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
        model,
        data["x_tensor"],
        data["coords_tensor"],
        data["edge_index"],
        random_state=seed,
    )

    # Compute baselines
    scores_baseline = compute_all_baselines(
        data["X_norm"], coords,
        k=config.k_neighbors,
        random_state=seed,
    )

    # Combine all scores
    all_scores = {
        "Inv_PosError": scores_inv["s_pos"],
        "Inv_SelfRecon": scores_inv["s_self"],
        "Inv_NeighborRecon": scores_inv["s_neighbor"],
        "PCA_Error": scores_baseline["pca_error"],
        "Neighbor_Diff": scores_baseline["neighbor_diff"],
        "LISA": scores_baseline["lisa"],
        "LOF": scores_baseline["lof"],
        "IF": scores_baseline["isolation_forest"],
    }

    # Compute AUC metrics
    auc_df = compute_auc_metrics(all_scores, labels)
    auc_df["seed"] = seed

    return auc_df


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """
    Run cross-detection experiment across all seeds.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    verbose : bool
        Print progress

    Returns
    -------
    results : pd.DataFrame
        AUC metrics for all scores across all seeds
    """
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []

    iterator = tqdm(config.seeds, desc="Cross-Detection") if verbose else config.seeds

    for seed in iterator:
        try:
            result = run_single_seed(seed, config, verbose=False)
            all_results.append(result)
        except Exception as e:
            if verbose:
                print(f"Seed {seed} failed: {e}")

    if not all_results:
        raise RuntimeError("All seeds failed")

    results = pd.concat(all_results, ignore_index=True)

    # Compute summary statistics
    summary = results.groupby("score").agg({
        "auc_ectopic": ["mean", "std"],
        "auc_intrinsic": ["mean", "std"],
        "selectivity_ectopic": ["mean", "std"],
        "selectivity_intrinsic": ["mean", "std"],
    })

    if verbose:
        print("\n" + "=" * 60)
        print("Cross-Detection AUC Summary")
        print("=" * 60)
        for score in results["score"].unique():
            score_data = results[results["score"] == score]
            print(f"\n{score}:")
            print(f"  Ectopic AUC:   {score_data['auc_ectopic'].mean():.3f} ± {score_data['auc_ectopic'].std():.3f}")
            print(f"  Intrinsic AUC: {score_data['auc_intrinsic'].mean():.3f} ± {score_data['auc_intrinsic'].std():.3f}")
            print(f"  Selectivity (Ect): {score_data['selectivity_ectopic'].mean():.3f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp01_cross-detection_auc.csv"))
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    results = run(config, verbose=True)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
