"""
Experiment 02: Competitor Comparison

Compare inverse prediction against existing anomaly detection methods:
- Spatial: LISA, SpotSweeper, Neighbor Difference
- Global: LOF, Isolation Forest, Mahalanobis, OCSVM, PCA

Key metric: Combined detection performance (Ectopic + Intrinsic)
using the two-axis framework (Inv_PosError + PCA_Error).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
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

    # Get scores
    scores_inv = compute_scores(
        model, data["x_tensor"], data["coords_tensor"],
        data["edge_index"], random_state=seed,
    )
    scores_baseline = compute_all_baselines(
        data["X_norm"], coords, k=config.k_neighbors, random_state=seed,
    )

    # Define methods to compare
    methods = {
        # Our method (two-axis)
        "Inv_PosError": scores_inv["s_pos"],
        "PCA_Error": scores_baseline["pca_error"],
        # Spatial methods
        "Neighbor_Diff": scores_baseline["neighbor_diff"],
        "LISA": scores_baseline["lisa"],
        "SpotSweeper": scores_baseline["spotsweeper"],
        # Global methods
        "LOF": scores_baseline["lof"],
        "IF": scores_baseline["isolation_forest"],
        "Mahalanobis": scores_baseline["mahalanobis"],
        "OCSVM": scores_baseline["ocsvm"],
    }

    # Normalize scores to [0, 1] for combination
    def normalize(s):
        s = np.nan_to_num(s)
        s_min, s_max = s.min(), s.max()
        if s_max - s_min < 1e-10:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)

    # Compute AUCs
    results = []
    ectopic_mask = labels == 1
    intrinsic_mask = labels == 2
    any_anomaly_mask = (labels > 0)

    for name, score in methods.items():
        score = np.nan_to_num(score)

        # Ectopic detection (Ectopic vs ALL)
        auc_ectopic = roc_auc_score(ectopic_mask.astype(int), score)
        ap_ectopic = average_precision_score(ectopic_mask.astype(int), score)

        # Intrinsic detection (Intrinsic vs ALL)
        auc_intrinsic = roc_auc_score(intrinsic_mask.astype(int), score)
        ap_intrinsic = average_precision_score(intrinsic_mask.astype(int), score)

        # Any anomaly detection
        auc_any = roc_auc_score(any_anomaly_mask.astype(int), score)

        results.append({
            "method": name,
            "auc_ectopic": auc_ectopic,
            "auc_intrinsic": auc_intrinsic,
            "ap_ectopic": ap_ectopic,
            "ap_intrinsic": ap_intrinsic,
            "auc_any": auc_any,
            "seed": seed,
        })

    # Two-axis combination: max(Inv_PosError, PCA_Error)
    combined_max = np.maximum(
        normalize(scores_inv["s_pos"]),
        normalize(scores_baseline["pca_error"])
    )
    results.append({
        "method": "TwoAxis_Max",
        "auc_ectopic": roc_auc_score(ectopic_mask.astype(int), combined_max),
        "auc_intrinsic": roc_auc_score(intrinsic_mask.astype(int), combined_max),
        "ap_ectopic": average_precision_score(ectopic_mask.astype(int), combined_max),
        "ap_intrinsic": average_precision_score(intrinsic_mask.astype(int), combined_max),
        "auc_any": roc_auc_score(any_anomaly_mask.astype(int), combined_max),
        "seed": seed,
    })

    # Two-axis combination: sum
    combined_sum = normalize(scores_inv["s_pos"]) + normalize(scores_baseline["pca_error"])
    results.append({
        "method": "TwoAxis_Sum",
        "auc_ectopic": roc_auc_score(ectopic_mask.astype(int), combined_sum),
        "auc_intrinsic": roc_auc_score(intrinsic_mask.astype(int), combined_sum),
        "ap_ectopic": average_precision_score(ectopic_mask.astype(int), combined_sum),
        "ap_intrinsic": average_precision_score(intrinsic_mask.astype(int), combined_sum),
        "auc_any": roc_auc_score(any_anomaly_mask.astype(int), combined_sum),
        "seed": seed,
    })

    return pd.DataFrame(results)


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run competitor comparison across all seeds."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []
    iterator = tqdm(config.seeds, desc="Competitor Comparison") if verbose else config.seeds

    for seed in iterator:
        try:
            result = run_single_seed(seed, config, verbose=False)
            all_results.append(result)
        except Exception as e:
            if verbose:
                print(f"Seed {seed} failed: {e}")

    if not all_results:
        raise RuntimeError("All seeds failed in exp02_competitor")
    results = pd.concat(all_results, ignore_index=True)

    if verbose:
        print("\n" + "=" * 60)
        print("Competitor Comparison Summary")
        print("=" * 60)
        summary = results.groupby("method").agg({
            "auc_ectopic": ["mean", "std"],
            "auc_intrinsic": ["mean", "std"],
            "auc_any": ["mean", "std"],
        }).round(3)
        print(summary.to_string())

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp02_competitor_comparison.csv"))
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
