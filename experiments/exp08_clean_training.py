"""
Experiment 08: Clean Training

Tests the semi-supervised scenario where the model is trained on
clean (normal) data only, then evaluated on data containing anomalies.

This is the realistic deployment scenario where we don't have
labeled anomalies for training.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
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
    compute_pca_error,
    build_spatial_graph,
    normalize_expression,
    normalize_coordinates,
    set_seed,
)


def run_single_seed(seed: int, config: ExperimentConfig, verbose: bool = False):
    """Run clean training experiment for a single seed."""
    # Set all random seeds for reproducibility (numpy + torch)
    set_seed(seed)

    # Generate clean training data (no anomalies)
    X_clean, coords_clean, _, _, _ = generate_synthetic_data(
        n_spots=config.n_spots,
        n_genes=config.n_genes,
        n_ectopic=0,  # No anomalies in training
        n_intrinsic=0,
        random_state=seed,
    )

    # Generate test data with anomalies
    X_test, coords_test, labels_test, _, _ = generate_synthetic_data(
        n_spots=config.n_spots,
        n_genes=config.n_genes,
        n_ectopic=config.n_ectopic,
        n_intrinsic=config.n_intrinsic,
        random_state=seed + 1000,  # Different seed for test
    )

    # Prepare training data
    data_train = prepare_data(
        X_clean, coords_clean,
        k=config.k_neighbors,
        device=config.device,
    )

    # Train model on clean data
    model = InversePredictionModel(
        in_dim=data_train["n_genes"],
        hid_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(config.device)

    model = train_model(
        model,
        data_train["x_tensor"],
        data_train["coords_tensor"],
        data_train["edge_index"],
        n_epochs=config.n_epochs,
        lr=config.learning_rate,
        lambda_pos=config.lambda_pos,
        lambda_self=config.lambda_self,
        verbose=verbose,
    )

    # Prepare test data (using same normalization params would be better
    # but for synthetic data, prepare_data works fine)
    data_test = prepare_data(
        X_test, coords_test,
        k=config.k_neighbors,
        device=config.device,
    )

    # Compute scores on test data
    scores = compute_scores(
        model, data_test["x_tensor"], data_test["coords_tensor"],
        data_test["edge_index"], random_state=seed,
    )

    pca_error = compute_pca_error(data_test["X_norm"])

    # Compute AUCs
    ectopic_mask = labels_test == 1
    intrinsic_mask = labels_test == 2
    any_anomaly = (labels_test > 0)

    result = {"seed": seed}

    # Inv_PosError
    result["auc_ectopic_pos"] = roc_auc_score(ectopic_mask.astype(int), scores["s_pos"])
    result["auc_intrinsic_pos"] = roc_auc_score(intrinsic_mask.astype(int), scores["s_pos"])
    result["auc_any_pos"] = roc_auc_score(any_anomaly.astype(int), scores["s_pos"])

    # PCA Error
    result["auc_ectopic_pca"] = roc_auc_score(ectopic_mask.astype(int), pca_error)
    result["auc_intrinsic_pca"] = roc_auc_score(intrinsic_mask.astype(int), pca_error)
    result["auc_any_pca"] = roc_auc_score(any_anomaly.astype(int), pca_error)

    # Combined (max of normalized scores)
    def normalize(s):
        s = np.nan_to_num(s)
        return (s - s.min()) / (s.max() - s.min() + 1e-10)

    combined = np.maximum(normalize(scores["s_pos"]), normalize(pca_error))
    result["auc_ectopic_combined"] = roc_auc_score(ectopic_mask.astype(int), combined)
    result["auc_intrinsic_combined"] = roc_auc_score(intrinsic_mask.astype(int), combined)
    result["auc_any_combined"] = roc_auc_score(any_anomaly.astype(int), combined)

    # Also run contaminated training for comparison
    data_contam = prepare_data(
        X_test, coords_test,
        k=config.k_neighbors,
        device=config.device,
    )

    model_contam = InversePredictionModel(
        in_dim=data_contam["n_genes"],
        hid_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(config.device)

    model_contam = train_model(
        model_contam,
        data_contam["x_tensor"],
        data_contam["coords_tensor"],
        data_contam["edge_index"],
        n_epochs=config.n_epochs,
        lr=config.learning_rate,
        lambda_pos=config.lambda_pos,
        lambda_self=config.lambda_self,
        verbose=False,
    )

    scores_contam = compute_scores(
        model_contam, data_contam["x_tensor"], data_contam["coords_tensor"],
        data_contam["edge_index"], random_state=seed,
    )

    result["auc_ectopic_contam"] = roc_auc_score(ectopic_mask.astype(int), scores_contam["s_pos"])
    result["auc_intrinsic_contam"] = roc_auc_score(intrinsic_mask.astype(int), scores_contam["s_pos"])

    return result


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run clean training experiment."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []
    iterator = tqdm(config.seeds, desc="Clean Training") if verbose else config.seeds

    for seed in iterator:
        try:
            result = run_single_seed(seed, config, verbose=False)
            all_results.append(result)
        except Exception as e:
            if verbose:
                print(f"Seed {seed} failed: {e}")

    results = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 60)
        print("Clean Training vs Contaminated Training")
        print("=" * 60)
        print("\nClean Training (trained on normal data only):")
        print(f"  Ectopic AUC (Inv_PosError):   {results['auc_ectopic_pos'].mean():.3f} ± {results['auc_ectopic_pos'].std():.3f}")
        print(f"  Intrinsic AUC (PCA_Error):    {results['auc_intrinsic_pca'].mean():.3f} ± {results['auc_intrinsic_pca'].std():.3f}")
        print(f"  Any Anomaly AUC (Combined):   {results['auc_any_combined'].mean():.3f} ± {results['auc_any_combined'].std():.3f}")

        print("\nContaminated Training (trained on data with anomalies):")
        print(f"  Ectopic AUC (Inv_PosError):   {results['auc_ectopic_contam'].mean():.3f} ± {results['auc_ectopic_contam'].std():.3f}")

        print("\nDifference (Clean - Contaminated):")
        diff = results['auc_ectopic_pos'] - results['auc_ectopic_contam']
        print(f"  Ectopic AUC difference:       {diff.mean():.3f} ± {diff.std():.3f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp08_clean_training.csv"))
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
