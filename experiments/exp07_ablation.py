"""
Experiment 07: Ablation Studies

Systematic ablation to understand the contribution of each component:
1. Lambda_pos: Weight for position prediction loss
2. Hidden dimension: Model capacity
3. K neighbors: Spatial context size

This identifies the critical design choices.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from config import ExperimentConfig, AblationConfig, DEFAULT_CONFIG, ABLATION_CONFIG
from core import (
    generate_synthetic_data,
    prepare_data,
    InversePredictionModel,
    train_model,
    compute_scores,
    compute_pca_error,
    set_seed,
)


def run_ablation(
    X: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    seed: int,
    lambda_pos: float,
    hidden_dim: int,
    k_neighbors: int,
    config: ExperimentConfig,
) -> dict:
    """Run a single ablation configuration."""
    # Set all random seeds for reproducibility
    set_seed(seed)

    # Prepare data with specified k
    data = prepare_data(X, coords, k=k_neighbors, device=config.device)

    # Build model with specified hidden dim
    model = InversePredictionModel(
        in_dim=data["n_genes"],
        hid_dim=hidden_dim,
        dropout=config.dropout,
    ).to(config.device)

    # Train with specified lambda_pos
    model = train_model(
        model,
        data["x_tensor"],
        data["coords_tensor"],
        data["edge_index"],
        n_epochs=config.n_epochs,
        lr=config.learning_rate,
        lambda_pos=lambda_pos,
        lambda_self=config.lambda_self,
        verbose=False,
    )

    # Compute scores
    scores = compute_scores(
        model, data["x_tensor"], data["coords_tensor"],
        data["edge_index"], random_state=seed,
    )

    pca_error = compute_pca_error(data["X_norm"])

    # Compute AUCs
    ectopic_mask = labels == 1
    intrinsic_mask = labels == 2

    return {
        "lambda_pos": lambda_pos,
        "hidden_dim": hidden_dim,
        "k_neighbors": k_neighbors,
        "seed": seed,
        "auc_ectopic_pos": roc_auc_score(ectopic_mask.astype(int), scores["s_pos"]),
        "auc_intrinsic_pos": roc_auc_score(intrinsic_mask.astype(int), scores["s_pos"]),
        "auc_ectopic_pca": roc_auc_score(ectopic_mask.astype(int), pca_error),
        "auc_intrinsic_pca": roc_auc_score(intrinsic_mask.astype(int), pca_error),
    }


def run(
    config: ExperimentConfig = None,
    ablation_config: AblationConfig = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run ablation studies."""
    if config is None:
        config = DEFAULT_CONFIG
    if ablation_config is None:
        ablation_config = ABLATION_CONFIG

    all_results = []

    # Use all seeds for Nature Methods submission (statistical rigor)
    seeds = config.seeds

    # 1. Lambda_pos ablation (keep others at default)
    if verbose:
        print("Ablation 1: Lambda_pos")

    for lambda_pos in tqdm(ablation_config.lambda_pos_values, disable=not verbose):
        for seed in seeds:
            try:
                X, coords, labels, _, _ = generate_synthetic_data(
                    n_spots=config.n_spots,
                    n_genes=config.n_genes,
                    n_ectopic=config.n_ectopic,
                    n_intrinsic=config.n_intrinsic,
                    n_modules=config.n_modules,
                    random_state=seed,
                )
                result = run_ablation(
                    X, coords, labels, seed,
                    lambda_pos=lambda_pos,
                    hidden_dim=config.hidden_dim,
                    k_neighbors=config.k_neighbors,
                    config=config,
                )
                result["ablation_type"] = "lambda_pos"
                all_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Failed: lambda_pos={lambda_pos}, seed={seed}: {e}")

    # 2. Hidden dimension ablation
    if verbose:
        print("Ablation 2: Hidden dimension")

    for hidden_dim in tqdm(ablation_config.hidden_dim_values, disable=not verbose):
        for seed in seeds:
            try:
                X, coords, labels, _, _ = generate_synthetic_data(
                    n_spots=config.n_spots,
                    n_genes=config.n_genes,
                    n_ectopic=config.n_ectopic,
                    n_intrinsic=config.n_intrinsic,
                    n_modules=config.n_modules,
                    random_state=seed,
                )
                result = run_ablation(
                    X, coords, labels, seed,
                    lambda_pos=config.lambda_pos,
                    hidden_dim=hidden_dim,
                    k_neighbors=config.k_neighbors,
                    config=config,
                )
                result["ablation_type"] = "hidden_dim"
                all_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Failed: hidden_dim={hidden_dim}, seed={seed}: {e}")

    # 3. K neighbors ablation
    if verbose:
        print("Ablation 3: K neighbors")

    for k in tqdm(ablation_config.k_neighbor_values, disable=not verbose):
        for seed in seeds:
            try:
                X, coords, labels, _, _ = generate_synthetic_data(
                    n_spots=config.n_spots,
                    n_genes=config.n_genes,
                    n_ectopic=config.n_ectopic,
                    n_intrinsic=config.n_intrinsic,
                    n_modules=config.n_modules,
                    random_state=seed,
                )
                result = run_ablation(
                    X, coords, labels, seed,
                    lambda_pos=config.lambda_pos,
                    hidden_dim=config.hidden_dim,
                    k_neighbors=k,
                    config=config,
                )
                result["ablation_type"] = "k_neighbors"
                all_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Failed: k={k}, seed={seed}: {e}")

    results = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 60)
        print("Ablation Summary")
        print("=" * 60)

        for ablation_type in results["ablation_type"].unique():
            print(f"\n{ablation_type}:")
            subset = results[results["ablation_type"] == ablation_type]

            if ablation_type == "lambda_pos":
                group_col = "lambda_pos"
            elif ablation_type == "hidden_dim":
                group_col = "hidden_dim"
            else:
                group_col = "k_neighbors"

            summary = subset.groupby(group_col)["auc_ectopic_pos"].agg(["mean", "std"])
            print(summary.round(3).to_string())

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp07_ablation.csv"))
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    results = run(config, ABLATION_CONFIG, verbose=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
