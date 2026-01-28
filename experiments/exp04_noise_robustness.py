"""
Experiment 04: Noise Robustness

Tests the robustness of inverse prediction under various noise conditions:
1. Expression noise (Gaussian)
2. Coordinate noise (jitter)
3. Dropout (missing genes)
4. Library size variation

This validates that the method is practical for real ST data.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
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
    set_seed,
)


def add_expression_noise(X: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add Gaussian noise to expression."""
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_level * X.std(), X.shape)
    return np.clip(X + noise, 0, None)


def add_coordinate_noise(coords: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add jitter to coordinates."""
    rng = np.random.RandomState(seed)
    scale = coords.std() * noise_level
    noise = rng.normal(0, scale, coords.shape)
    return coords + noise


def apply_dropout(X: np.ndarray, dropout_rate: float, seed: int) -> np.ndarray:
    """Apply dropout to expression (set values to 0)."""
    rng = np.random.RandomState(seed)
    mask = rng.rand(*X.shape) > dropout_rate
    return X * mask


def add_library_size_variation(X: np.ndarray, cv: float, seed: int) -> np.ndarray:
    """Add library size variation."""
    rng = np.random.RandomState(seed)
    factors = rng.lognormal(0, cv, (X.shape[0], 1))
    return X * factors


def run_with_perturbation(
    X: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    config: ExperimentConfig,
    seed: int,
    perturbation_type: str,
    perturbation_level: float,
) -> dict:
    """Run experiment with a specific perturbation."""
    # Set all random seeds for reproducibility
    set_seed(seed)

    # Apply perturbation
    if perturbation_type == "expression_noise":
        X_pert = add_expression_noise(X, perturbation_level, seed)
        coords_pert = coords
    elif perturbation_type == "coordinate_noise":
        X_pert = X
        coords_pert = add_coordinate_noise(coords, perturbation_level, seed)
    elif perturbation_type == "dropout":
        X_pert = apply_dropout(X, perturbation_level, seed)
        coords_pert = coords
    elif perturbation_type == "library_size":
        X_pert = add_library_size_variation(X, perturbation_level, seed)
        coords_pert = coords
    else:
        X_pert = X
        coords_pert = coords

    # Prepare data
    data = prepare_data(X_pert, coords_pert, k=config.k_neighbors, device=config.device)

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

    result = {
        "perturbation_type": perturbation_type,
        "perturbation_level": perturbation_level,
        "seed": seed,
    }

    for score_name, score_val in [
        ("Inv_PosError", scores["s_pos"]),
        ("PCA_Error", pca_error),
    ]:
        if ectopic_mask.sum() > 0:
            result[f"{score_name}_auc_ectopic"] = roc_auc_score(
                ectopic_mask.astype(int), score_val
            )
        if intrinsic_mask.sum() > 0:
            result[f"{score_name}_auc_intrinsic"] = roc_auc_score(
                intrinsic_mask.astype(int), score_val
            )

    return result


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run noise robustness experiments."""
    if config is None:
        config = DEFAULT_CONFIG

    # Define perturbation settings
    perturbations = {
        "none": [0.0],
        "expression_noise": [0.1, 0.2, 0.5, 1.0],
        "coordinate_noise": [0.05, 0.1, 0.2],
        "dropout": [0.1, 0.2, 0.3, 0.5],
        "library_size": [0.2, 0.5, 1.0],
    }

    all_results = []

    for perturbation_type, levels in perturbations.items():
        for level in levels:
            if verbose:
                print(f"Running {perturbation_type} @ level={level}")

            for seed in tqdm(config.seeds, desc=f"{perturbation_type}={level}", disable=not verbose):
                try:
                    # Generate fresh data
                    X, coords, labels, _, _ = generate_synthetic_data(
                        n_spots=config.n_spots,
                        n_genes=config.n_genes,
                        n_ectopic=config.n_ectopic,
                        n_intrinsic=config.n_intrinsic,
                        n_modules=config.n_modules,
                        min_distance_factor=config.min_distance_factor,
                        random_state=seed,
                    )

                    result = run_with_perturbation(
                        X, coords, labels, config, seed,
                        perturbation_type, level,
                    )
                    all_results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"  Seed {seed} failed: {e}")

    if not all_results:
        raise RuntimeError("All seeds failed in exp04_noise_robustness")
    results = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 60)
        print("Noise Robustness Summary")
        print("=" * 60)
        summary = results.groupby(["perturbation_type", "perturbation_level"]).agg({
            "Inv_PosError_auc_ectopic": ["mean", "std"],
            "PCA_Error_auc_intrinsic": ["mean", "std"],
        }).round(3)
        print(summary.to_string())

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp04_noise_robustness.csv"))
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
