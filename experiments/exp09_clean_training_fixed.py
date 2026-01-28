"""
Experiment 09: Clean Training (Fixed)

CRITICAL FIX: Previous exp08 used DIFFERENT spatial patterns for train and test.
This experiment uses the SAME spatial structure, adding anomalies only for testing.

Approach:
1. Generate base spatial pattern (normal data) using ZINB
2. Split into train/test BY SPOTS (not by regenerating)
3. Inject anomalies ONLY into test spots (on mu, before count generation)
4. Train on clean train split, evaluate on test split with anomalies

This is the correct way to test if inverse prediction works without labeled anomalies.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from config import ExperimentConfig, DEFAULT_CONFIG
from core import (
    prepare_data,
    InversePredictionModel,
    train_model,
    compute_scores,
    compute_pca_error,
    set_seed,
)


def generate_zinb_clean_and_test(
    n_spots: int = 3000,
    n_genes: int = 500,
    n_modules: int = 20,
    n_ectopic: int = 50,
    n_intrinsic: int = 100,
    min_distance_factor: float = 0.5,
    test_size: float = 0.3,
    random_state: int = 42,
    dispersion: float = 2.0,
    dropout_rate: float = 0.3,
    library_size_mean: float = 10000,
    library_size_cv: float = 0.3,
):
    """
    Generate ZINB data with clean train split and anomaly-injected test split.

    The key design: mu is shared, anomalies are injected on mu for test spots
    BEFORE ZINB count generation, so train and test come from the same spatial
    structure but with independent NB sampling.

    Returns
    -------
    X_train : log1p-normalized training data (clean)
    coords_train : training coordinates
    X_test : log1p-normalized test data (with anomalies)
    coords_test : test coordinates
    test_labels : 0=normal, 1=ectopic, 2=intrinsic
    donor_positions : donor coordinates for ectopic test spots
    """
    rng = np.random.RandomState(random_state)

    # ======================================================================
    # Step 1: Create shared spatial structure (mu matrix)
    # ======================================================================
    side = int(np.ceil(np.sqrt(n_spots)))
    x = np.tile(np.arange(side), side)[:n_spots].astype(float)
    y = np.repeat(np.arange(side), side)[:n_spots].astype(float)
    x += rng.normal(0, 0.1, n_spots)
    y += rng.normal(0, 0.1, n_spots)
    coords = np.column_stack([x, y])

    coords_norm = (coords - coords.min(axis=0)) / (
        coords.max(axis=0) - coords.min(axis=0) + 1e-8
    )

    # Gene base rates
    gene_base_rates = rng.gamma(shape=0.5, scale=2.0, size=n_genes)
    mu_matrix = np.zeros((n_spots, n_genes))
    genes_per_module = n_genes // n_modules

    for m in range(n_modules):
        center = rng.rand(2)
        distances = np.linalg.norm(coords_norm - center, axis=1)
        spatial_weight = np.exp(-(distances**2) / 0.1)

        gene_start = m * genes_per_module
        gene_end = min(gene_start + genes_per_module, n_genes)

        for g in range(gene_start, gene_end):
            spatial_effect = rng.uniform(0.5, 3.0)
            mu_matrix[:, g] = gene_base_rates[g] * (1 + spatial_effect * spatial_weight)

    # Remaining genes
    remaining_start = n_modules * genes_per_module
    for g in range(remaining_start, n_genes):
        center = rng.rand(2)
        distances = np.linalg.norm(coords_norm - center, axis=1)
        spatial_weight = np.exp(-(distances**2) / rng.uniform(0.05, 0.2))
        mu_matrix[:, g] = gene_base_rates[g] * (
            1 + rng.uniform(0.5, 2.0) * spatial_weight
        )

    # Library size variation
    library_sizes = rng.lognormal(
        mean=np.log(library_size_mean), sigma=library_size_cv, size=(n_spots, 1)
    )
    current_totals = mu_matrix.sum(axis=1, keepdims=True) + 1e-8
    mu_matrix = mu_matrix * (library_sizes / current_totals)

    # ======================================================================
    # Step 2: Split into train/test BY SPOTS
    # ======================================================================
    all_idx = np.arange(n_spots)
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=random_state
    )

    # ======================================================================
    # Step 3: Inject anomalies on mu for TEST spots only
    # ======================================================================
    mu_test = mu_matrix.copy()  # Copy to avoid modifying train mu
    n_test = len(test_idx)
    test_labels = np.zeros(n_test, dtype=int)
    donor_positions = coords[test_idx].copy()

    # Ectopic: copy mu from distant spot
    min_dist = min_distance_factor
    n_ectopic = min(n_ectopic, n_test // 4)
    ectopic_local_idx = rng.choice(n_test, n_ectopic, replace=False)
    successful_ectopic = []

    for local_idx in ectopic_local_idx:
        global_idx = test_idx[local_idx]
        distances = np.linalg.norm(
            coords_norm - coords_norm[global_idx], axis=1
        )
        distant_spots = np.where(distances > min_dist)[0]

        if len(distant_spots) > 0:
            donor = rng.choice(distant_spots)
            mu_test[global_idx] = mu_matrix[donor].copy()
            donor_positions[local_idx] = coords[donor]
            test_labels[local_idx] = 1
            successful_ectopic.append(local_idx)

    if len(successful_ectopic) < n_ectopic:
        warnings.warn(
            f"Only {len(successful_ectopic)}/{n_ectopic} ectopic anomalies injected"
        )

    # Intrinsic: boost mu for selected genes
    remaining_local = np.setdiff1d(np.arange(n_test), successful_ectopic)
    n_intrinsic = min(n_intrinsic, len(remaining_local))
    intrinsic_local_idx = rng.choice(remaining_local, n_intrinsic, replace=False)

    for local_idx in intrinsic_local_idx:
        global_idx = test_idx[local_idx]
        n_affected = rng.randint(max(20, n_genes // 7), max(50, n_genes // 3))
        affected_genes = rng.choice(n_genes, n_affected, replace=False)
        boost_factors = rng.uniform(3.0, 10.0, n_affected)
        mu_test[global_idx, affected_genes] *= boost_factors

        n_down = n_affected // 3
        down_genes = rng.choice(
            np.setdiff1d(np.arange(n_genes), affected_genes),
            min(n_down, n_genes - n_affected),
            replace=False,
        )
        mu_test[global_idx, down_genes] *= rng.uniform(0.1, 0.3, len(down_genes))
        test_labels[local_idx] = 2

    # ======================================================================
    # Step 4: Generate ZINB counts independently for train and test
    # ======================================================================
    r = dispersion

    # Train counts (from clean mu)
    mu_train = mu_matrix[train_idx]
    p_train = r / (r + mu_train + 1e-12)
    X_train_counts = rng.negative_binomial(n=r, p=p_train)

    # Test counts (from anomaly-modified mu)
    mu_test_subset = mu_test[test_idx]
    p_test = r / (r + mu_test_subset + 1e-12)
    X_test_counts = rng.negative_binomial(n=r, p=p_test)

    # Expression-dependent dropout (shared gene-level rates from original mu)
    gene_means = mu_matrix.mean(axis=0)
    gene_dropout_rates = dropout_rate * np.exp(-gene_means / gene_means.mean())
    gene_dropout_rates = np.clip(gene_dropout_rates, 0.1, 0.8)

    dropout_train = rng.random(X_train_counts.shape) < gene_dropout_rates
    X_train_counts[dropout_train] = 0

    dropout_test = rng.random(X_test_counts.shape) < gene_dropout_rates
    X_test_counts[dropout_test] = 0

    # Log-normalize
    X_train = X_train_counts.astype(float)
    X_test = X_test_counts.astype(float)

    coords_train = coords[train_idx]
    coords_test = coords[test_idx]

    return X_train, coords_train, X_test, coords_test, test_labels, donor_positions


def run_single_seed(seed: int, config: ExperimentConfig, verbose: bool = False):
    """Run clean training experiment with fixed methodology."""
    # Set all random seeds for reproducibility
    set_seed(seed)

    # 1. Generate ZINB data with clean train / anomaly test
    X_train, coords_train, X_test, coords_test, test_labels, donor_positions = (
        generate_zinb_clean_and_test(
            n_spots=config.n_spots,
            n_genes=config.n_genes,
            n_modules=config.n_modules,
            n_ectopic=int(config.n_spots * config.test_size * 0.05),
            n_intrinsic=int(config.n_spots * config.test_size * 0.10),
            min_distance_factor=config.min_distance_factor,
            test_size=config.test_size,
            random_state=seed,
        )
    )

    # 2. Prepare training data
    data_train = prepare_data(
        X_train,
        coords_train,
        k=min(config.k_neighbors, len(X_train) - 1),
        device=config.device,
    )

    # 3. Train model on CLEAN data
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

    # 4. Prepare test data
    data_test = prepare_data(
        X_test,
        coords_test,
        k=min(config.k_neighbors, len(X_test) - 1),
        device=config.device,
    )

    # 5. Compute scores on test data using CLEAN-trained model
    scores = compute_scores(
        model,
        data_test["x_tensor"],
        data_test["coords_tensor"],
        data_test["edge_index"],
        random_state=seed,
    )

    pca_error = compute_pca_error(data_test["X_norm"])

    # 6. Compute AUCs
    ectopic_mask = test_labels == 1
    intrinsic_mask = test_labels == 2
    any_anomaly = test_labels > 0

    result = {"seed": seed}

    if ectopic_mask.sum() > 0:
        result["auc_ectopic_pos"] = roc_auc_score(
            ectopic_mask.astype(int), scores["s_pos"]
        )
        result["auc_ectopic_pca"] = roc_auc_score(
            ectopic_mask.astype(int), pca_error
        )
        result["n_ectopic"] = int(ectopic_mask.sum())
    else:
        result["auc_ectopic_pos"] = np.nan
        result["auc_ectopic_pca"] = np.nan
        result["n_ectopic"] = 0

    if intrinsic_mask.sum() > 0:
        result["auc_intrinsic_pos"] = roc_auc_score(
            intrinsic_mask.astype(int), scores["s_pos"]
        )
        result["auc_intrinsic_pca"] = roc_auc_score(
            intrinsic_mask.astype(int), pca_error
        )
        result["n_intrinsic"] = int(intrinsic_mask.sum())
    else:
        result["auc_intrinsic_pos"] = np.nan
        result["auc_intrinsic_pca"] = np.nan
        result["n_intrinsic"] = 0

    if any_anomaly.sum() > 0:
        combined = np.maximum(
            (scores["s_pos"] - scores["s_pos"].min())
            / (scores["s_pos"].max() - scores["s_pos"].min() + 1e-10),
            (pca_error - pca_error.min())
            / (pca_error.max() - pca_error.min() + 1e-10),
        )
        result["auc_any_combined"] = roc_auc_score(any_anomaly.astype(int), combined)

    # 7. Also test with contaminated training for comparison
    model_contam = InversePredictionModel(
        in_dim=data_test["n_genes"],
        hid_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(config.device)

    model_contam = train_model(
        model_contam,
        data_test["x_tensor"],
        data_test["coords_tensor"],
        data_test["edge_index"],
        n_epochs=config.n_epochs,
        lr=config.learning_rate,
        lambda_pos=config.lambda_pos,
        lambda_self=config.lambda_self,
        verbose=False,
    )

    scores_contam = compute_scores(
        model_contam,
        data_test["x_tensor"],
        data_test["coords_tensor"],
        data_test["edge_index"],
        random_state=seed,
    )

    if ectopic_mask.sum() > 0:
        result["auc_ectopic_contam"] = roc_auc_score(
            ectopic_mask.astype(int), scores_contam["s_pos"]
        )

    # 8. Position prediction analysis for ectopic spots
    if ectopic_mask.sum() > 0:
        ectopic_local_idx = np.where(ectopic_mask)[0]

        # Denormalize using coords_train (the model's training space)
        coords_min = coords_train.min(axis=0)
        coords_range = np.ptp(coords_train, axis=0)
        pos_pred_denorm = scores["pos_pred"] * coords_range + coords_min

        pred_ect = pos_pred_denorm[ectopic_local_idx]
        true_ect = coords_test[ectopic_local_idx]
        donor_ect = donor_positions[ectopic_local_idx]

        dist_to_true = np.linalg.norm(pred_ect - true_ect, axis=1)
        dist_to_donor = np.linalg.norm(pred_ect - donor_ect, axis=1)

        result["fraction_closer_to_donor"] = float(
            (dist_to_donor < dist_to_true).mean()
        )
        result["mean_dist_to_true"] = float(dist_to_true.mean())
        result["mean_dist_to_donor"] = float(dist_to_donor.mean())

    return result


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run clean training experiment with fixed methodology."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []
    iterator = (
        tqdm(config.seeds, desc="Clean Training (Fixed)") if verbose else config.seeds
    )

    for seed in iterator:
        try:
            result = run_single_seed(seed, config, verbose=False)
            all_results.append(result)
        except Exception as e:
            if verbose:
                print(f"Seed {seed} failed: {e}")
            import traceback

            traceback.print_exc()

    results = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 60)
        print("Clean Training (FIXED - Same Spatial Structure, ZINB)")
        print("=" * 60)
        print(
            "\nClean Training (trained on normal ZINB data from SAME spatial pattern):"
        )
        print(
            f"  Ectopic AUC (Inv_PosError):   {results['auc_ectopic_pos'].mean():.3f} ± {results['auc_ectopic_pos'].std():.3f}"
        )
        print(
            f"  Intrinsic AUC (PCA_Error):    {results['auc_intrinsic_pca'].mean():.3f} ± {results['auc_intrinsic_pca'].std():.3f}"
        )

        if "auc_ectopic_contam" in results.columns:
            print("\nContaminated Training (for comparison):")
            print(
                f"  Ectopic AUC (Inv_PosError):   {results['auc_ectopic_contam'].mean():.3f} ± {results['auc_ectopic_contam'].std():.3f}"
            )

        if "fraction_closer_to_donor" in results.columns:
            print("\nPosition Prediction Analysis:")
            print(
                f"  Fraction closer to donor:     {results['fraction_closer_to_donor'].mean():.3f} ± {results['fraction_closer_to_donor'].std():.3f}"
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output", type=str, default=os.path.join(RESULTS_DIR, "exp09_clean_training_fixed.csv")
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
