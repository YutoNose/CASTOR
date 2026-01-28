"""
Experiment 11: Real Data Validation

Validates inverse prediction on real spatial transcriptomics data:
1. 10x Genomics Visium public datasets
2. Semi-synthetic validation (real data + artificial ectopic injection)

Biological significance:
- Ectopic cells = cells with expression profile from a different spatial location
- Examples: tumor invasion, immune cell infiltration, developmental heterotopia

This experiment validates that the method works on real ST data structure,
not just synthetic spatial patterns.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from config import ExperimentConfig, DEFAULT_CONFIG, QUICK_CONFIG
from core import (
    prepare_data,
    InversePredictionModel,
    train_model,
    compute_scores,
    compute_pca_error,
    compute_neighbor_diff,
    compute_lof,
    set_seed,
)
from core.real_data import (
    download_visium_dataset,
    inject_ectopic_by_distance,
    inject_ectopic_by_region,
    cluster_spots,
    filter_genes,
    filter_spots,
    subsample_genes,
    VISIUM_DATASETS,
)


def run_semi_synthetic_validation(
    dataset_name: str,
    config: ExperimentConfig,
    n_ectopic: int = 100,
    noise_level: float = 0.0,
    mix_alpha: float = 1.0,
    use_clustering: bool = True,
    n_clusters: int = 7,
    seeds: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run semi-synthetic validation on real Visium data.

    1. Load real ST data
    2. Optionally cluster spots to define regions
    3. Inject artificial ectopic anomalies
    4. Train inverse prediction model (clean training)
    5. Evaluate detection performance

    Parameters
    ----------
    dataset_name : str
        Visium dataset name
    config : ExperimentConfig
        Experiment configuration
    n_ectopic : int
        Number of ectopic spots to inject
    noise_level : float
        Noise to add to ectopic expression
    mix_alpha : float
        Mixing ratio (1.0 = full copy)
    use_clustering : bool
        Use clustering to define regions for injection
    n_clusters : int
        Number of clusters for region definition
    seeds : list
        Random seeds
    verbose : bool
        Print progress

    Returns
    -------
    results : pd.DataFrame
        Experiment results
    """
    if seeds is None:
        seeds = config.seeds

    # Load real data
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'=' * 60}")

    X_raw, coords, gene_names = download_visium_dataset(dataset_name)

    # Preprocess
    X_filtered, gene_names_filtered = filter_genes(X_raw, gene_names, min_cells=10)
    X_filtered, coords_filtered, keep_idx = filter_spots(
        X_filtered, coords, min_genes=200, min_counts=500
    )

    # Subsample genes for computational efficiency
    X_sub, gene_names_sub = subsample_genes(
        X_filtered, gene_names_filtered, n_genes=2000
    )

    if verbose:
        print(f"After preprocessing: {X_sub.shape[0]} spots, {X_sub.shape[1]} genes")

    # Define regions
    if use_clustering:
        if verbose:
            print(f"Clustering spots into {n_clusters} regions...")
        region_labels = cluster_spots(
            X_sub, coords_filtered, n_clusters=n_clusters, random_state=42
        )
    else:
        region_labels = None

    all_results = []

    for seed in tqdm(seeds, desc=f"{dataset_name}", disable=not verbose):
        try:
            result = run_single_seed(
                X=X_sub,
                coords=coords_filtered,
                region_labels=region_labels,
                n_ectopic=n_ectopic,
                noise_level=noise_level,
                mix_alpha=mix_alpha,
                seed=seed,
                config=config,
                use_region_injection=use_clustering,
            )
            result["dataset"] = dataset_name
            result["n_spots"] = len(X_sub)
            result["n_genes"] = X_sub.shape[1]
            all_results.append(result)

        except Exception as e:
            if verbose:
                print(f"  Seed {seed} failed: {e}")

    return pd.DataFrame(all_results)


def run_single_seed(
    X: np.ndarray,
    coords: np.ndarray,
    region_labels: np.ndarray,
    n_ectopic: int,
    noise_level: float,
    mix_alpha: float,
    seed: int,
    config: ExperimentConfig,
    use_region_injection: bool = True,
) -> dict:
    """Run single seed experiment."""
    # Set all random seeds for reproducibility (numpy + torch)
    set_seed(seed)

    # Split into train/test (70%/30%)
    n_spots = len(X)
    indices = np.random.permutation(n_spots)
    n_train = int(n_spots * 0.7)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # Train data: clean (no anomalies)
    X_train = X[train_idx]
    coords_train = coords[train_idx]

    # Test data: inject ectopic anomalies
    X_test = X[test_idx].copy()
    coords_test = coords[test_idx]

    if use_region_injection and region_labels is not None:
        region_labels_test = region_labels[test_idx]
        X_test, labels_test, donor_positions, metadata = inject_ectopic_by_region(
            X_test, coords_test, region_labels_test,
            n_ectopic=n_ectopic,
            noise_level=noise_level,
            mix_alpha=mix_alpha,
            random_state=seed,
        )
    else:
        X_test, labels_test, donor_positions, metadata = inject_ectopic_by_distance(
            X_test, coords_test,
            n_ectopic=n_ectopic,
            min_distance_fraction=0.3,
            noise_level=noise_level,
            mix_alpha=mix_alpha,
            random_state=seed,
        )

    # Prepare training data
    k_train = min(config.k_neighbors, len(train_idx) - 1)
    data_train = prepare_data(X_train, coords_train, k=k_train, device=config.device)

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
        verbose=False,
    )

    # Prepare test data
    k_test = min(config.k_neighbors, len(test_idx) - 1)
    data_test = prepare_data(X_test, coords_test, k=k_test, device=config.device)

    # Compute scores
    scores = compute_scores(
        model,
        data_test["x_tensor"],
        data_test["coords_tensor"],
        data_test["edge_index"],
        random_state=seed,
    )
    pca_error = compute_pca_error(data_test["X_norm"])
    neighbor_diff = compute_neighbor_diff(data_test["X_norm"], coords_test, k=k_test)
    lof_score = compute_lof(data_test["X_norm"], n_neighbors=20)

    # Evaluate detection
    ectopic_mask = labels_test == 1
    n_ectopic_actual = ectopic_mask.sum()

    result = {
        "seed": seed,
        "n_ectopic": int(n_ectopic_actual),
        "n_test": len(test_idx),
        "noise_level": noise_level,
        "mix_alpha": mix_alpha,
    }

    if n_ectopic_actual > 0 and (~ectopic_mask).sum() > 0:
        result["auc_pos"] = roc_auc_score(ectopic_mask.astype(int), scores["s_pos"])
        result["auc_pca"] = roc_auc_score(ectopic_mask.astype(int), pca_error)
        result["auc_neighbor"] = roc_auc_score(ectopic_mask.astype(int), neighbor_diff)
        result["auc_lof"] = roc_auc_score(ectopic_mask.astype(int), lof_score)

        # Position prediction analysis
        coords_min = coords_test.min(axis=0)
        coords_range = np.ptp(coords_test, axis=0) + 1e-8
        pos_pred_denorm = scores["pos_pred"] * coords_range + coords_min

        ectopic_local = np.where(ectopic_mask)[0]
        pred_ect = pos_pred_denorm[ectopic_local]
        true_ect = coords_test[ectopic_local]
        donor_ect = donor_positions[ectopic_local]

        dist_to_true = np.linalg.norm(pred_ect - true_ect, axis=1)
        dist_to_donor = np.linalg.norm(pred_ect - donor_ect, axis=1)

        result["fraction_closer_to_donor"] = float((dist_to_donor < dist_to_true).mean())
        result["mean_dist_to_true"] = float(dist_to_true.mean())
        result["mean_dist_to_donor"] = float(dist_to_donor.mean())
    else:
        result["auc_pos"] = np.nan
        result["auc_pca"] = np.nan
        result["auc_neighbor"] = np.nan
        result["auc_lof"] = np.nan
        result["fraction_closer_to_donor"] = np.nan

    return result


def run_difficulty_sweep(
    dataset_name: str,
    config: ExperimentConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run experiments with varying difficulty levels.

    Sweeps over:
    - noise_level: [0.0, 0.1, 0.2, 0.3]
    - mix_alpha: [1.0, 0.7, 0.5]
    """
    difficulty_configs = [
        # Easy: exact copy
        {"noise_level": 0.0, "mix_alpha": 1.0, "difficulty": "easy"},
        # Medium: some noise
        {"noise_level": 0.1, "mix_alpha": 1.0, "difficulty": "medium_noise"},
        # Medium: partial mix
        {"noise_level": 0.0, "mix_alpha": 0.7, "difficulty": "medium_mix"},
        # Hard: noise + partial mix
        {"noise_level": 0.1, "mix_alpha": 0.7, "difficulty": "hard"},
        # Hardest
        {"noise_level": 0.2, "mix_alpha": 0.5, "difficulty": "hardest"},
    ]

    all_results = []

    for diff_config in difficulty_configs:
        if verbose:
            print(f"\n--- Difficulty: {diff_config['difficulty']} ---")

        results = run_semi_synthetic_validation(
            dataset_name=dataset_name,
            config=config,
            noise_level=diff_config["noise_level"],
            mix_alpha=diff_config["mix_alpha"],
            verbose=verbose,
        )
        results["difficulty"] = diff_config["difficulty"]
        all_results.append(results)

    return pd.concat(all_results, ignore_index=True)


def run(config: ExperimentConfig = None, datasets: list = None, verbose: bool = True) -> pd.DataFrame:
    """
    Run real data validation experiments.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration
    datasets : list
        List of dataset names to test
    verbose : bool
        Print progress

    Returns
    -------
    results : pd.DataFrame
        All results
    """
    if config is None:
        config = DEFAULT_CONFIG

    if datasets is None:
        # Default: human lymph node (most accessible via scanpy)
        datasets = ["human_lymph_node"]

    all_results = []

    for dataset_name in datasets:
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'=' * 80}")

        # Run with different difficulty levels
        results = run_difficulty_sweep(dataset_name, config, verbose=verbose)
        all_results.append(results)

    final_results = pd.concat(all_results, ignore_index=True)

    if verbose:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)

        summary = final_results.groupby(["dataset", "difficulty"]).agg({
            "auc_pos": ["mean", "std"],
            "auc_pca": ["mean", "std"],
            "fraction_closer_to_donor": ["mean", "std"],
        }).round(3)

        print("\nEctopic Detection (Inv_PosError AUC):")
        for dataset in datasets:
            print(f"\n  {dataset}:")
            for _, row in final_results[final_results["dataset"] == dataset].groupby("difficulty").agg({
                "auc_pos": ["mean", "std"],
                "fraction_closer_to_donor": ["mean"],
            }).iterrows():
                diff = _
                mean = row[("auc_pos", "mean")]
                std = row[("auc_pos", "std")]
                donor = row[("fraction_closer_to_donor", "mean")]
                print(f"    {diff:15s}: {mean:.3f} ± {std:.3f} (donor: {donor:.1%})")

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real data validation for inverse prediction")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=list(VISIUM_DATASETS.keys()),
                        help="Datasets to test")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp11_real_data_validation.csv"),
                        help="Output file path")
    args = parser.parse_args()

    if args.quick:
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    results = run(config, datasets=args.datasets, verbose=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
