"""
Experiment 19: Clustered Ectopic Advantage

Demonstrates that Inv_PosError maintains detection performance on
spatially clustered ectopic anomalies, while LISA (Local Moran's I)
degrades because clustered ectopic spots appear locally consistent
to their neighbors.

LISA compares each spot to its spatial neighbors. When ectopic spots
form contiguous clusters, interior spots are surrounded by other ectopic
spots with similar expression — LISA sees them as normal (positive
spatial autocorrelation). Only boundary spots are flagged.

Inv_PosError learns a global expression->position mapping and detects
all ectopic spots regardless of local context.

Independent variable: cluster_size (1, 5, 10, 25, 50, 100)
Total ectopic spots: 100 (constant), n_clusters = 100 / cluster_size
Seeds: 30 (42-71)
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
    set_seed,
)
from core.baselines import (
    compute_pca_error,
    compute_neighbor_diff,
    compute_lisa,
    compute_lof,
    compute_isolation_forest,
)
from core.data_generation import inject_clustered_ectopic

CLUSTER_SIZES = [1, 5, 10, 25, 50, 100]


def run_single_seed(
    seed: int,
    config: ExperimentConfig,
    cluster_size: int,
    verbose: bool = False,
) -> dict:
    """Run experiment for a single (seed, cluster_size) combination."""
    set_seed(seed)

    # Generate clean data (no ectopic, no intrinsic)
    X, coords, _, _, _ = generate_synthetic_data(
        n_spots=config.n_spots,
        n_genes=config.n_genes,
        n_ectopic=0,
        n_intrinsic=0,
        n_modules=config.n_modules,
        min_distance_factor=config.min_distance_factor,
        random_state=seed,
    )

    # Inject clustered ectopic anomalies
    X_injected, labels, ectopic_idx, n_ectopic_actual = inject_clustered_ectopic(
        X, coords,
        cluster_size=cluster_size,
        n_total_ectopic=config.n_ectopic,
        min_distance_factor=config.min_distance_factor,
        random_state=seed,
    )

    if n_ectopic_actual == 0:
        return None

    # Prepare data
    data = prepare_data(
        X_injected, coords,
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

    # Compute inverse prediction scores
    scores_inv = compute_scores(
        model,
        data["x_tensor"],
        data["coords_tensor"],
        data["edge_index"],
        random_state=seed,
    )

    # Compute baseline scores
    X_norm = data["X_norm"]

    y_true = (labels == 1).astype(int)

    result = {
        "seed": seed,
        "cluster_size": cluster_size,
        "n_ectopic_actual": n_ectopic_actual,
        "auc_inv_pos": roc_auc_score(y_true, scores_inv["s_pos"]),
        "auc_lisa": roc_auc_score(y_true, compute_lisa(X_norm, coords)),
        "auc_neighbor_diff": roc_auc_score(y_true, compute_neighbor_diff(X_norm, coords)),
        "auc_pca_error": roc_auc_score(y_true, compute_pca_error(X_norm)),
        "auc_lof": roc_auc_score(y_true, compute_lof(X_norm)),
    }

    return result


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """
    Run clustered ectopic experiment across all cluster sizes and seeds.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    verbose : bool
        Print progress

    Returns
    -------
    results : pd.DataFrame
        AUC metrics for all methods across cluster sizes and seeds
    """
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []
    total = len(CLUSTER_SIZES) * len(config.seeds)

    if verbose:
        print("=" * 60)
        print("Experiment 19: Clustered Ectopic Advantage")
        print("=" * 60)
        print(f"Cluster sizes: {CLUSTER_SIZES}")
        print(f"Seeds: {len(config.seeds)}")
        print(f"Total runs: {total}")

    pbar = tqdm(total=total, desc="Exp19") if verbose else None

    for cluster_size in CLUSTER_SIZES:
        for seed in config.seeds:
            try:
                result = run_single_seed(seed, config, cluster_size)
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"\ncluster_size={cluster_size}, seed={seed} failed: {e}")

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    if not all_results:
        raise RuntimeError("All runs failed")

    results = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 60)
        print("Summary: Ectopic Detection AUC by Cluster Size")
        print("=" * 60)
        auc_cols = [c for c in results.columns if c.startswith("auc_")]
        for cs in CLUSTER_SIZES:
            sub = results[results["cluster_size"] == cs]
            if len(sub) == 0:
                continue
            print(f"\n  cluster_size={cs} (n={len(sub)}):")
            for col in auc_cols:
                method = col.replace("auc_", "")
                vals = sub[col].dropna()
                print(f"    {method:20s}: {vals.mean():.3f} +/- {vals.std():.3f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp19: Clustered Ectopic Advantage")
    parser.add_argument("--quick", action="store_true", help="Quick test (3 seeds)")
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(RESULTS_DIR, "exp19_clustered_ectopic.csv"),
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
