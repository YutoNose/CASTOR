"""
Experiment 10: Multi-Scenario Validation

Tests inverse prediction across multiple synthetic data scenarios to ensure
robustness and identify limitations.

Training: UNSUPERVISED — model trains on ALL spots (normal + anomalies).
This is the only setting applicable to real data where labels are unknown.

Scenarios:
- baseline: Current exact copy (easy)
- noisy_ectopic: Add noise to copied expression
- partial_ectopic: Mix of original and donor (70%)
- hard_ectopic: Mix (50%) + noise
- medium_intrinsic: Smaller effect size
- hard_intrinsic: Very small effect size
- realistic_counts: Negative binomial noise
- cell_type_based: Cell type marker swapping
- hardest: All difficulty factors combined
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
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
from core.scenarios import SCENARIOS, generate_scenario_data, ScenarioConfig


def run_scenario(
    scenario_name: str,
    scenario: ScenarioConfig,
    seed: int,
    config: ExperimentConfig,
    verbose: bool = False,
) -> dict:
    """Run a single scenario with unsupervised training (all data)."""
    # Set all random seeds for reproducibility (numpy + torch)
    set_seed(seed)

    # Generate data for this scenario
    X, coords, labels, ectopic_idx, intrinsic_idx, metadata = generate_scenario_data(
        scenario=scenario,
        n_spots=config.n_spots,
        n_genes=config.n_genes,
        n_ectopic=config.n_ectopic,
        n_intrinsic=config.n_intrinsic,
        min_distance_factor=config.min_distance_factor,
        random_state=seed,
    )

    # UNSUPERVISED: train on ALL data (no label-based filtering)
    data = prepare_data(X, coords, k=config.k_neighbors, device=config.device)

    # Train model on all spots
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

    # Compute scores on ALL data (same as training — unsupervised)
    scores = compute_scores(
        model, data["x_tensor"], data["coords_tensor"],
        data["edge_index"], random_state=seed,
    )
    pca_error = compute_pca_error(data["X_norm"])

    # Compute metrics using labels (for evaluation only, NOT used in training)
    ectopic_mask = labels == 1
    intrinsic_mask = labels == 2

    result = {
        "scenario": scenario_name,
        "scenario_full_name": scenario.name,
        "seed": seed,
        "n_ectopic": int(ectopic_mask.sum()),
        "n_intrinsic": int(intrinsic_mask.sum()),
        "n_total": len(X),
        "ectopic_type": scenario.ectopic_type.value,
        "intrinsic_type": scenario.intrinsic_type.value,
        "noise_model": "zinb",
    }

    # Ectopic detection
    if ectopic_mask.sum() > 0 and (~ectopic_mask).sum() > 0:
        result["auc_ectopic_pos"] = roc_auc_score(ectopic_mask.astype(int), scores["s_pos"])
        result["auc_ectopic_pca"] = roc_auc_score(ectopic_mask.astype(int), pca_error)
        result["auc_ectopic_neighbor"] = roc_auc_score(ectopic_mask.astype(int), scores["s_neighbor"])
    else:
        result["auc_ectopic_pos"] = np.nan
        result["auc_ectopic_pca"] = np.nan
        result["auc_ectopic_neighbor"] = np.nan

    # Intrinsic detection
    if intrinsic_mask.sum() > 0 and (~intrinsic_mask).sum() > 0:
        result["auc_intrinsic_pos"] = roc_auc_score(intrinsic_mask.astype(int), scores["s_pos"])
        result["auc_intrinsic_pca"] = roc_auc_score(intrinsic_mask.astype(int), pca_error)
    else:
        result["auc_intrinsic_pos"] = np.nan
        result["auc_intrinsic_pca"] = np.nan

    return result


def run(config: ExperimentConfig = None, scenarios: list = None, verbose: bool = True) -> pd.DataFrame:
    """Run multi-scenario experiment."""
    if config is None:
        config = DEFAULT_CONFIG

    if scenarios is None:
        scenarios = list(SCENARIOS.keys())

    all_results = []

    for scenario_name in scenarios:
        if scenario_name not in SCENARIOS:
            print(f"Unknown scenario: {scenario_name}")
            continue

        scenario = SCENARIOS[scenario_name]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Scenario: {scenario.name}")
            print(f"  Ectopic: {scenario.ectopic_type.value}")
            print(f"  Intrinsic: {scenario.intrinsic_type.value}")
            print(f"{'=' * 60}")

        iterator = tqdm(config.seeds, desc=scenario_name) if verbose else config.seeds

        for seed in iterator:
            try:
                result = run_scenario(scenario_name, scenario, seed, config, verbose=False)
                all_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Seed {seed} failed: {e}")

    results = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 80)
        print("Multi-Scenario Summary (Unsupervised Training)")
        print("=" * 80)

        summary = results.groupby("scenario").agg({
            "auc_ectopic_pos": ["mean", "std"],
            "auc_ectopic_pca": ["mean", "std"],
            "auc_intrinsic_pos": ["mean", "std"],
            "auc_intrinsic_pca": ["mean", "std"],
        }).round(3)

        print("\nEctopic Detection AUC:")
        for scenario in scenarios:
            if scenario in summary.index:
                pos_m = summary.loc[scenario, ("auc_ectopic_pos", "mean")]
                pos_s = summary.loc[scenario, ("auc_ectopic_pos", "std")]
                pca_m = summary.loc[scenario, ("auc_ectopic_pca", "mean")]
                print(f"  {scenario:20s}: Inv_Pos={pos_m:.3f}±{pos_s:.3f}  PCA={pca_m:.3f}")

        print("\nIntrinsic Detection AUC:")
        for scenario in scenarios:
            if scenario in summary.index:
                pos_m = summary.loc[scenario, ("auc_intrinsic_pos", "mean")]
                pca_m = summary.loc[scenario, ("auc_intrinsic_pca", "mean")]
                pca_s = summary.loc[scenario, ("auc_intrinsic_pca", "std")]
                print(f"  {scenario:20s}: Inv_Pos={pos_m:.3f}  PCA={pca_m:.3f}±{pca_s:.3f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Specific scenarios to run")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp10_multi-scenario_validation.csv"))
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    results = run(config, scenarios=args.scenarios, verbose=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
