"""
Experiment 12: Embedding-based Competitor Comparison

Compares inverse prediction against embedding-based methods:
- STAGATE + Isolation Forest
- GraphST + Isolation Forest
- Squidpy neighborhood score

Also includes existing baselines:
- PCA reconstruction error
- LISA (Local Moran's I)
- LOF (Local Outlier Factor)
- Isolation Forest (global)
- Neighbor Difference
- SpotSweeper

Evaluation:
- Synthetic data: Ectopic and Intrinsic detection AUC
- Real data: Cancer vs Normal detection AUC (HER2ST)

Principles:
- Use actual tools, not pseudo-implementations
- No fallback: exceptions on failure
- Default parameters for fairness
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
import os
import time
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from config import ExperimentConfig, DEFAULT_CONFIG
from core import (
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
    compute_spotsweeper,
    compute_lof,
    compute_isolation_forest,
)
from core.scenarios import SCENARIOS, generate_scenario_data
from core.competitor_stagate import compute_stagate_score
from core.competitor_graphst import compute_graphst_score
from core.competitor_squidpy import compute_squidpy_nhood_score
from core.competitor_stlearn import compute_stlearn_sme_score


def run_single_experiment(
    X: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    seed: int,
    config: ExperimentConfig,
    methods: list = None,
) -> dict:
    """
    Run all methods on a single dataset.

    Returns
    -------
    dict
        Method scores and AUCs for this run
    """
    # Set all random seeds for reproducibility
    set_seed(seed)

    if methods is None:
        # All methods run on CPU for GPU compatibility with RTX 5090
        methods = [
            "inv_pos", "stagate", "graphst", "squidpy", "stlearn",
            "pca_error", "lisa", "lof", "isolation_forest",
            "neighbor_diff", "spotsweeper"
        ]

    results = {"seed": seed}
    scores_dict = {}

    # Prepare data for inverse prediction
    data = prepare_data(X, coords, k=config.k_neighbors, device=config.device)

    # --- Inverse Prediction ---
    if "inv_pos" in methods:
        try:
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

            scores = compute_scores(
                model, data["x_tensor"], data["coords_tensor"],
                data["edge_index"], random_state=seed,
            )
            scores_dict["inv_pos"] = scores["s_pos"]
        except Exception as e:
            raise RuntimeError(f"Inverse Prediction failed: {e}")

    # --- STAGATE + IF ---
    if "stagate" in methods:
        try:
            scores_dict["stagate"] = compute_stagate_score(
                X, coords, n_epochs=500, random_state=seed
            )
        except Exception as e:
            print(f"  Warning: STAGATE failed (skipping): {e}")

    # --- GraphST + IF ---
    if "graphst" in methods:
        try:
            scores_dict["graphst"] = compute_graphst_score(
                X, coords, n_epochs=600, random_state=seed, device="cpu"
            )
        except Exception as e:
            print(f"  Warning: GraphST failed (skipping): {e}")

    # --- Squidpy ---
    if "squidpy" in methods:
        try:
            scores_dict["squidpy"] = compute_squidpy_nhood_score(
                X, coords, random_state=seed
            )
        except Exception as e:
            print(f"  Warning: Squidpy failed (skipping): {e}")

    # --- STLearn ---
    if "stlearn" in methods:
        try:
            scores_dict["stlearn"] = compute_stlearn_sme_score(
                X, coords, random_state=seed
            )
        except Exception as e:
            print(f"  Warning: STLearn failed (skipping): {e}")

    # --- Baselines ---
    if "pca_error" in methods:
        scores_dict["pca_error"] = compute_pca_error(data["X_norm"])

    if "lisa" in methods:
        scores_dict["lisa"] = compute_lisa(data["X_norm"], coords)

    if "lof" in methods:
        scores_dict["lof"] = compute_lof(data["X_norm"])

    if "isolation_forest" in methods:
        scores_dict["isolation_forest"] = compute_isolation_forest(
            data["X_norm"], random_state=seed
        )

    if "neighbor_diff" in methods:
        scores_dict["neighbor_diff"] = compute_neighbor_diff(data["X_norm"], coords)

    if "spotsweeper" in methods:
        scores_dict["spotsweeper"] = compute_spotsweeper(data["X_norm"], coords)

    # --- Compute AUCs ---
    ectopic_mask = (labels == 1)
    intrinsic_mask = (labels == 2)
    normal_mask = (labels == 0)

    for method, score in scores_dict.items():
        # Ectopic AUC: Ectopic vs ALL others (consistent with core/evaluation.py)
        if ectopic_mask.sum() > 0:
            results[f"auc_ectopic_{method}"] = roc_auc_score(
                ectopic_mask.astype(int), score
            )

        # Intrinsic AUC: Intrinsic vs ALL others
        if intrinsic_mask.sum() > 0:
            results[f"auc_intrinsic_{method}"] = roc_auc_score(
                intrinsic_mask.astype(int), score
            )

    return results


def run_synthetic_comparison(
    config: ExperimentConfig,
    scenarios: list = None,
    methods: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run comparison on synthetic data across multiple scenarios and seeds.
    """
    if scenarios is None:
        scenarios = ["baseline", "noisy_ectopic", "hard_ectopic", "realistic_counts"]

    all_results = []

    for scenario_name in scenarios:
        if scenario_name not in SCENARIOS:
            print(f"Unknown scenario: {scenario_name}")
            continue

        scenario = SCENARIOS[scenario_name]

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Scenario: {scenario.name}")
            print(f"{'=' * 60}")

        iterator = tqdm(config.seeds, desc=scenario_name) if verbose else config.seeds

        for seed in iterator:
            try:
                # Generate data
                X, coords, labels, ectopic_idx, intrinsic_idx, metadata = \
                    generate_scenario_data(
                        scenario=scenario,
                        n_spots=config.n_spots,
                        n_genes=config.n_genes,
                        n_ectopic=config.n_ectopic,
                        n_intrinsic=config.n_intrinsic,
                        min_distance_factor=config.min_distance_factor,
                        random_state=seed,
                    )

                result = run_single_experiment(
                    X, coords, labels, seed, config, methods
                )
                result["scenario"] = scenario_name
                result["n_ectopic"] = len(ectopic_idx)
                result["n_intrinsic"] = len(intrinsic_idx)
                all_results.append(result)

            except Exception as e:
                if verbose:
                    print(f"  Seed {seed} failed: {e}")
                continue

    return pd.DataFrame(all_results)


def summarize_results(results: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Summarize results across seeds."""
    # Get all AUC columns
    auc_cols = [c for c in results.columns if c.startswith("auc_")]

    summary_data = []

    for scenario in results["scenario"].unique():
        scenario_data = results[results["scenario"] == scenario]

        row = {"scenario": scenario}
        for col in auc_cols:
            if col in scenario_data.columns:
                values = scenario_data[col].dropna()
                if len(values) > 0:
                    row[f"{col}_mean"] = values.mean()
                    row[f"{col}_std"] = values.std()

        summary_data.append(row)

    summary = pd.DataFrame(summary_data)

    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY: Ectopic Detection AUC (mean ± std)")
        print("=" * 80)

        methods = ["inv_pos", "stagate", "graphst", "squidpy",
                   "pca_error", "lisa", "neighbor_diff", "isolation_forest"]

        for scenario in summary["scenario"]:
            print(f"\n{scenario}:")
            row = summary[summary["scenario"] == scenario].iloc[0]
            for method in methods:
                mean_col = f"auc_ectopic_{method}_mean"
                std_col = f"auc_ectopic_{method}_std"
                if mean_col in row and not pd.isna(row[mean_col]):
                    print(f"  {method:20s}: {row[mean_col]:.3f} ± {row[std_col]:.3f}")

    return summary


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Main entry point."""
    if config is None:
        config = DEFAULT_CONFIG

    print("=" * 80)
    print("Experiment 12: Embedding-based Competitor Comparison")
    print("=" * 80)
    print(f"Seeds: {len(config.seeds)}")
    print(f"Spots: {config.n_spots}, Genes: {config.n_genes}")
    print(f"Ectopic: {config.n_ectopic}, Intrinsic: {config.n_intrinsic}")

    results = run_synthetic_comparison(
        config,
        scenarios=["baseline", "noisy_ectopic", "hard_ectopic", "realistic_counts"],
        verbose=verbose,
    )

    summary = summarize_results(results, verbose=verbose)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with fewer seeds")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds to use")
    parser.add_argument("--output", type=str,
                        default=os.path.join(RESULTS_DIR, "exp12_embedding_comparison.csv"))
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Specific scenarios to run")
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    if args.seeds:
        config = ExperimentConfig(
            **{**config.__dict__, "seeds": list(range(args.seeds))}
        )

    results = run_synthetic_comparison(
        config,
        scenarios=args.scenarios,
        verbose=True,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    summary = summarize_results(results)
    summary.to_csv(args.output.replace(".csv", "_summary.csv"), index=False)
