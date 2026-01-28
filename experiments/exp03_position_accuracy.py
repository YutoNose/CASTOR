"""
Experiment 03: Position Prediction Accuracy

Validates the key hypothesis: For ectopic anomalies, the predicted
position should be closer to the DONOR location than the TRUE location.

This provides interpretability for the inverse prediction approach.
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
    generate_controlled_ectopic,
    prepare_data,
    InversePredictionModel,
    train_model,
    compute_scores,
    compute_position_accuracy,
    set_seed,
)


def run_single_seed(seed: int, config: ExperimentConfig, verbose: bool = False):
    """Run position accuracy experiment for a single seed."""
    # Set all random seeds for reproducibility
    set_seed(seed)

    # Generate data with known donor positions
    X, coords, labels, donor_positions = generate_controlled_ectopic(
        n_spots=config.n_spots,
        n_genes=config.n_genes,
        n_ectopic=config.n_ectopic,
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

    # Get predicted positions
    scores = compute_scores(
        model, data["x_tensor"], data["coords_tensor"],
        data["edge_index"], random_state=seed,
    )

    # Denormalize predicted positions
    coords_min = coords.min(axis=0)
    coords_range = coords.max(axis=0) - coords_min
    pos_pred_denorm = scores["pos_pred"] * coords_range + coords_min

    # Compute position accuracy metrics
    metrics = compute_position_accuracy(
        pos_pred_denorm, coords, donor_positions, labels
    )
    metrics["seed"] = seed

    # Additional analysis: distance ratios for ectopic spots
    ectopic_mask = labels == 1
    if ectopic_mask.sum() > 0:
        pred_ect = pos_pred_denorm[ectopic_mask]
        true_ect = coords[ectopic_mask]
        donor_ect = donor_positions[ectopic_mask]

        dist_to_true = np.linalg.norm(pred_ect - true_ect, axis=1)
        dist_to_donor = np.linalg.norm(pred_ect - donor_ect, axis=1)

        # Ratio: < 1 means closer to donor
        ratios = dist_to_donor / (dist_to_true + 1e-10)
        metrics["ratio_mean"] = float(ratios.mean())
        metrics["ratio_median"] = float(np.median(ratios))

        # What fraction predicts closer to donor?
        metrics["fraction_ratio_lt_1"] = float((ratios < 1).mean())

    # Normal spots analysis (should predict own position)
    normal_mask = labels == 0
    if normal_mask.sum() > 0:
        pred_normal = pos_pred_denorm[normal_mask]
        true_normal = coords[normal_mask]
        dist_normal = np.linalg.norm(pred_normal - true_normal, axis=1)
        metrics["normal_mean_dist"] = float(dist_normal.mean())
        metrics["normal_median_dist"] = float(np.median(dist_normal))

    return metrics


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run position accuracy experiment across all seeds."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []
    iterator = tqdm(config.seeds, desc="Position Accuracy") if verbose else config.seeds

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
        print("Position Prediction Accuracy Summary")
        print("=" * 60)
        print(f"\nEctopic anomalies:")
        print(f"  Fraction closer to donor: {results['fraction_closer_to_donor'].mean():.3f} ± {results['fraction_closer_to_donor'].std():.3f}")
        print(f"  Mean dist to true pos:    {results['mean_dist_to_true'].mean():.3f} ± {results['mean_dist_to_true'].std():.3f}")
        print(f"  Mean dist to donor pos:   {results['mean_dist_to_donor'].mean():.3f} ± {results['mean_dist_to_donor'].std():.3f}")
        print(f"  Ratio (donor/true) < 1:   {results['fraction_ratio_lt_1'].mean():.3f}")

        if "normal_mean_dist" in results.columns:
            print(f"\nNormal spots:")
            print(f"  Mean dist to true pos:    {results['normal_mean_dist'].mean():.3f} ± {results['normal_mean_dist'].std():.3f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "exp03_position_accuracy.csv"))
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
