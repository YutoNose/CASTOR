"""
Experiment 14: HER2ST Real Data Validation

Validates anomaly detection on HER2-positive breast cancer spatial
transcriptomics data with pathologist annotations as ground truth.

Ground Truth:
- Anomaly: invasive cancer, DCIS (ductal carcinoma in situ)
- Normal: connective tissue, fat, etc.

Note: Ground truth is spatially clustered (tumor regions), not scattered.

Methods:
- Inv_PosError (ours)
- STAGATE + IF
- GraphST + IF
- Squidpy neighborhood
- PCA reconstruction error
- LISA
- Neighbor Difference
- LOF, Isolation Forest
- SpotSweeper

Principles:
- Use actual tools, not pseudo-implementations
- No fallback: exceptions on failure
- Default parameters for fairness
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Add experiment dir first, then benchmarks
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP_DIR)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Import from experiment config first
from config import ExperimentConfig, DEFAULT_CONFIG

# EXP_DIR already points to /home/yutonose/Projects which contains core/ and data/

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
from core.competitor_stagate import compute_stagate_score
from core.competitor_graphst import compute_graphst_score
from core.competitor_squidpy import compute_squidpy_nhood_score
from core.competitor_stlearn import compute_stlearn_sme_score

from data.generators.her2st import HER2STDataLoader


def run_her2st_sample(
    X,
    coords,
    y_true,
    metadata,
    seed: int,
    config: ExperimentConfig,
    methods: list = None,
) -> dict:
    """
    Run all methods on a single HER2ST sample.

    Parameters
    ----------
    X : sparse matrix or ndarray
        Expression matrix [n_spots, n_genes]
    coords : ndarray
        Spatial coordinates [n_spots, 2]
    y_true : ndarray
        Binary labels (1=cancer, 0=normal)
    metadata : dict
        Sample metadata
    seed : int
        Random seed
    config : ExperimentConfig
        Experiment configuration
    methods : list
        Methods to run

    Returns
    -------
    dict
        Method scores and AUCs for this sample
    """
    if methods is None:
        methods = [
            "inv_pos", "graphst", "squidpy", "stlearn",
            "pca_error", "lisa", "lof", "isolation_forest",
            "neighbor_diff", "spotsweeper",
            # "stagate",  # Excluded: TensorFlow CUDA incompatibility with RTX 5090
        ]

    # Set all random seeds for reproducibility
    set_seed(seed)

    # Convert sparse to dense if needed
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()

    results = {
        "sample_id": metadata["sample_id"],
        "seed": seed,
        "n_spots": len(y_true),
        "n_cancer": int(y_true.sum()),
        "cancer_fraction": float(y_true.sum() / len(y_true)),
    }
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
            raise RuntimeError(f"STAGATE failed: {e}")

    # --- GraphST + IF ---
    if "graphst" in methods:
        try:
            scores_dict["graphst"] = compute_graphst_score(
                X, coords, n_epochs=600, random_state=seed, device="cpu"
            )
        except Exception as e:
            raise RuntimeError(f"GraphST failed: {e}")

    # --- Squidpy ---
    if "squidpy" in methods:
        try:
            scores_dict["squidpy"] = compute_squidpy_nhood_score(
                X, coords, random_state=seed
            )
        except Exception as e:
            raise RuntimeError(f"Squidpy failed: {e}")

    # --- STLearn ---
    if "stlearn" in methods:
        try:
            scores_dict["stlearn"] = compute_stlearn_sme_score(
                X, coords, random_state=seed
            )
        except Exception as e:
            raise RuntimeError(f"STLearn failed: {e}")

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

    # --- Compute metrics ---
    for method, score in scores_dict.items():
        # Cancer vs Normal AUC
        if y_true.sum() > 0 and (1 - y_true).sum() > 0:
            results[f"auc_{method}"] = roc_auc_score(y_true, score)
            results[f"auprc_{method}"] = average_precision_score(y_true, score)

    return results


def run_her2st_validation(
    her2st_dir: str,
    config: ExperimentConfig = None,
    sample_ids: list = None,
    methods: list = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run validation on HER2ST dataset.

    Parameters
    ----------
    her2st_dir : str
        Path to HER2ST data directory
    config : ExperimentConfig
        Experiment configuration
    sample_ids : list
        Specific samples to run (None = all)
    methods : list
        Methods to run
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Results for all samples
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Load HER2ST data
    loader = HER2STDataLoader(her2st_dir)

    if sample_ids is None:
        sample_ids = loader.available_samples

    if verbose:
        print("=" * 80)
        print("Experiment 14: HER2ST Real Data Validation")
        print("=" * 80)
        print(f"Available samples: {loader.available_samples}")
        print(f"Running on: {sample_ids}")
        print(f"Seeds: {len(config.seeds)}")

    all_results = []

    for sample_id in sample_ids:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Sample: {sample_id}")
            print("=" * 60)

        try:
            # Load sample
            X, coords, y_true, metadata = loader.load(sample_id)

            if verbose:
                print(f"  Spots: {metadata['n_spots']}")
                print(f"  Genes: {metadata['n_genes']}")
                print(f"  Cancer fraction: {metadata['anomaly_fraction']:.1%}")
                print(f"  Labels: {metadata['unique_labels']}")

            # Run with multiple seeds
            iterator = tqdm(config.seeds, desc=sample_id) if verbose else config.seeds

            for seed in iterator:
                try:
                    result = run_her2st_sample(
                        X, coords, y_true, metadata, seed, config, methods
                    )
                    all_results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"  Seed {seed} failed: {e}")
                    raise  # No fallback

        except Exception as e:
            if verbose:
                print(f"  Failed to load sample: {e}")
            raise  # No fallback

    return pd.DataFrame(all_results)


def summarize_her2st_results(
    results: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Summarize HER2ST results across samples and seeds."""
    auc_cols = [c for c in results.columns if c.startswith("auc_")]

    # Per-sample summary
    sample_summary = results.groupby("sample_id").agg({
        **{col: ["mean", "std"] for col in auc_cols},
        "cancer_fraction": "first",
    }).round(3)

    # Overall summary
    overall_summary = {}
    for col in auc_cols:
        method = col.replace("auc_", "")
        values = results[col].dropna()
        if len(values) > 0:
            overall_summary[method] = {
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
            }

    overall_df = pd.DataFrame(overall_summary).T

    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY: Cancer Detection AUC (mean ± std)")
        print("=" * 80)

        print("\nOverall (all samples, all seeds):")
        for method, row in overall_df.iterrows():
            print(f"  {method:20s}: {row['mean']:.3f} ± {row['std']:.3f} "
                  f"[{row['min']:.3f}, {row['max']:.3f}]")

        print("\nPer-sample performance:")
        for sample_id in results["sample_id"].unique():
            sample_data = results[results["sample_id"] == sample_id]
            print(f"\n  {sample_id} (cancer: {sample_data['cancer_fraction'].iloc[0]:.1%}):")
            for col in auc_cols:
                method = col.replace("auc_", "")
                values = sample_data[col].dropna()
                if len(values) > 0:
                    print(f"    {method:20s}: {values.mean():.3f} ± {values.std():.3f}")

    return overall_df


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Main entry point."""
    if config is None:
        config = DEFAULT_CONFIG

    her2st_dir = config.her2st_dir

    results = run_her2st_validation(
        her2st_dir,
        config=config,
        verbose=verbose,
    )

    summarize_her2st_results(results, verbose=verbose)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode with fewer seeds")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds to use")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="Specific samples to run")
    parser.add_argument("--her2st-dir", type=str,
                        default=DEFAULT_CONFIG.her2st_dir)
    parser.add_argument("--output", type=str,
                        default=os.path.join(RESULTS_DIR, "exp14_her2st_validation.csv"))
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

    results = run_her2st_validation(
        args.her2st_dir,
        config=config,
        sample_ids=args.samples,
        verbose=True,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    summary = summarize_her2st_results(results)
    summary.to_csv(args.output.replace(".csv", "_summary.csv"))
