"""
Experiment 17: HER2ST Tumor-to-Normal Transplantation

Creates ground-truth ectopic anomalies by transplanting tumor spot expression
profiles into normal tissue regions using pathologist annotations.

Protocol:
1. Load HER2ST sample with pathologist annotations
2. Identify normal tissue spots (connective tissue, fat, breast glands)
3. Identify tumor spots (invasive cancer, DCIS)
4. Select a subset of normal spots as "recipients"
5. Replace their expression with tumor spot expression (= ectopic injection)
6. Train inverse prediction model on the modified data (unsupervised)
7. Evaluate: can the model detect the transplanted spots?

This creates REAL ectopic anomalies: tumor expression at normal positions.

No fabricated data. No fallbacks.
"""

import gzip
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os
import warnings
import torch

warnings.filterwarnings('ignore')

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP_DIR)
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
    compute_lof,
    compute_isolation_forest,
)

# EXP_DIR already points to /home/yutonose/Projects which contains data/
from data.generators.her2st import HER2STDataLoader


def transplant_tumor_to_normal(
    X: np.ndarray,
    coords: np.ndarray,
    labels_series: pd.Series,
    n_transplant: int = 30,
    random_state: int = 42,
    tumor_labels: list = None,
    normal_labels: list = None,
) -> dict:
    """
    Transplant tumor expression into normal tissue positions.

    Parameters
    ----------
    X : np.ndarray
        Raw expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    labels_series : pd.Series
        Pathologist tissue labels for each spot
    n_transplant : int
        Number of spots to transplant
    random_state : int
        Random seed
    tumor_labels : list
        Labels considered as tumor (default: invasive cancer, dcis)
    normal_labels : list
        Labels considered as normal tissue (default: connective tissue, etc.)

    Returns
    -------
    dict with:
        - X_modified: expression matrix with transplanted spots
        - transplant_mask: boolean mask (True = transplanted spots)
        - original_labels: original pathologist labels
        - donor_indices: index of tumor spot each transplant came from
        - recipient_indices: indices of normal spots that received tumor expression
    """
    rng = np.random.RandomState(random_state)

    if tumor_labels is None:
        tumor_labels = ["invasive cancer", "dcis", "cancer in situ"]
    if normal_labels is None:
        normal_labels = ["connective tissue", "adipose tissue", "breast glands", "fat"]

    labels_lower = labels_series.str.lower().str.strip()

    # Find tumor and normal spots
    tumor_mask = np.zeros(len(labels_lower), dtype=bool)
    for lbl in tumor_labels:
        tumor_mask |= labels_lower.str.contains(lbl, na=False).values

    normal_mask = np.zeros(len(labels_lower), dtype=bool)
    for lbl in normal_labels:
        normal_mask |= labels_lower.str.contains(lbl, na=False).values

    tumor_idx = np.where(tumor_mask)[0]
    normal_idx = np.where(normal_mask)[0]

    n_tumor = len(tumor_idx)
    n_normal = len(normal_idx)

    if n_tumor == 0:
        raise ValueError("No tumor spots found in this sample")
    if n_normal == 0:
        raise ValueError("No normal spots found in this sample")

    # Determine how many to transplant
    n_actual = min(n_transplant, n_normal, n_tumor)
    if n_actual < n_transplant:
        print(f"  Warning: Can only transplant {n_actual}/{n_transplant} "
              f"(normal={n_normal}, tumor={n_tumor})")

    # Select recipients (normal spots) and donors (tumor spots)
    recipient_idx = rng.choice(normal_idx, n_actual, replace=False)
    donor_idx = rng.choice(tumor_idx, n_actual, replace=True)  # allow reuse of donors

    # Perform transplantation
    X_modified = X.copy().astype(float)
    for i in range(n_actual):
        X_modified[recipient_idx[i]] = X[donor_idx[i]].copy()

    # Create transplant mask
    transplant_mask = np.zeros(len(X), dtype=bool)
    transplant_mask[recipient_idx] = True

    return {
        "X_modified": X_modified,
        "X_original": X,
        "transplant_mask": transplant_mask,
        "tumor_mask": tumor_mask,
        "normal_mask": normal_mask,
        "donor_indices": donor_idx,
        "recipient_indices": recipient_idx,
        "n_transplanted": n_actual,
        "donor_coords": coords[donor_idx],
        "recipient_coords": coords[recipient_idx],
    }


def evaluate_transplant_detection(
    X_modified: np.ndarray,
    coords: np.ndarray,
    transplant_mask: np.ndarray,
    seed: int,
    config: ExperimentConfig,
    n_top_genes: int = 2000,
    X_original: np.ndarray = None,
) -> dict:
    """
    Train model on modified data and evaluate detection of transplanted spots.

    Parameters
    ----------
    X_modified : np.ndarray
        Expression matrix with transplanted tumor spots
    coords : np.ndarray
        Spatial coordinates
    transplant_mask : np.ndarray
        Boolean mask (True = transplanted spot)
    seed : int
        Random seed
    config : ExperimentConfig
        Configuration
    n_top_genes : int
        Number of HVGs for training
    X_original : np.ndarray
        Pre-transplant expression matrix for HVG selection.
        Required to avoid data leakage from transplanted spots.

    Returns
    -------
    dict with AUC scores for each method
    """
    # Set all random seeds for reproducibility (numpy + torch)
    set_seed(seed)  # Also sets torch.manual_seed internally

    # HVG selection on pre-transplant data to avoid data leak
    if X_original is None:
        raise ValueError("X_original is required for HVG selection to avoid data leakage")
    X_for_hvg = X_original
    gene_means = X_for_hvg.mean(axis=0) + 1e-8
    gene_vars = X_for_hvg.var(axis=0)
    fano = gene_vars / gene_means
    n_select = min(n_top_genes, X_for_hvg.shape[1])
    hvg_idx = np.argsort(fano)[-n_select:]
    hvg_idx = np.sort(hvg_idx)
    X_hvg = X_modified[:, hvg_idx]

    # Prepare data
    data = prepare_data(X_hvg, coords, k=config.k_neighbors, device=config.device)

    result = {"seed": seed}

    # --- Inverse Prediction ---
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

    y_true = transplant_mask.astype(int)

    # Compute AUCs for inverse prediction scores
    result["auc_inv_pos"] = roc_auc_score(y_true, scores["s_pos"])
    result["auc_inv_self"] = roc_auc_score(y_true, scores["s_self"])
    result["auc_inv_neighbor"] = roc_auc_score(y_true, scores["s_neighbor"])
    result["auc_inv_if"] = roc_auc_score(y_true, scores["s_if"])

    # AUPRC
    result["auprc_inv_pos"] = average_precision_score(y_true, scores["s_pos"])

    # --- Baselines ---
    X_norm = data["X_norm"]

    pca_scores = compute_pca_error(X_norm)
    result["auc_pca_error"] = roc_auc_score(y_true, pca_scores)

    nd_scores = compute_neighbor_diff(X_norm, coords)
    result["auc_neighbor_diff"] = roc_auc_score(y_true, nd_scores)

    lisa_scores = compute_lisa(X_norm, coords)
    result["auc_lisa"] = roc_auc_score(y_true, lisa_scores)

    lof_scores = compute_lof(X_norm)
    result["auc_lof"] = roc_auc_score(y_true, lof_scores)

    if_scores = compute_isolation_forest(X_norm, random_state=seed)
    result["auc_isolation_forest"] = roc_auc_score(y_true, if_scores)

    # Store raw scores for visualization
    result["scores_inv_pos"] = scores["s_pos"]
    result["scores_pca"] = pca_scores
    result["scores_neighbor_diff"] = nd_scores
    result["scores_lisa"] = lisa_scores
    result["pos_pred"] = scores["pos_pred"]

    return result


def run_transplantation_experiment(
    her2st_dir: str,
    config: ExperimentConfig = None,
    sample_ids: list = None,
    n_transplant: int = 30,
    n_seeds: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run tumor-to-normal transplantation experiment on HER2ST data.

    Parameters
    ----------
    her2st_dir : str
        Path to HER2ST data directory
    config : ExperimentConfig
        Configuration
    sample_ids : list
        Specific samples (None = all)
    n_transplant : int
        Number of spots to transplant per sample
    n_seeds : int
        Number of random seeds
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame with results
    """
    if config is None:
        config = DEFAULT_CONFIG

    loader = HER2STDataLoader(her2st_dir)
    if sample_ids is None:
        sample_ids = loader.available_samples

    seeds = config.seeds[:n_seeds]

    if verbose:
        print("=" * 80)
        print("Experiment 17: HER2ST Tumor-to-Normal Transplantation")
        print("=" * 80)
        print(f"Samples: {sample_ids}")
        print(f"Seeds: {seeds}")
        print(f"Transplant spots per sample: {n_transplant}")

    all_results = []

    for sample_id in sample_ids:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Sample: {sample_id}")
            print("=" * 60)

        # Load data
        X_sparse, coords, y_true_cancer, metadata = loader.load(sample_id)
        X_raw = X_sparse.toarray() if sparse.issparse(X_sparse) else np.asarray(X_sparse)

        # Load pathologist labels directly
        labels_file = loader.labels_dir / f"{sample_id}_labeled_coordinates.tsv"
        labels_df = pd.read_csv(labels_file, sep="\t")
        labels_df = labels_df.dropna(subset=["x", "y"])
        labels_df["array_x"] = labels_df["x"].round().astype(int)
        labels_df["array_y"] = labels_df["y"].round().astype(int)
        labels_df["spot_id"] = (
            labels_df["array_x"].astype(str) + "x" + labels_df["array_y"].astype(str)
        )

        # Align labels with counts (same order as loader.load())
        counts_file = loader.counts_dir / f"{sample_id}.tsv.gz"
        with gzip.open(counts_file, "rt") as f:
            counts_df = pd.read_csv(f, sep="\t", index_col=0)
        common_spots = counts_df.index.intersection(labels_df["spot_id"])
        labels_df_indexed = labels_df.set_index("Row.names")
        labels_df_indexed = labels_df_indexed[labels_df_indexed["spot_id"].isin(common_spots)]
        spot_id_to_rowname = dict(zip(labels_df_indexed["spot_id"], labels_df_indexed.index))
        row_order = [spot_id_to_rowname[sid] for sid in common_spots]
        labels_aligned = labels_df_indexed.loc[row_order]
        tissue_labels = labels_aligned["label"]

        if verbose:
            print(f"  Spots: {X_raw.shape[0]}, Genes: {X_raw.shape[1]}")
            print(f"  Labels: {tissue_labels.value_counts().to_dict()}")

        for seed in seeds:
            if verbose:
                print(f"  Seed {seed}...", end=" ", flush=True)

            # Transplant tumor expression into normal spots
            transplant = transplant_tumor_to_normal(
                X_raw, coords, tissue_labels,
                n_transplant=n_transplant,
                random_state=seed,
            )

            if verbose:
                print(f"transplanted {transplant['n_transplanted']}...", end=" ", flush=True)

            # Evaluate detection
            eval_result = evaluate_transplant_detection(
                transplant["X_modified"],
                coords,
                transplant["transplant_mask"],
                seed=seed,
                config=config,
                X_original=transplant["X_original"],
            )

            # Add metadata
            eval_result["sample_id"] = sample_id
            eval_result["n_spots"] = X_raw.shape[0]
            eval_result["n_transplanted"] = transplant["n_transplanted"]
            eval_result["n_tumor_available"] = int(transplant["tumor_mask"].sum())
            eval_result["n_normal_available"] = int(transplant["normal_mask"].sum())

            # Remove raw score arrays before adding to results table
            scores_to_save = {}
            keys_to_remove = []
            for k, v in eval_result.items():
                if isinstance(v, np.ndarray):
                    scores_to_save[k] = v
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del eval_result[k]

            all_results.append(eval_result)

            if verbose:
                auc_pos = eval_result["auc_inv_pos"]
                auc_pca = eval_result["auc_pca_error"]
                auc_lisa = eval_result["auc_lisa"]
                print(f"AUC: InvPos={auc_pos:.3f}, PCA={auc_pca:.3f}, LISA={auc_lisa:.3f}")

    results_df = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        auc_cols = [c for c in results_df.columns if c.startswith("auc_")]
        for col in sorted(auc_cols):
            method = col.replace("auc_", "")
            vals = results_df[col].dropna()
            print(f"  {method:20s}: {vals.mean():.3f} ± {vals.std():.3f}")

    return results_df


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """
    Run HER2ST transplantation experiment (for run_all.py compatibility).

    Returns a DataFrame with detection AUC results.
    """
    if config is None:
        config = DEFAULT_CONFIG

    her2st_dir = config.her2st_dir

    # Check if HER2ST data is available
    if not os.path.exists(her2st_dir):
        print(f"  Warning: HER2ST data not found at {her2st_dir}")
        print("  Skipping exp17_her2st_transplantation")
        return pd.DataFrame()

    n_seeds = len(config.seeds) if hasattr(config, 'seeds') else 10

    results = run_transplantation_experiment(
        her2st_dir,
        config=config,
        sample_ids=None,  # Use all available samples
        n_transplant=30,
        n_seeds=min(n_seeds, 10),
        verbose=verbose,
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--her2st-dir", type=str,
                        default=DEFAULT_CONFIG.her2st_dir)
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--n-transplant", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=str,
                        default=os.path.join(RESULTS_DIR, "exp17_transplantation.csv"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    results = run_transplantation_experiment(
        args.her2st_dir,
        config=config,
        sample_ids=args.samples,
        n_transplant=args.n_transplant,
        n_seeds=args.seeds,
        verbose=True,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
