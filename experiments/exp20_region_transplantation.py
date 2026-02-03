"""
Experiment 20: Region Transplantation on HER2ST

Extends exp17 (scattered tumor-to-normal transplantation) by transplanting
contiguous REGIONS of tumor tissue into normal tissue, not individual spots.

This is the real-data counterpart of exp19 (clustered ectopic on synthetic
data). The key variable is region_size: how many contiguous spots per
transplanted region.

Protocol:
1. Load HER2ST sample with pathologist annotations
2. Identify normal tissue spots and tumor spots
3. Select contiguous blocks of normal spots as recipients (via KDTree)
4. Find corresponding contiguous tumor spots as donors (offset mapping)
5. Copy tumor expression into recipient positions from original snapshot
6. Train inverse prediction model on modified data (unsupervised)
7. Evaluate detection across all methods

Independent variable: region_size (1, 3, 5, 10, 15, 30)
  - region_size=1 reproduces exp17 (scattered transplant)
  - Total transplanted spots = 30 (constant, same as exp17)
  - n_regions = 30 // region_size

Seeds: 10 per sample (same as exp17)
Minimum normal spots required: 30 (skip samples with fewer)
"""

import gzip
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import KDTree
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

from data.generators.her2st import HER2STDataLoader

REGION_SIZES = [1, 3, 5, 10, 15, 30]


def transplant_region_to_normal(
    X: np.ndarray,
    coords: np.ndarray,
    labels_series: pd.Series,
    region_size: int = 1,
    n_total_transplant: int = 30,
    random_state: int = 42,
    tumor_labels: list = None,
    normal_labels: list = None,
) -> dict:
    """
    Transplant contiguous regions of tumor expression into normal tissue.

    For each region:
    1. Select a contiguous block of `region_size` normal spots (recipients)
       using KDTree nearest-neighbor queries
    2. Select a contiguous block of tumor spots as donors using spatial
       offset mapping from a distant tumor center
    3. Copy tumor expression to recipient positions from the original
       (pre-modification) expression snapshot

    When region_size=1, this reduces to exp17's scattered transplantation.

    Parameters
    ----------
    X : np.ndarray
        Raw expression matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    labels_series : pd.Series
        Pathologist tissue labels for each spot
    region_size : int
        Number of contiguous spots per transplanted region
    n_total_transplant : int
        Total number of spots to transplant (constant across region_sizes)
    random_state : int
        Random seed
    tumor_labels : list
        Labels considered as tumor
    normal_labels : list
        Labels considered as normal tissue

    Returns
    -------
    dict with:
        - X_modified: expression matrix with transplanted spots
        - transplant_mask: boolean mask (True = transplanted spots)
        - n_transplanted: actual number of transplanted spots
        - donor_indices, recipient_indices: index arrays
        - tumor_mask, normal_mask: boolean masks for original labels
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
    normal_set = set(normal_idx.tolist())

    if len(tumor_idx) == 0:
        raise ValueError("No tumor spots found")
    if len(normal_idx) < n_total_transplant:
        raise ValueError(
            f"Not enough normal spots ({len(normal_idx)}) for "
            f"n_total_transplant={n_total_transplant}"
        )

    # Compute number of regions
    n_regions = max(1, n_total_transplant // region_size)
    target_per_region = region_size

    # Build KDTree over all spot coordinates
    tree = KDTree(coords)

    # Snapshot original expression to prevent cascading contamination
    X_modified = X.copy().astype(float)
    X_original = X.copy()

    used_normal = set()
    all_recipients = []
    all_donors = []

    for _ in range(n_regions):
        # Pick recipient center from unused normal spots
        available_normal = np.array(
            [i for i in normal_idx if i not in used_normal]
        )
        if len(available_normal) < target_per_region:
            break
        recipient_center = rng.choice(available_normal)

        # Select `target_per_region` contiguous normal spots around center
        k_query = min(target_per_region * 5, len(coords))
        _, neighbor_indices = tree.query(
            coords[recipient_center].reshape(1, -1), k=k_query
        )
        neighbor_indices = neighbor_indices[0]

        recipient_spots = []
        for idx in neighbor_indices:
            if idx in normal_set and idx not in used_normal:
                recipient_spots.append(idx)
            if len(recipient_spots) >= target_per_region:
                break

        if len(recipient_spots) < target_per_region:
            # Not enough contiguous normal spots near this center
            continue

        recipient_spots = np.array(recipient_spots)

        # Pick donor center uniformly at random from all tumor spots
        # (no distance bias — avoids systematic overrepresentation of
        # distant donors that could inflate detection performance)
        donor_center = rng.choice(tumor_idx)

        # For each recipient, find corresponding donor via offset mapping
        for r_idx in recipient_spots:
            offset = coords[r_idx] - coords[recipient_center]
            donor_target = coords[donor_center] + offset

            # Find nearest tumor spot to the donor target position
            _, candidate_donors = tree.query(
                donor_target.reshape(1, -1), k=min(20, len(coords))
            )
            candidate_donors = candidate_donors[0]
            d_idx = None
            for c_idx in candidate_donors:
                if tumor_mask[c_idx]:
                    d_idx = c_idx
                    break
            if d_idx is None:
                # Fallback: random tumor donor
                d_idx = rng.choice(tumor_idx)

            # Copy original expression from donor to recipient
            X_modified[r_idx] = X_original[d_idx].copy()
            all_recipients.append(r_idx)
            all_donors.append(d_idx)

        used_normal.update(recipient_spots.tolist())

    recipient_indices = np.array(all_recipients, dtype=int)
    donor_indices = np.array(all_donors, dtype=int)
    transplant_mask = np.zeros(len(X), dtype=bool)
    if len(recipient_indices) > 0:
        transplant_mask[recipient_indices] = True

    return {
        "X_modified": X_modified,
        "X_original": X_original,
        "transplant_mask": transplant_mask,
        "tumor_mask": tumor_mask,
        "normal_mask": normal_mask,
        "donor_indices": donor_indices,
        "recipient_indices": recipient_indices,
        "n_transplanted": len(recipient_indices),
        "donor_coords": coords[donor_indices] if len(donor_indices) > 0 else np.array([]),
        "recipient_coords": coords[recipient_indices] if len(recipient_indices) > 0 else np.array([]),
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
    set_seed(seed)

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

    # Inverse Prediction
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

    result["auc_inv_pos"] = roc_auc_score(y_true, scores["s_pos"])
    result["auc_inv_self"] = roc_auc_score(y_true, scores["s_self"])
    result["auc_inv_neighbor"] = roc_auc_score(y_true, scores["s_neighbor"])
    result["auc_inv_if"] = roc_auc_score(y_true, scores["s_if"])
    result["auprc_inv_pos"] = average_precision_score(y_true, scores["s_pos"])

    # Baselines
    X_norm = data["X_norm"]

    result["auc_pca_error"] = roc_auc_score(y_true, compute_pca_error(X_norm))
    result["auc_neighbor_diff"] = roc_auc_score(y_true, compute_neighbor_diff(X_norm, coords))
    result["auc_lisa"] = roc_auc_score(y_true, compute_lisa(X_norm, coords))
    result["auc_lof"] = roc_auc_score(y_true, compute_lof(X_norm))
    result["auc_isolation_forest"] = roc_auc_score(
        y_true, compute_isolation_forest(X_norm, random_state=seed)
    )

    return result


def run_region_transplantation_experiment(
    her2st_dir: str,
    config: ExperimentConfig = None,
    sample_ids: list = None,
    region_sizes: list = None,
    n_total_transplant: int = 30,
    n_seeds: int = 10,
    min_normal_spots: int = 30,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run region transplantation experiment across samples, region sizes, and seeds.

    Parameters
    ----------
    her2st_dir : str
        Path to HER2ST data directory
    config : ExperimentConfig
        Configuration
    sample_ids : list
        Specific samples (None = all)
    region_sizes : list
        Region sizes to test (default: REGION_SIZES)
    n_total_transplant : int
        Total spots to transplant per run
    n_seeds : int
        Number of random seeds per sample
    min_normal_spots : int
        Skip samples with fewer normal spots
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame with results
    """
    if config is None:
        config = DEFAULT_CONFIG
    if region_sizes is None:
        region_sizes = REGION_SIZES

    loader = HER2STDataLoader(her2st_dir)
    if sample_ids is None:
        sample_ids = loader.available_samples

    seeds = config.seeds[:n_seeds]

    if verbose:
        print("=" * 80)
        print("Experiment 20: Region Transplantation on HER2ST")
        print("=" * 80)
        print(f"Samples: {sample_ids}")
        print(f"Region sizes: {region_sizes}")
        print(f"Seeds: {seeds}")
        print(f"Transplant spots per sample: {n_total_transplant}")

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

        # Align labels with counts
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

        # Count normal spots
        normal_labels_list = ["connective tissue", "adipose tissue", "breast glands", "fat"]
        labels_lower = tissue_labels.str.lower().str.strip()
        n_normal = 0
        for lbl in normal_labels_list:
            n_normal += labels_lower.str.contains(lbl, na=False).sum()

        if verbose:
            print(f"  Spots: {X_raw.shape[0]}, Genes: {X_raw.shape[1]}")
            print(f"  Normal spots: {n_normal}")

        if n_normal < min_normal_spots:
            if verbose:
                print(f"  Skipping {sample_id}: only {n_normal} normal spots "
                      f"(need >= {min_normal_spots})")
            continue

        for rs_idx, region_size in enumerate(region_sizes):
            for seed in seeds:
                # Decorrelate seeds across region sizes to avoid
                # correlated sampling from the same RNG trajectory
                transplant_seed = seed * len(region_sizes) + rs_idx
                if verbose:
                    print(f"  region_size={region_size}, seed={seed}...",
                          end=" ", flush=True)

                try:
                    transplant = transplant_region_to_normal(
                        X_raw, coords, tissue_labels,
                        region_size=region_size,
                        n_total_transplant=n_total_transplant,
                        random_state=transplant_seed,
                    )

                    if transplant["n_transplanted"] == 0:
                        if verbose:
                            print("no transplants, skipping")
                        continue

                    if verbose:
                        print(f"transplanted {transplant['n_transplanted']}...",
                              end=" ", flush=True)

                    eval_result = evaluate_transplant_detection(
                        transplant["X_modified"],
                        coords,
                        transplant["transplant_mask"],
                        seed=seed,
                        config=config,
                        X_original=transplant["X_original"],
                    )

                    eval_result["region_size"] = region_size
                    eval_result["sample_id"] = sample_id
                    eval_result["n_spots"] = X_raw.shape[0]
                    eval_result["n_transplanted"] = transplant["n_transplanted"]
                    eval_result["n_tumor_available"] = int(transplant["tumor_mask"].sum())
                    eval_result["n_normal_available"] = int(transplant["normal_mask"].sum())

                    all_results.append(eval_result)

                    if verbose:
                        auc_pos = eval_result["auc_inv_pos"]
                        auc_lisa = eval_result["auc_lisa"]
                        print(f"InvPos={auc_pos:.3f}, LISA={auc_lisa:.3f}")

                except Exception as e:
                    if verbose:
                        print(f"failed: {e}")

    if not all_results:
        raise RuntimeError("All runs failed")

    results_df = pd.DataFrame(all_results)

    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY: Detection AUC by Region Size")
        print("=" * 80)
        auc_cols = [c for c in results_df.columns if c.startswith("auc_")]
        for rs in region_sizes:
            sub = results_df[results_df["region_size"] == rs]
            if len(sub) == 0:
                continue
            print(f"\n  region_size={rs} (n={len(sub)}):")
            for col in sorted(auc_cols):
                method = col.replace("auc_", "")
                vals = sub[col].dropna()
                if len(vals) > 0:
                    print(f"    {method:20s}: {vals.mean():.3f} +/- {vals.std():.3f}")

    return results_df


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """
    Run HER2ST region transplantation experiment (for CLI compatibility).

    Returns a DataFrame with detection AUC results.
    """
    if config is None:
        config = DEFAULT_CONFIG

    her2st_dir = config.her2st_dir

    if not os.path.exists(her2st_dir):
        print(f"  Warning: HER2ST data not found at {her2st_dir}")
        print("  Skipping exp20_region_transplantation")
        return pd.DataFrame()

    n_seeds = len(config.seeds) if hasattr(config, 'seeds') else 10

    results = run_region_transplantation_experiment(
        her2st_dir,
        config=config,
        sample_ids=None,
        region_sizes=REGION_SIZES,
        n_total_transplant=30,
        n_seeds=min(n_seeds, 10),
        verbose=verbose,
    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Exp20: Region Transplantation on HER2ST"
    )
    parser.add_argument("--her2st-dir", type=str,
                        default=DEFAULT_CONFIG.her2st_dir)
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--region-sizes", nargs="+", type=int, default=None,
                        help="Region sizes to test (default: 1,3,5,10,15,30)")
    parser.add_argument("--n-transplant", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=str,
                        default=os.path.join(RESULTS_DIR,
                                             "exp20_region_transplantation.csv"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    region_sizes = args.region_sizes or REGION_SIZES

    results = run_region_transplantation_experiment(
        args.her2st_dir,
        config=config,
        sample_ids=args.samples,
        region_sizes=region_sizes,
        n_total_transplant=args.n_transplant,
        n_seeds=args.seeds,
        verbose=True,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
