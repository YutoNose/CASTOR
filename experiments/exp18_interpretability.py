"""
Experiment 18: Predicted Position Interpretability Analysis

For transplanted spots (tumor expression at normal positions),
verify that the model's predicted position points toward the
donor (tumor) location rather than the actual (recipient) position.

This validates the core claim: "Ectopic cells' predicted positions
point to their origin."

Protocol:
1. Transplant tumor expression into normal tissue positions (same as exp17)
2. Train inverse prediction model (unsupervised)
3. For each transplanted spot, compute:
   - Predicted position (from model)
   - Distance to donor position (where the expression came from)
   - Distance to actual position (where the spot physically is)
4. Compare: predicted position should be closer to donor than to actual

No fabricated data. No fallbacks.
"""

import numpy as np
import pandas as pd
from scipy import sparse
import gzip
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

# EXP_DIR already points to /home/yutonose/Projects which contains data/
from data.generators.her2st import HER2STDataLoader


def run_interpretability_analysis(
    her2st_dir: str,
    config: ExperimentConfig = None,
    sample_ids: list = None,
    n_transplant: int = 30,
    n_seeds: int = 5,
    n_top_genes: int = 2000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run transplantation and analyze predicted positions.

    Returns DataFrame with one row per (sample, seed, transplanted_spot).
    """
    if config is None:
        config = DEFAULT_CONFIG

    loader = HER2STDataLoader(her2st_dir)
    if sample_ids is None:
        sample_ids = loader.available_samples

    seeds = config.seeds[:n_seeds]

    tumor_labels = ["invasive cancer", "dcis", "cancer in situ"]
    normal_labels = ["connective tissue", "adipose tissue", "breast glands", "fat"]

    if verbose:
        print("=" * 70)
        print("Experiment 18: Predicted Position Interpretability")
        print("=" * 70)
        print(f"Samples: {sample_ids}")
        print(f"Seeds: {seeds}")

    all_rows = []

    for sample_id in sample_ids:
        if verbose:
            print(f"\n--- Sample: {sample_id} ---")

        # Load data
        X_sparse, coords, y_true, meta = loader.load(sample_id)
        X_raw = X_sparse.toarray() if sparse.issparse(X_sparse) else np.asarray(X_sparse)

        # Load pathologist labels
        labels_file = loader.labels_dir / f"{sample_id}_labeled_coordinates.tsv"
        labels_df = pd.read_csv(labels_file, sep="\t")
        labels_df = labels_df.dropna(subset=["x", "y"])
        labels_df["array_x"] = labels_df["x"].round().astype(int)
        labels_df["array_y"] = labels_df["y"].round().astype(int)
        labels_df["spot_id"] = (
            labels_df["array_x"].astype(str) + "x" + labels_df["array_y"].astype(str)
        )

        # Align labels with counts (same approach as exp17)
        counts_file = loader.counts_dir / f"{sample_id}.tsv.gz"
        with gzip.open(counts_file, "rt") as f:
            counts_df = pd.read_csv(f, sep="\t", index_col=0)
        common_spots = list(counts_df.index.intersection(labels_df["spot_id"]))

        labels_df_indexed = labels_df.set_index("spot_id")
        tissue_labels_list = []
        for sid in common_spots[:len(coords)]:
            if sid in labels_df_indexed.index:
                val = labels_df_indexed.loc[sid, "label"]
                if isinstance(val, pd.Series):
                    tissue_labels_list.append(val.iloc[0])
                else:
                    tissue_labels_list.append(val)
            else:
                tissue_labels_list.append("undetermined")

        tissue_labels = pd.Series(tissue_labels_list, index=range(len(tissue_labels_list)))

        # Identify tumor and normal spots
        labels_lower = tissue_labels.str.lower().str.strip()
        tumor_mask = np.zeros(len(labels_lower), dtype=bool)
        for lbl in tumor_labels:
            tumor_mask |= labels_lower.str.contains(lbl, na=False).values
        normal_mask = np.zeros(len(labels_lower), dtype=bool)
        for lbl in normal_labels:
            normal_mask |= labels_lower.str.contains(lbl, na=False).values

        tumor_idx = np.where(tumor_mask)[0]
        normal_idx = np.where(normal_mask)[0]

        if len(tumor_idx) == 0 or len(normal_idx) == 0:
            if verbose:
                print(f"  Skipping: no tumor or normal spots")
            continue

        n_spots = X_raw.shape[0]
        if verbose:
            print(f"  Spots: {n_spots}, Tumor: {len(tumor_idx)}, Normal: {len(normal_idx)}")

        # Normalize coordinates to [0,1] for distance comparisons
        coords_min = coords.min(axis=0)
        coords_range = coords.max(axis=0) - coords.min(axis=0)
        coords_range[coords_range == 0] = 1
        coords_norm = (coords - coords_min) / coords_range

        for seed in seeds:
            if verbose:
                print(f"  Seed {seed}...", end=" ", flush=True)

            rng = np.random.RandomState(seed)
            # Set all random seeds for reproducibility (numpy + torch)
            set_seed(seed)  # Also sets torch.manual_seed internally

            # Transplantation
            n_actual = min(n_transplant, len(normal_idx), len(tumor_idx))
            recipient_idx = rng.choice(normal_idx, n_actual, replace=False)
            donor_idx = rng.choice(tumor_idx, n_actual, replace=True)

            X_modified = X_raw.copy().astype(float)
            for i in range(n_actual):
                X_modified[recipient_idx[i]] = X_raw[donor_idx[i]].copy()

            # HVG selection on pre-transplant data to avoid data leakage
            gene_means = X_raw.mean(axis=0) + 1e-8
            gene_vars = X_raw.var(axis=0)
            fano = gene_vars / gene_means
            n_select = min(n_top_genes, X_raw.shape[1])
            hvg_idx = np.argsort(fano)[-n_select:]
            hvg_idx = np.sort(hvg_idx)
            X_hvg = X_modified[:, hvg_idx]

            # Prepare and train
            data = prepare_data(X_hvg, coords, k=config.k_neighbors, device=config.device)

            model = InversePredictionModel(
                in_dim=data["n_genes"],
                hid_dim=config.hidden_dim,
                dropout=config.dropout,
            ).to(config.device)

            model = train_model(
                model, data["x_tensor"], data["coords_tensor"],
                data["edge_index"],
                n_epochs=config.n_epochs, lr=config.learning_rate,
                lambda_pos=config.lambda_pos, lambda_self=config.lambda_self,
                verbose=False,
            )

            scores = compute_scores(
                model, data["x_tensor"], data["coords_tensor"],
                data["edge_index"], random_state=seed,
            )

            # Predicted positions are in normalized [0,1] space
            pos_pred = scores["pos_pred"]  # [n_spots, 2]

            # Coords used by model are also normalized to [0,1]
            coords_model = data["coords_norm"]  # [n_spots, 2]

            # For each transplanted spot, compute distances
            for i in range(n_actual):
                r_idx = recipient_idx[i]  # actual position (recipient)
                d_idx = donor_idx[i]      # donor position (tumor)

                pred_pos = pos_pred[r_idx]           # model's predicted position
                actual_pos = coords_model[r_idx]     # actual (recipient) position
                donor_pos = coords_model[d_idx]      # donor (tumor) position

                dist_to_actual = np.sqrt(np.sum((pred_pos - actual_pos) ** 2))
                dist_to_donor = np.sqrt(np.sum((pred_pos - donor_pos) ** 2))
                dist_donor_actual = np.sqrt(np.sum((donor_pos - actual_pos) ** 2))

                all_rows.append({
                    "sample_id": sample_id,
                    "seed": seed,
                    "transplant_idx": i,
                    "recipient_spot": r_idx,
                    "donor_spot": d_idx,
                    "pred_x": pred_pos[0],
                    "pred_y": pred_pos[1],
                    "actual_x": actual_pos[0],
                    "actual_y": actual_pos[1],
                    "donor_x": donor_pos[0],
                    "donor_y": donor_pos[1],
                    "dist_pred_to_actual": dist_to_actual,
                    "dist_pred_to_donor": dist_to_donor,
                    "dist_donor_to_actual": dist_donor_actual,
                    "closer_to_donor": int(dist_to_donor < dist_to_actual),
                })

            n_closer = sum(1 for r in all_rows[-n_actual:] if r["closer_to_donor"])
            if verbose:
                print(f"done. {n_closer}/{n_actual} closer to donor ({100*n_closer/n_actual:.0f}%)")

    results_df = pd.DataFrame(all_rows)

    if verbose and len(results_df) > 0:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        frac = results_df["closer_to_donor"].mean()
        n_total = len(results_df)
        n_closer = results_df["closer_to_donor"].sum()

        print(f"  Fraction closer to donor: {frac:.3f} ({100*frac:.1f}%)")
        print(f"  Mean dist to donor:  {results_df['dist_pred_to_donor'].mean():.4f}")
        print(f"  Mean dist to actual: {results_df['dist_pred_to_actual'].mean():.4f}")
        print(f"  Mean donor-actual:   {results_df['dist_donor_to_actual'].mean():.4f}")

        # Statistical tests
        print("\n" + "-" * 40)
        print("STATISTICAL TESTS")
        print("-" * 40)

        # 1. Binomial test: H0: p=0.5 (random prediction would be equally likely
        #    to be closer to donor or actual)
        from scipy.stats import wilcoxon

        # Use binomtest (scipy >= 1.7) or binom_test (older)
        try:
            from scipy.stats import binomtest
            result = binomtest(n_closer, n_total, p=0.5, alternative='greater')
            binom_pval = result.pvalue
        except ImportError:
            # Fallback for older scipy
            from scipy.stats import binom_test
            binom_pval = binom_test(n_closer, n_total, p=0.5, alternative='greater')

        print(f"  Binomial test (H0: p=0.5):")
        print(f"    n_closer={n_closer}, n_total={n_total}")
        print(f"    p-value = {binom_pval:.2e}")
        if binom_pval < 0.05:
            print(f"    Result: SIGNIFICANT (p < 0.05)")
        else:
            print(f"    Result: Not significant (p >= 0.05)")

        # 2. Wilcoxon signed-rank test on paired distances
        # H0: dist_to_donor and dist_to_actual come from same distribution
        # H1: dist_to_donor < dist_to_actual (one-sided)
        dist_donor = results_df["dist_pred_to_donor"].values
        dist_actual = results_df["dist_pred_to_actual"].values

        w_pval = np.nan  # Initialize in case test fails
        try:
            w_stat, w_pval = wilcoxon(dist_donor, dist_actual, alternative='less')
            print(f"\n  Wilcoxon signed-rank test (H0: dist_donor >= dist_actual):")
            print(f"    W statistic = {w_stat:.1f}")
            print(f"    p-value = {w_pval:.2e}")
            if w_pval < 0.05:
                print(f"    Result: SIGNIFICANT (p < 0.05)")
            else:
                print(f"    Result: Not significant (p >= 0.05)")
        except Exception as e:
            print(f"\n  Wilcoxon test failed: {e}")

        # Add statistical test results to DataFrame metadata (for programmatic access)
        results_df.attrs["binomial_pval"] = binom_pval
        results_df.attrs["wilcoxon_pval"] = w_pval
        results_df.attrs["fraction_closer_to_donor"] = frac

    return results_df


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """
    Run interpretability analysis (CLI-compatible wrapper).

    Returns a DataFrame with per-spot interpretability results.
    """
    if config is None:
        config = DEFAULT_CONFIG

    her2st_dir = config.her2st_dir

    if not os.path.exists(her2st_dir):
        print(f"  Warning: HER2ST data not found at {her2st_dir}")
        print("  Skipping exp18_interpretability")
        return pd.DataFrame()

    n_seeds = min(len(config.seeds), 5) if hasattr(config, 'seeds') else 5

    return run_interpretability_analysis(
        her2st_dir,
        config=config,
        sample_ids=None,
        n_transplant=30,
        n_seeds=n_seeds,
        verbose=verbose,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--her2st-dir", type=str,
                        default=DEFAULT_CONFIG.her2st_dir)
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-transplant", type=int, default=30)
    parser.add_argument("--output", type=str,
                        default=os.path.join(RESULTS_DIR, "exp18_interpretability.csv"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    results = run_interpretability_analysis(
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
