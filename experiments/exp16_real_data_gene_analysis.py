"""
Experiment 16: Real Data Application with Gene Contribution Analysis

Applies the inverse prediction model to real ST data (HER2ST Visium-like),
computes position prediction error for each spot, then identifies genes
that contribute most to high prediction error (DEG-like analysis).

Analysis:
1. Train inverse prediction model on full real data (unsupervised)
2. Compute per-spot position prediction error
3. Stratify spots into high-error vs low-error groups
4. Run differential expression (Wilcoxon rank-sum) between groups
5. Visualize: spatial error map, volcano plot, top gene heatmap

No fabricated data. No fallbacks.
"""

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import roc_auc_score
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


def run_gene_analysis_single_sample(
    X_raw: np.ndarray,
    coords: np.ndarray,
    gene_names: list,
    sample_id: str,
    seed: int,
    config: ExperimentConfig,
    n_top_genes: int = 2000,
    error_percentile: float = 90,
) -> dict:
    """
    Run inverse prediction and identify contributing genes on a single sample.

    Parameters
    ----------
    X_raw : np.ndarray
        Raw count matrix [n_spots, n_genes]
    coords : np.ndarray
        Spatial coordinates [n_spots, 2]
    gene_names : list
        Gene names matching columns of X_raw
    sample_id : str
        Sample identifier
    seed : int
        Random seed
    config : ExperimentConfig
        Experiment configuration
    n_top_genes : int
        Number of HVGs to use for model training
    error_percentile : float
        Percentile threshold for high-error group

    Returns
    -------
    dict with:
        - s_pos: per-spot position prediction error
        - pos_pred: predicted positions (normalized)
        - deg_results: DataFrame of differential expression results
        - high_error_mask: boolean mask for high-error spots
    """
    # Set all random seeds for reproducibility (numpy + torch)
    set_seed(seed)  # Also sets torch.manual_seed internally

    # --- Gene selection (HVG) for model training ---
    # Use Fano factor (variance / mean) to pick HVGs
    gene_means = X_raw.mean(axis=0) + 1e-8
    gene_vars = X_raw.var(axis=0)
    fano = gene_vars / gene_means
    n_select = min(n_top_genes, X_raw.shape[1])
    hvg_idx = np.argsort(fano)[-n_select:]
    hvg_idx = np.sort(hvg_idx)

    X_hvg = X_raw[:, hvg_idx]
    hvg_names = [gene_names[i] for i in hvg_idx]

    # --- Prepare data and train model ---
    data = prepare_data(X_hvg, coords, k=config.k_neighbors, device=config.device)

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

    s_pos = scores["s_pos"]
    pos_pred = scores["pos_pred"]

    # --- Stratify into high-error vs low-error groups ---
    threshold = np.percentile(s_pos, error_percentile)
    high_error_mask = s_pos >= threshold
    low_error_mask = s_pos < np.percentile(s_pos, 100 - error_percentile)

    n_high = high_error_mask.sum()
    n_low = low_error_mask.sum()

    if n_high < 5 or n_low < 5:
        raise ValueError(
            f"Too few spots in groups: high={n_high}, low={n_low}. "
            f"Sample may be too small."
        )

    # --- Differential expression (on ALL genes, not just HVGs) ---
    # Use log1p-transformed raw counts for DE
    X_log = np.log1p(X_raw.astype(float))

    deg_results = []
    for g_idx in range(X_raw.shape[1]):
        expr_high = X_log[high_error_mask, g_idx]
        expr_low = X_log[low_error_mask, g_idx]

        # Skip genes with no expression variation in either group
        if expr_high.std() < 1e-10 and expr_low.std() < 1e-10:
            continue

        # Wilcoxon rank-sum test
        try:
            stat, pval = stats.mannwhitneyu(
                expr_high, expr_low, alternative='two-sided'
            )
        except ValueError:
            continue

        # Log2 fold change: convert from ln-space to log2-space
        mean_high = expr_high.mean()
        mean_low = expr_low.mean()
        log2fc = (mean_high - mean_low) / np.log(2)

        deg_results.append({
            "gene": gene_names[g_idx],
            "log2fc": log2fc,
            "pval": pval,
            "mean_high_error": mean_high,
            "mean_low_error": mean_low,
            "pct_high": (X_raw[high_error_mask, g_idx] > 0).mean(),
            "pct_low": (X_raw[low_error_mask, g_idx] > 0).mean(),
        })

    deg_df = pd.DataFrame(deg_results)

    # Multiple testing correction (Benjamini-Hochberg)
    if len(deg_df) > 0:
        from statsmodels.stats.multitest import multipletests
        reject, padj, _, _ = multipletests(deg_df["pval"].values, method="fdr_bh")
        deg_df["padj"] = padj
        deg_df["significant"] = reject
        deg_df = deg_df.sort_values("pval")
    else:
        raise ValueError("No genes passed DE filtering")

    return {
        "s_pos": s_pos,
        "pos_pred": pos_pred,
        "coords_norm": data["coords_norm"],
        "deg_results": deg_df,
        "high_error_mask": high_error_mask,
        "low_error_mask": low_error_mask,
        "threshold": threshold,
        "hvg_names": hvg_names,
        "hvg_idx": hvg_idx,
    }


def run_real_data_gene_analysis(
    her2st_dir: str,
    config: ExperimentConfig = None,
    sample_ids: list = None,
    n_seeds: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Run gene contribution analysis on HER2ST real data.

    Returns
    -------
    dict mapping sample_id -> list of per-seed results
    """
    if config is None:
        config = DEFAULT_CONFIG

    loader = HER2STDataLoader(her2st_dir)
    if sample_ids is None:
        sample_ids = loader.available_samples

    seeds = config.seeds[:n_seeds]

    if verbose:
        print("=" * 80)
        print("Experiment 16: Real Data Gene Contribution Analysis")
        print("=" * 80)
        print(f"Samples: {sample_ids}")
        print(f"Seeds: {seeds}")

    all_results = {}

    for sample_id in sample_ids:
        if verbose:
            print(f"\n--- Sample: {sample_id} ---")

        # Load data
        X_sparse, coords, y_true, metadata = loader.load(sample_id)
        X_raw = X_sparse.toarray() if sparse.issparse(X_sparse) else np.asarray(X_sparse)

        # Get gene names from count file
        import gzip
        counts_file = loader.counts_dir / f"{sample_id}.tsv.gz"
        with gzip.open(counts_file, "rt") as f:
            counts_df = pd.read_csv(f, sep="\t", index_col=0, nrows=0)
        gene_names = list(counts_df.columns)

        # Align gene names with X_raw columns
        assert len(gene_names) == X_raw.shape[1], (
            f"Gene name mismatch: {len(gene_names)} vs {X_raw.shape[1]}"
        )

        if verbose:
            print(f"  Spots: {X_raw.shape[0]}, Genes: {X_raw.shape[1]}")
            print(f"  Cancer fraction: {metadata['anomaly_fraction']:.1%}")

        sample_results = []
        for seed in seeds:
            if verbose:
                print(f"  Seed {seed}...", end=" ", flush=True)

            result = run_gene_analysis_single_sample(
                X_raw, coords, gene_names, sample_id, seed, config
            )
            result["y_true"] = y_true
            result["metadata"] = metadata
            sample_results.append(result)

            if verbose:
                n_sig = result["deg_results"]["significant"].sum()
                print(f"done. {n_sig} significant DEGs (FDR<0.05)")

        all_results[sample_id] = sample_results

    return all_results


def aggregate_deg_across_seeds(results_list: list) -> pd.DataFrame:
    """Aggregate DEG results across multiple seeds using Fisher's method."""
    # Collect p-values per gene across seeds
    gene_pvals = {}
    gene_fc = {}

    for result in results_list:
        deg = result["deg_results"]
        for _, row in deg.iterrows():
            gene = row["gene"]
            if gene not in gene_pvals:
                gene_pvals[gene] = []
                gene_fc[gene] = []
            gene_pvals[gene].append(row["pval"])
            gene_fc[gene].append(row["log2fc"])

    # Fisher's combined p-value and median fold change
    combined = []
    for gene in gene_pvals:
        pvals = gene_pvals[gene]
        fcs = gene_fc[gene]

        if len(pvals) < 2:
            # Not enough seeds, use single p-value
            combined_p = pvals[0]
        else:
            # Fisher's method
            chi2 = -2 * np.sum(np.log(np.clip(pvals, 1e-300, 1.0)))
            combined_p = stats.chi2.sf(chi2, 2 * len(pvals))

        combined.append({
            "gene": gene,
            "median_log2fc": np.median(fcs),
            "mean_log2fc": np.mean(fcs),
            "combined_pval": combined_p,
            "n_seeds": len(pvals),
            "fc_consistency": np.sign(fcs).mean(),  # -1 to 1
        })

    combined_df = pd.DataFrame(combined)

    # Multiple testing correction
    from statsmodels.stats.multitest import multipletests
    reject, padj, _, _ = multipletests(
        combined_df["combined_pval"].values, method="fdr_bh"
    )
    combined_df["padj"] = padj
    combined_df["significant"] = reject
    combined_df = combined_df.sort_values("combined_pval")

    return combined_df


def save_results(all_results: dict, output_dir: str):
    """Save all results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    for sample_id, results_list in all_results.items():
        # Save per-seed DEG results
        for i, result in enumerate(results_list):
            deg = result["deg_results"]
            deg.to_csv(
                os.path.join(output_dir, f"{sample_id}_seed{i}_deg.csv"),
                index=False,
            )

        # Save aggregated DEG results
        combined = aggregate_deg_across_seeds(results_list)
        combined.to_csv(
            os.path.join(output_dir, f"{sample_id}_combined_deg.csv"),
            index=False,
        )

        # Save per-spot scores (from first seed as representative)
        spot_data = pd.DataFrame({
            "s_pos": results_list[0]["s_pos"],
            "high_error": results_list[0]["high_error_mask"].astype(int),
            "y_true": results_list[0]["y_true"],
        })
        spot_data.to_csv(
            os.path.join(output_dir, f"{sample_id}_spot_scores.csv"),
            index=False,
        )

    print(f"Results saved to {output_dir}")


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """
    Run gene contribution analysis experiment (for run_all.py compatibility).

    Returns a summary DataFrame with per-sample statistics.
    """
    if config is None:
        config = DEFAULT_CONFIG

    her2st_dir = "/home/yutonose/Projects/her2st"
    output_dir = os.path.join(RESULTS_DIR, "exp16_gene_analysis")

    # Check if HER2ST data is available
    if not os.path.exists(her2st_dir):
        print(f"  Warning: HER2ST data not found at {her2st_dir}")
        print("  Skipping exp16_real_data_gene_analysis")
        return pd.DataFrame()

    n_seeds = len(config.seeds) if hasattr(config, 'seeds') else 5

    all_results = run_real_data_gene_analysis(
        her2st_dir,
        config=config,
        sample_ids=None,  # Use all available samples
        n_seeds=min(n_seeds, 5),  # Limit seeds for reasonable runtime
        verbose=verbose,
    )

    # Save detailed results
    save_results(all_results, output_dir)

    # Create summary DataFrame for run_all.py
    summary_rows = []
    for sample_id, results_list in all_results.items():
        combined_deg = aggregate_deg_across_seeds(results_list)
        n_sig = combined_deg["significant"].sum()
        n_up = ((combined_deg["significant"]) & (combined_deg["median_log2fc"] > 0)).sum()
        n_down = ((combined_deg["significant"]) & (combined_deg["median_log2fc"] < 0)).sum()

        summary_rows.append({
            "sample_id": sample_id,
            "n_spots": results_list[0]["high_error_mask"].shape[0],
            "n_significant_degs": n_sig,
            "n_upregulated": n_up,
            "n_downregulated": n_down,
            "n_seeds": len(results_list),
        })

    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--her2st-dir", type=str,
                        default="/home/yutonose/Projects/her2st")
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default=os.path.join(RESULTS_DIR, "exp16_gene_analysis"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    results = run_real_data_gene_analysis(
        args.her2st_dir,
        config=config,
        sample_ids=args.samples,
        n_seeds=args.seeds,
        verbose=True,
    )

    save_results(results, args.output)
