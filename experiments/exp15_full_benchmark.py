"""
Experiment 15: Full Benchmark with Visualization and Statistical Tests

Comprehensive comparison of all methods with:
- Multiple scenarios (baseline, noisy, hard, realistic)
- Real data (HER2ST, Visium)
- Noise robustness analysis
- Statistical significance tests
- Publication-ready figures

Methods compared:
- Inv_PosError (ours)
- STAGATE + IF
- GraphST + IF
- STLearn SME + IF
- Squidpy neighborhood
- LISA, Neighbor_Diff, SpotSweeper
- PCA_Error, LOF, Isolation Forest
"""

# MUST be set before importing TensorFlow for STAGATE CPU mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Add experiment dir FIRST for config
EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXP_DIR)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

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
from core.scenarios import SCENARIOS, generate_scenario_data
from core.competitor_stagate import compute_stagate_score
from core.competitor_graphst import compute_graphst_score
from core.competitor_squidpy import compute_squidpy_nhood_score
from core.competitor_stlearn import compute_stlearn_sme_score

# Method display names for figures
METHOD_NAMES = {
    "inv_pos": "Inv_PosError (Ours)",
    "stagate": "STAGATE + IF",
    "graphst": "GraphST + IF",
    "stlearn": "STLearn SME + IF",
    "squidpy": "Squidpy Nhood",
    "pca_error": "PCA Recon Error",
    "lisa": "LISA",
    "neighbor_diff": "Neighbor Diff",
    "spotsweeper": "SpotSweeper",
    "lof": "LOF",
    "isolation_forest": "Isolation Forest",
}

# Method categories
SPATIAL_METHODS = ["inv_pos", "stagate", "graphst", "stlearn", "squidpy", "lisa", "neighbor_diff", "spotsweeper"]
GLOBAL_METHODS = ["pca_error", "lof", "isolation_forest"]


def compute_all_scores(X, coords, data, seed, config, methods=None):
    """Compute scores for all methods."""
    from scipy import sparse
    if sparse.issparse(X):
        X = X.toarray()

    if methods is None:
        methods = list(METHOD_NAMES.keys())

    scores_dict = {}

    # Inverse Prediction
    if "inv_pos" in methods:
        model = InversePredictionModel(
            in_dim=data["n_genes"],
            hid_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(config.device)
        model = train_model(
            model, data["x_tensor"], data["coords_tensor"],
            data["edge_index"], n_epochs=config.n_epochs,
            lr=config.learning_rate, lambda_pos=config.lambda_pos,
            lambda_self=config.lambda_self, verbose=False,
        )
        scores = compute_scores(
            model, data["x_tensor"], data["coords_tensor"],
            data["edge_index"], random_state=seed,
        )
        scores_dict["inv_pos"] = scores["s_pos"]

    # STAGATE (CPU)
    if "stagate" in methods:
        scores_dict["stagate"] = compute_stagate_score(X, coords, n_epochs=200, random_state=seed)

    # GraphST (CPU)
    if "graphst" in methods:
        scores_dict["graphst"] = compute_graphst_score(X, coords, device='cpu', n_epochs=200, random_state=seed)

    # STLearn
    if "stlearn" in methods:
        scores_dict["stlearn"] = compute_stlearn_sme_score(X, coords, random_state=seed)

    # Squidpy
    if "squidpy" in methods:
        scores_dict["squidpy"] = compute_squidpy_nhood_score(X, coords, random_state=seed)

    # Baselines
    if "pca_error" in methods:
        scores_dict["pca_error"] = compute_pca_error(data["X_norm"])
    if "lisa" in methods:
        scores_dict["lisa"] = compute_lisa(data["X_norm"], coords)
    if "lof" in methods:
        scores_dict["lof"] = compute_lof(data["X_norm"])
    if "isolation_forest" in methods:
        scores_dict["isolation_forest"] = compute_isolation_forest(data["X_norm"], random_state=seed)
    if "neighbor_diff" in methods:
        scores_dict["neighbor_diff"] = compute_neighbor_diff(data["X_norm"], coords)
    if "spotsweeper" in methods:
        scores_dict["spotsweeper"] = compute_spotsweeper(data["X_norm"], coords)

    return scores_dict


def run_synthetic_benchmark(config, scenarios=None, seeds=None, methods=None, verbose=True):
    """Run benchmark on synthetic data."""
    if scenarios is None:
        scenarios = ["baseline", "noisy_ectopic", "hard_ectopic", "realistic_counts"]

    if seeds is None:
        seeds = config.seeds

    if methods is None:
        methods = list(METHOD_NAMES.keys())

    all_results = []

    for scenario_name in scenarios:
        scenario = SCENARIOS[scenario_name]
        if verbose:
            print(f"\nScenario: {scenario.name}")

        for seed in tqdm(seeds, desc=scenario_name, disable=not verbose):
            # Set all random seeds for reproducibility
            set_seed(seed)

            X, coords, labels, ectopic_idx, intrinsic_idx, metadata = generate_scenario_data(
                scenario=scenario,
                n_spots=config.n_spots,
                n_genes=config.n_genes,
                n_ectopic=config.n_ectopic,
                n_intrinsic=config.n_intrinsic,
                random_state=seed,
            )

            data = prepare_data(X, coords, k=config.k_neighbors, device=config.device)
            scores_dict = compute_all_scores(X, coords, data, seed, config, methods)

            ectopic_mask = (labels == 1)
            intrinsic_mask = (labels == 2)

            result = {"scenario": scenario_name, "seed": seed}

            for method, score in scores_dict.items():
                # Ectopic AUC
                if ectopic_mask.sum() > 0:
                    mask = ~intrinsic_mask
                    if mask.sum() > 0 and ectopic_mask[mask].sum() > 0:
                        result[f"auc_ectopic_{method}"] = roc_auc_score(
                            ectopic_mask[mask].astype(int), score[mask]
                        )

                # Intrinsic AUC
                if intrinsic_mask.sum() > 0:
                    mask = ~ectopic_mask
                    if mask.sum() > 0 and intrinsic_mask[mask].sum() > 0:
                        result[f"auc_intrinsic_{method}"] = roc_auc_score(
                            intrinsic_mask[mask].astype(int), score[mask]
                        )

            all_results.append(result)

    return pd.DataFrame(all_results)


def run_noise_robustness(config, noise_levels=[0.0, 0.1, 0.2, 0.3, 0.5], seeds=None, verbose=True):
    """Test robustness to expression noise."""
    if seeds is None:
        seeds = config.seeds

    all_results = []
    scenario = SCENARIOS["baseline"]

    for noise_level in noise_levels:
        if verbose:
            print(f"\nNoise level: {noise_level}")

        for seed in tqdm(seeds, desc=f"noise={noise_level}", disable=not verbose):
            X, coords, labels, ectopic_idx, intrinsic_idx, metadata = generate_scenario_data(
                scenario=scenario,
                n_spots=config.n_spots,
                n_genes=config.n_genes,
                n_ectopic=config.n_ectopic,
                n_intrinsic=config.n_intrinsic,
                random_state=seed,
            )

            # Add noise (seeded for reproducibility)
            if noise_level > 0:
                rng = np.random.RandomState(seed)
                noise = rng.normal(0, noise_level * X.std(), X.shape)
                X = np.clip(X + noise, 0, None)

            data = prepare_data(X, coords, k=config.k_neighbors, device=config.device)
            scores_dict = compute_all_scores(X, coords, data, seed, config)

            ectopic_mask = (labels == 1)
            intrinsic_mask = (labels == 2)
            mask = ~intrinsic_mask

            result = {"noise_level": noise_level, "seed": seed}

            for method, score in scores_dict.items():
                if ectopic_mask[mask].sum() > 0:
                    result[f"auc_{method}"] = roc_auc_score(
                        ectopic_mask[mask].astype(int), score[mask]
                    )

            all_results.append(result)

    return pd.DataFrame(all_results)


def statistical_tests(results, baseline_method="inv_pos", alpha=0.05):
    """
    Perform statistical tests comparing inv_pos to other methods.

    Uses Bonferroni correction for multiple comparisons.
    Significance is determined by Wilcoxon signed-rank test (non-parametric,
    appropriate for bounded AUC values).
    """
    tests = []

    scenarios = results["scenario"].unique() if "scenario" in results.columns else [None]

    # Count total comparisons for Bonferroni correction
    n_comparisons = 0
    for scenario in scenarios:
        if scenario:
            data = results[results["scenario"] == scenario]
        else:
            data = results
        auc_cols = [c for c in data.columns if c.startswith("auc_ectopic_")]
        baseline_col = f"auc_ectopic_{baseline_method}"
        if baseline_col in data.columns:
            n_comparisons += len([c for c in auc_cols if c != baseline_col])

    # Bonferroni-corrected alpha
    alpha_corrected = alpha / max(n_comparisons, 1)

    for scenario in scenarios:
        if scenario:
            data = results[results["scenario"] == scenario]
        else:
            data = results

        auc_cols = [c for c in data.columns if c.startswith("auc_ectopic_")]
        baseline_col = f"auc_ectopic_{baseline_method}"

        if baseline_col not in data.columns:
            continue

        baseline_scores = data[baseline_col].dropna()

        for col in auc_cols:
            if col == baseline_col:
                continue

            method = col.replace("auc_ectopic_", "")
            other_scores = data[col].dropna()

            if len(baseline_scores) != len(other_scores):
                continue

            # Paired t-test (for reference)
            t_stat, t_pval = stats.ttest_rel(baseline_scores, other_scores)

            # Wilcoxon signed-rank test (primary test - non-parametric)
            try:
                w_stat, w_pval = stats.wilcoxon(baseline_scores, other_scores)
            except:
                w_stat, w_pval = np.nan, np.nan

            tests.append({
                "scenario": scenario,
                "method": method,
                "baseline_mean": baseline_scores.mean(),
                "method_mean": other_scores.mean(),
                "diff": baseline_scores.mean() - other_scores.mean(),
                "t_stat": t_stat,
                "t_pval": t_pval,
                "w_stat": w_stat,
                "w_pval": w_pval,
                "n_comparisons": n_comparisons,
                "alpha_corrected": alpha_corrected,
                # Use Wilcoxon p-value with Bonferroni correction for significance
                "significant": w_pval < alpha_corrected if not np.isnan(w_pval) else False,
                "significant_raw": w_pval < alpha if not np.isnan(w_pval) else False,
            })

    return pd.DataFrame(tests)


def plot_scenario_comparison(results, output_dir):
    """Create heatmap comparing methods across scenarios."""
    # Compute means
    pivot_data = []
    for scenario in results["scenario"].unique():
        scenario_data = results[results["scenario"] == scenario]
        row = {"scenario": scenario}
        for col in scenario_data.columns:
            if col.startswith("auc_ectopic_"):
                method = col.replace("auc_ectopic_", "")
                row[method] = scenario_data[col].mean()
        pivot_data.append(row)

    df = pd.DataFrame(pivot_data).set_index("scenario")

    # Rename methods
    df.columns = [METHOD_NAMES.get(c, c) for c in df.columns]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
                cbar_kws={"label": "Ectopic Detection AUC"}, ax=ax)
    ax.set_title("Ectopic Anomaly Detection Performance by Scenario", fontsize=14)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Scenario", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scenario_heatmap.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "scenario_heatmap.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_method_boxplots(results, output_dir):
    """Create boxplots comparing methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for idx, scenario in enumerate(results["scenario"].unique()[:4]):
        ax = axes[idx // 2, idx % 2]
        scenario_data = results[results["scenario"] == scenario]

        plot_data = []
        for col in scenario_data.columns:
            if col.startswith("auc_ectopic_"):
                method = col.replace("auc_ectopic_", "")
                for val in scenario_data[col].dropna():
                    plot_data.append({"Method": METHOD_NAMES.get(method, method), "AUC": val})

        plot_df = pd.DataFrame(plot_data)

        sns.boxplot(data=plot_df, x="Method", y="AUC", ax=ax)
        ax.set_title(f"Scenario: {scenario}", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "method_boxplots.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "method_boxplots.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_noise_robustness(results, output_dir):
    """Plot noise robustness curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [c.replace("auc_", "") for c in results.columns if c.startswith("auc_")]

    for method in methods:
        col = f"auc_{method}"
        if col not in results.columns:
            continue

        means = results.groupby("noise_level")[col].mean()
        stds = results.groupby("noise_level")[col].std()

        label = METHOD_NAMES.get(method, method)
        linestyle = "-" if method in SPATIAL_METHODS else "--"
        linewidth = 3 if method == "inv_pos" else 1.5

        ax.errorbar(means.index, means.values, yerr=stds.values,
                   label=label, linestyle=linestyle, linewidth=linewidth,
                   capsize=3, marker="o", markersize=6)

    ax.set_xlabel("Noise Level (fraction of std)", fontsize=12)
    ax.set_ylabel("Ectopic Detection AUC", fontsize=12)
    ax.set_title("Noise Robustness Comparison", fontsize=14)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "noise_robustness.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "noise_robustness.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_full_benchmark(config=None, seeds=None, output_dir="results/benchmark"):
    """Run full benchmark with all analyses."""
    os.makedirs(output_dir, exist_ok=True)
    if config is None:
        config = DEFAULT_CONFIG
    if seeds is None:
        seeds = config.seeds

    print("=" * 80)
    print("Full Benchmark: All Methods Comparison")
    print("=" * 80)
    print(f"Seeds: {len(seeds)} ({seeds[0]}-{seeds[-1]})")
    print(f"Output: {output_dir}")

    # 1. Synthetic data benchmark
    print("\n" + "=" * 60)
    print("1. Synthetic Data Benchmark")
    print("=" * 60)
    synthetic_results = run_synthetic_benchmark(config, seeds=seeds, verbose=True)
    synthetic_results.to_csv(os.path.join(output_dir, "synthetic_results.csv"), index=False)

    # 2. Statistical tests
    print("\n" + "=" * 60)
    print("2. Statistical Tests")
    print("=" * 60)
    stat_tests = statistical_tests(synthetic_results)
    stat_tests.to_csv(os.path.join(output_dir, "statistical_tests.csv"), index=False)
    print(stat_tests.to_string())

    # 3. Noise robustness
    print("\n" + "=" * 60)
    print("3. Noise Robustness Analysis")
    print("=" * 60)
    noise_results = run_noise_robustness(config, seeds=seeds, verbose=True)
    noise_results.to_csv(os.path.join(output_dir, "noise_robustness.csv"), index=False)

    # 4. Visualizations
    print("\n" + "=" * 60)
    print("4. Creating Visualizations")
    print("=" * 60)
    plot_scenario_comparison(synthetic_results, output_dir)
    print("  - scenario_heatmap.pdf")
    plot_method_boxplots(synthetic_results, output_dir)
    print("  - method_boxplots.pdf")
    plot_noise_robustness(noise_results, output_dir)
    print("  - noise_robustness.pdf")

    # 5. Summary
    print("\n" + "=" * 60)
    print("5. Summary")
    print("=" * 60)

    summary = synthetic_results.groupby("scenario").agg({
        c: ["mean", "std"] for c in synthetic_results.columns
        if c.startswith("auc_ectopic_")
    }).round(3)
    print("\nEctopic Detection AUC (mean ± std):")
    print(summary.to_string())

    print(f"\nResults saved to {output_dir}/")
    return synthetic_results, noise_results, stat_tests


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Main entry point for CLI compatibility."""
    if config is None:
        config = DEFAULT_CONFIG

    print("=" * 80)
    print("Experiment 15: Full Benchmark")
    print("=" * 80)

    output_dir = os.path.join(RESULTS_DIR, "benchmark")
    synthetic_results, noise_results, stat_tests = run_full_benchmark(
        config=config, seeds=config.seeds, output_dir=output_dir,
    )

    return synthetic_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=os.path.join(RESULTS_DIR, "benchmark"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        from config import QUICK_CONFIG
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    run_full_benchmark(config=config, seeds=config.seeds, output_dir=args.output)
