"""
Experiment 21: Anomaly Prevalence Sensitivity

Tests how detection performance degrades as the fraction of anomalous spots
decreases.  Low prevalence is the realistic regime for spatial transcriptomics
(< 5 % of spots are anomalous).

Hypothesis
----------
CASTOR (Inv_PosError for ectopic, PCA_Error for intrinsic) degrades
gracefully at low prevalence, outperforming baselines that implicitly assume
a non-negligible fraction of outliers.

Design
------
- Fix n_spots = 3000, vary the total anomaly prevalence in
  {0.5 %, 1 %, 3 %, 5 %, 10 %, 20 %}.
- Keep ectopic:intrinsic ratio at 1:3 (consistent with DEFAULT_CONFIG).
- For each (seed, prevalence_rate) pair, generate data, train, and evaluate.

Metrics: AUC and AUPRC (both reported) for ectopic and intrinsic detection.

Statistical controls
--------------------
- set_seed at the start of each run_single_seed call
- FDR correction (BH) across pairwise Wilcoxon tests per prevalence level
- Bootstrap 95 % CI in figures (not SD)
- Guard n_ectopic >= 3 / n_intrinsic >= 3 to avoid degenerate AUC
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
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
    compute_all_baselines,
    set_seed,
)
from core.evaluation import apply_fdr_correction

# Anomaly prevalence rates to test (fraction of total spots)
PREVALENCE_RATES = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2]


def run_single_seed(
    seed: int,
    config: ExperimentConfig,
    prevalence_rate: float,
    verbose: bool = False,
) -> dict:
    """Run prevalence sensitivity analysis for a single seed and prevalence."""
    set_seed(seed)

    n_spots = config.n_spots
    n_total_anomaly = max(6, int(n_spots * prevalence_rate))
    n_ectopic = max(3, int(n_total_anomaly * 0.25))
    n_intrinsic = max(3, n_total_anomaly - n_ectopic)

    # Generate data
    X, coords, labels, ectopic_idx, intrinsic_idx = generate_synthetic_data(
        n_spots=n_spots,
        n_genes=config.n_genes,
        n_ectopic=n_ectopic,
        n_intrinsic=n_intrinsic,
        n_modules=config.n_modules,
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

    # Compute scores
    scores_inv = compute_scores(
        model, data["x_tensor"], data["coords_tensor"],
        data["edge_index"], random_state=seed,
    )
    scores_baseline = compute_all_baselines(
        data["X_norm"], coords, k=config.k_neighbors, random_state=seed,
    )

    # Key scores
    s_pos = scores_inv["s_pos"]
    s_pca = scores_baseline["pca_error"]
    s_neighbor = scores_baseline["neighbor_diff"]
    s_lisa = scores_baseline["lisa"]
    s_lof = scores_baseline["lof"]
    s_if = scores_baseline["isolation_forest"]

    # Masks
    ectopic_mask = labels == 1
    intrinsic_mask = labels == 2

    result = {
        "seed": seed,
        "prevalence_rate": prevalence_rate,
        "n_ectopic_actual": int(ectopic_mask.sum()),
        "n_intrinsic_actual": int(intrinsic_mask.sum()),
    }

    # Compute AUC and AUPRC for each method
    method_scores = {
        "pos": s_pos,
        "pca": s_pca,
        "neighbor": s_neighbor,
        "lisa": s_lisa,
        "lof": s_lof,
        "if": s_if,
    }

    for method_name, score in method_scores.items():
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        # Ectopic detection
        if ectopic_mask.sum() >= 3 and (~ectopic_mask).sum() > 0:
            y_ect = ectopic_mask.astype(int)
            result[f"auc_ectopic_{method_name}"] = roc_auc_score(y_ect, score)
            result[f"ap_ectopic_{method_name}"] = average_precision_score(y_ect, score)
        else:
            result[f"auc_ectopic_{method_name}"] = np.nan
            result[f"ap_ectopic_{method_name}"] = np.nan

        # Intrinsic detection
        if intrinsic_mask.sum() >= 3 and (~intrinsic_mask).sum() > 0:
            y_int = intrinsic_mask.astype(int)
            result[f"auc_intrinsic_{method_name}"] = roc_auc_score(y_int, score)
            result[f"ap_intrinsic_{method_name}"] = average_precision_score(y_int, score)
        else:
            result[f"auc_intrinsic_{method_name}"] = np.nan
            result[f"ap_intrinsic_{method_name}"] = np.nan

    return result


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run prevalence sensitivity analysis across all seeds and prevalence rates."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []
    total = len(config.seeds) * len(PREVALENCE_RATES)
    iterator = tqdm(total=total, desc="Prevalence Sensitivity") if verbose else None

    for seed in config.seeds:
        for prevalence_rate in PREVALENCE_RATES:
            try:
                result = run_single_seed(seed, config, prevalence_rate, verbose=False)
                all_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Seed {seed}, prevalence {prevalence_rate} failed: {e}")
            if iterator is not None:
                iterator.update(1)

    if iterator is not None:
        iterator.close()

    if not all_results:
        raise RuntimeError("All seeds failed in exp21_prevalence_sensitivity")

    results = pd.DataFrame(all_results)

    # FDR correction: per prevalence level, test CASTOR vs each baseline
    baselines = ["pca", "neighbor", "lisa", "lof", "if"]
    for prev_rate in PREVALENCE_RATES:
        subset = results[results["prevalence_rate"] == prev_rate]
        if len(subset) < 10:
            continue

        raw_pvals = []
        test_keys = []
        for bl in baselines:
            col_castor = "auc_ectopic_pos"
            col_bl = f"auc_ectopic_{bl}"
            vals_c = subset[col_castor].dropna().values
            vals_b = subset[col_bl].dropna().values
            n_valid = min(len(vals_c), len(vals_b))
            if n_valid >= 10:
                _, pval = stats.wilcoxon(vals_c[:n_valid], vals_b[:n_valid])
                raw_pvals.append(pval)
                test_keys.append(f"pval_ectopic_pos_vs_{bl}_prev{prev_rate}")

        if raw_pvals:
            fdr_result = apply_fdr_correction(raw_pvals, alpha=0.05, method="fdr_bh")
            for key, adj_p in zip(test_keys, fdr_result["p_adjusted"]):
                # Store as metadata in the first row of this prevalence group
                idx = subset.index[0]
                results.loc[idx, key] = adj_p

    if verbose:
        print("\n" + "=" * 60)
        print("Prevalence Sensitivity Summary")
        print("=" * 60)
        for prev_rate in PREVALENCE_RATES:
            subset = results[results["prevalence_rate"] == prev_rate]
            n_ect = subset["n_ectopic_actual"].mean()
            n_int = subset["n_intrinsic_actual"].mean()
            auc_pos = subset["auc_ectopic_pos"].mean()
            auc_pca = subset["auc_intrinsic_pca"].mean()
            print(
                f"  prev={prev_rate:.3f}  n_ect={n_ect:.0f}  n_int={n_int:.0f}  "
                f"AUC_ect_pos={auc_pos:.3f}  AUC_int_pca={auc_pca:.3f}"
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "exp21_prevalence_sensitivity.csv"),
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
