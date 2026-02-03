"""
Experiment 23: Graph Construction Sensitivity

Tests whether detection performance is robust to the choice of spatial graph
construction method.

Hypothesis
----------
CASTOR's anomaly detection performance is not critically dependent on a
specific graph construction method.  k-NN, Delaunay triangulation, and
radius-based graphs should yield comparable AUC, as the inverse prediction
task fundamentally depends on local spatial context rather than exact graph
topology.

Design
------
- For each seed, generate synthetic data ONCE and reuse across all graph
  configurations (isolates graph effect from data randomness, same pattern
  as exp07_ablation.py).
- Graph types: k-NN (k=5,10,15,20,30), Delaunay, radius (r auto-calibrated
  to match median degree of k=15 graph).

Statistical controls
--------------------
- set_seed at the start of each configuration
- Data generated once per seed, graph construction varies
- FDR correction (BH) across graph type comparisons per metric
- Mean degree reported for each graph type
- Both AUC and AUPRC computed
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

from config import ExperimentConfig, DEFAULT_CONFIG
from core import (
    generate_synthetic_data,
    prepare_data,
    build_spatial_graph,
    build_delaunay_graph,
    build_radius_graph,
    InversePredictionModel,
    train_model,
    compute_scores,
    compute_all_baselines,
    set_seed,
)
from core.evaluation import apply_fdr_correction

# k-NN values to test
KNN_VALUES = [5, 10, 15, 20, 30]

# Graph configurations: (graph_type, graph_param_or_None)
# Delaunay and radius are added programmatically
GRAPH_CONFIGS = [(f"knn_k{k}", "knn", k) for k in KNN_VALUES]
# Delaunay and radius appended in run()


def _compute_mean_degree(edge_index) -> float:
    """Compute mean node degree from edge_index [2, n_edges]."""
    if edge_index.shape[1] == 0:
        return 0.0
    row = edge_index[0].numpy()
    n_nodes = int(row.max()) + 1
    degrees = np.bincount(row, minlength=n_nodes)
    return float(degrees.mean())


def _calibrate_radius(coords: np.ndarray, target_k: int = 15) -> float:
    """
    Calibrate radius to match the median degree of a k-NN graph.

    Uses the median k-NN distance as the radius, which produces a graph
    with approximately the same average connectivity as k-NN(k=target_k).
    """
    nn = NearestNeighbors(n_neighbors=target_k + 1)
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords)
    # Use median distance to the k-th neighbor as the radius
    return float(np.median(distances[:, target_k]))


def run_single_config(
    X: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    seed: int,
    config: ExperimentConfig,
    graph_type: str,
    graph_param,
    data_normalized: dict,
    verbose: bool = False,
) -> dict:
    """Run detection with a specific graph configuration.

    Parameters
    ----------
    X, coords, labels : arrays
        Original data (for baselines).
    seed : int
        Random seed.
    config : ExperimentConfig
        Experiment configuration.
    graph_type : str
        One of 'knn', 'delaunay', 'radius'.
    graph_param : int or float or None
        Parameter for graph construction (k for knn, radius for radius).
    data_normalized : dict
        Pre-computed normalized data from prepare_data (we override edge_index).
    """
    set_seed(seed)

    # Build graph
    if graph_type == "knn":
        edge_index = build_spatial_graph(coords, k=graph_param)
    elif graph_type == "delaunay":
        edge_index = build_delaunay_graph(coords)
    elif graph_type == "radius":
        edge_index = build_radius_graph(coords, radius=graph_param)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    edge_index = edge_index.to(config.device)
    mean_degree = _compute_mean_degree(edge_index.cpu())
    n_edges = edge_index.shape[1]

    # Train model with this graph
    model = InversePredictionModel(
        in_dim=data_normalized["n_genes"],
        hid_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(config.device)

    model = train_model(
        model,
        data_normalized["x_tensor"],
        data_normalized["coords_tensor"],
        edge_index,
        n_epochs=config.n_epochs,
        lr=config.learning_rate,
        lambda_pos=config.lambda_pos,
        lambda_self=config.lambda_self,
        verbose=verbose,
    )

    # Compute scores
    scores_inv = compute_scores(
        model, data_normalized["x_tensor"], data_normalized["coords_tensor"],
        edge_index, random_state=seed,
    )

    # Baselines use their own k-NN internally
    scores_baseline = compute_all_baselines(
        data_normalized["X_norm"], coords, k=config.k_neighbors, random_state=seed,
    )

    s_pos = scores_inv["s_pos"]
    s_pca = scores_baseline["pca_error"]
    s_neighbor = scores_baseline["neighbor_diff"]

    # Masks
    ectopic_mask = labels == 1
    intrinsic_mask = labels == 2

    result = {
        "seed": seed,
        "graph_type": graph_type,
        "graph_param": graph_param if graph_param is not None else np.nan,
        "n_edges": n_edges,
        "mean_degree": mean_degree,
    }

    method_scores = {
        "pos": s_pos,
        "pca": s_pca,
        "neighbor": s_neighbor,
    }

    for method_name, score in method_scores.items():
        score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

        if ectopic_mask.sum() >= 3 and (~ectopic_mask).sum() > 0:
            y_ect = ectopic_mask.astype(int)
            result[f"auc_ectopic_{method_name}"] = roc_auc_score(y_ect, score)
            result[f"ap_ectopic_{method_name}"] = average_precision_score(y_ect, score)
        else:
            result[f"auc_ectopic_{method_name}"] = np.nan
            result[f"ap_ectopic_{method_name}"] = np.nan

        if intrinsic_mask.sum() >= 3 and (~intrinsic_mask).sum() > 0:
            y_int = intrinsic_mask.astype(int)
            result[f"auc_intrinsic_{method_name}"] = roc_auc_score(y_int, score)
            result[f"ap_intrinsic_{method_name}"] = average_precision_score(y_int, score)
        else:
            result[f"auc_intrinsic_{method_name}"] = np.nan
            result[f"ap_intrinsic_{method_name}"] = np.nan

    return result


def run(config: ExperimentConfig = None, verbose: bool = True) -> pd.DataFrame:
    """Run graph sensitivity analysis across all seeds and graph configurations."""
    if config is None:
        config = DEFAULT_CONFIG

    all_results = []

    for seed in tqdm(config.seeds, desc="Graph Sensitivity", disable=not verbose):
        try:
            # Generate data ONCE per seed
            set_seed(seed)
            X, coords, labels, _, _ = generate_synthetic_data(
                n_spots=config.n_spots,
                n_genes=config.n_genes,
                n_ectopic=config.n_ectopic,
                n_intrinsic=config.n_intrinsic,
                n_modules=config.n_modules,
                random_state=seed,
            )

            # Normalize once (prepare_data with default k; we override edge_index)
            data_normalized = prepare_data(
                X, coords, k=config.k_neighbors, device=config.device,
            )
        except Exception as e:
            if verbose:
                print(f"Seed {seed} data generation failed: {e}")
            continue

        # Calibrate radius based on this seed's data
        radius = _calibrate_radius(coords, target_k=config.k_neighbors)

        # Build full config list including Delaunay and radius
        configs = [
            ("knn", k) for k in KNN_VALUES
        ] + [
            ("delaunay", None),
            ("radius", radius),
        ]

        for graph_type, graph_param in configs:
            try:
                result = run_single_config(
                    X, coords, labels, seed, config,
                    graph_type, graph_param, data_normalized,
                    verbose=False,
                )
                all_results.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Seed {seed}, {graph_type}({graph_param}) failed: {e}")

    if not all_results:
        raise RuntimeError("All seeds failed in exp23_graph_sensitivity")

    results = pd.DataFrame(all_results)

    # FDR correction: compare each graph type to default k-NN(k=15)
    ref_type = "knn"
    ref_param = 15
    ref_mask = (results["graph_type"] == ref_type) & (results["graph_param"] == ref_param)

    raw_pvals = []
    test_keys = []

    for gt in results["graph_type"].unique():
        param_values = results[results["graph_type"] == gt]["graph_param"].unique()
        for gp in param_values:
            if gt == ref_type and gp == ref_param:
                continue

            comp_mask = (results["graph_type"] == gt)
            if isinstance(gp, float) and np.isnan(gp):
                comp_mask = comp_mask & (results["graph_param"].isna())
            else:
                comp_mask = comp_mask & (results["graph_param"] == gp)

            # Match seeds
            ref_seeds = set(results.loc[ref_mask, "seed"])
            comp_seeds = set(results.loc[comp_mask, "seed"])
            common_seeds = sorted(ref_seeds & comp_seeds)

            if len(common_seeds) < 10:
                continue

            ref_vals = results.loc[ref_mask & results["seed"].isin(common_seeds)].sort_values("seed")["auc_ectopic_pos"].values
            comp_vals = results.loc[comp_mask & results["seed"].isin(common_seeds)].sort_values("seed")["auc_ectopic_pos"].values

            n_valid = min(len(ref_vals), len(comp_vals))
            if n_valid >= 10:
                _, pval = stats.wilcoxon(ref_vals[:n_valid], comp_vals[:n_valid])
                raw_pvals.append(pval)
                test_keys.append(f"pval_knn15_vs_{gt}_{gp}")

    if raw_pvals:
        fdr_result = apply_fdr_correction(raw_pvals, alpha=0.05, method="fdr_bh")
        # Store in summary (not per-row since graph types vary)
        if verbose:
            print("\nFDR-corrected p-values (vs k-NN k=15):")
            for key, adj_p, reject in zip(
                test_keys, fdr_result["p_adjusted"], fdr_result["reject"]
            ):
                sig = "*" if reject else ""
                print(f"  {key}: p_adj={adj_p:.4e} {sig}")

    if verbose:
        print("\n" + "=" * 60)
        print("Graph Sensitivity Summary")
        print("=" * 60)

        # Fill NaN graph_param for Delaunay so groupby includes it
        results_display = results.copy()
        results_display["graph_param"] = results_display["graph_param"].fillna(-1)
        summary = results_display.groupby(["graph_type", "graph_param"]).agg(
            auc_mean=("auc_ectopic_pos", "mean"),
            auc_std=("auc_ectopic_pos", "std"),
            degree_mean=("mean_degree", "mean"),
        ).reset_index()

        for _, row in summary.iterrows():
            param_str = "N/A" if row["graph_param"] == -1 else f"{row['graph_param']}"
            print(
                f"  {row['graph_type']}(param={param_str})  "
                f"AUC={row['auc_mean']:.3f}±{row['auc_std']:.3f}  "
                f"degree={row['degree_mean']:.1f}"
            )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "exp23_graph_sensitivity.csv"),
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
