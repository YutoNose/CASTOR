"""
Figure S7: Graph Construction Sensitivity

Shows that detection performance is robust to graph construction method.

Panels:
(a) Bar chart: AUC by graph type with bootstrap 95% CI
(b) Box plot: degree distribution by graph type
(c) Line plot: AUC vs k for k-NN only

Data source: exp23_graph_sensitivity.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, SINGLE_COL, DOUBLE_COL, RESULTS_DIR,
)


def _bootstrap_ci(values, confidence=0.95, n_bootstrap=10000, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    values = values.dropna().values if hasattr(values, 'dropna') else np.asarray(values)
    n = len(values)
    if n < 2:
        return values.mean() if n == 1 else np.nan, 0.0, 0.0
    boot_means = np.array([rng.choice(values, size=n, replace=True).mean()
                           for _ in range(n_bootstrap)])
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    mean_val = values.mean()
    return mean_val, mean_val - ci_lower, ci_upper - mean_val


def _load_data():
    """Load graph sensitivity results."""
    path = RESULTS_DIR / "exp23_graph_sensitivity.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    return pd.read_csv(path)


def _get_graph_label(row):
    """Create a display label from graph type and param."""
    gt = row["graph_type"]
    gp = row["graph_param"]
    if gt == "knn":
        return f"k-NN\n(k={int(gp)})"
    elif gt == "delaunay":
        return "Delaunay"
    elif gt == "radius":
        return f"Radius\n(r={gp:.2f})"
    return str(gt)


def _draw_bar_panel(ax, df):
    """Panel (a): AUC by graph type with bootstrap 95% CI."""
    # Create unique graph configurations
    configs = df.groupby(["graph_type", "graph_param"]).size().reset_index()
    configs = configs.sort_values(["graph_type", "graph_param"])

    labels = []
    means = []
    errs_lo = []
    errs_hi = []
    bar_colors = []

    for _, row in configs.iterrows():
        gt = row["graph_type"]
        gp = row["graph_param"]
        mask = (df["graph_type"] == gt)
        if not np.isnan(gp):
            mask = mask & (df["graph_param"] == gp)

        subset = df[mask]
        m, lo, hi = _bootstrap_ci(subset["auc_ectopic_pos"])
        labels.append(_get_graph_label(row))
        means.append(m)
        errs_lo.append(lo)
        errs_hi.append(hi)

        if gt == "knn" and gp == 15:
            bar_colors.append(COLORS["inv_pos"])
        elif gt == "delaunay":
            bar_colors.append("#009E73")
        elif gt == "radius":
            bar_colors.append("#E69F00")
        else:
            bar_colors.append("#56B4E9")

    x = np.arange(len(labels))
    ax.bar(x, means, color=bar_colors,
           yerr=[errs_lo, errs_hi], capsize=2, error_kw={"linewidth": 0.5})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("AUROC (ectopic)")
    ax.set_ylim(0.5, 1.05)


def _draw_degree_panel(ax, df):
    """Panel (b): Degree distribution box plot."""
    configs = df.groupby(["graph_type", "graph_param"]).size().reset_index()
    configs = configs.sort_values(["graph_type", "graph_param"])

    data = []
    labels = []
    for _, row in configs.iterrows():
        gt = row["graph_type"]
        gp = row["graph_param"]
        mask = (df["graph_type"] == gt)
        if not np.isnan(gp):
            mask = mask & (df["graph_param"] == gp)

        subset = df[mask]
        data.append(subset["mean_degree"].values)
        labels.append(_get_graph_label(row))

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 0.8},
                    boxprops={"linewidth": 0.5},
                    whiskerprops={"linewidth": 0.5},
                    capprops={"linewidth": 0.5},
                    flierprops={"markersize": 2})

    for patch in bp["boxes"]:
        patch.set_facecolor("#D3D3D3")

    ax.set_ylabel("Mean node degree")
    ax.tick_params(axis="x", labelsize=5)


def _draw_knn_line_panel(ax, df):
    """Panel (c): AUC vs k for k-NN graphs only."""
    knn_df = df[df["graph_type"] == "knn"]
    k_values = sorted(knn_df["graph_param"].unique())

    for metric_col, label, color in [
        ("auc_ectopic_pos", "Ectopic (Inv_PosError)", COLORS["inv_pos"]),
        ("auc_intrinsic_pca", "Intrinsic (PCA_Error)", COLORS["pca_error"]),
    ]:
        if metric_col not in knn_df.columns:
            continue

        means, ci_lows, ci_highs = [], [], []
        for k in k_values:
            subset = knn_df[knn_df["graph_param"] == k]
            m, lo, hi = _bootstrap_ci(subset[metric_col])
            means.append(m)
            ci_lows.append(m - lo)
            ci_highs.append(m + hi)

        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        ax.plot(k_values, means, "o-", color=color, label=label,
                markersize=3, linewidth=1.0)
        ax.fill_between(k_values, ci_lows, ci_highs, alpha=0.15, color=color)

    ax.set_xlabel("k (number of neighbors)")
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=5)


def main():
    """Generate combined Figure S7."""
    set_nature_style()
    df = _load_data()

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.9))

    _draw_bar_panel(axes[0], df)
    add_panel_label(axes[0], "a")

    _draw_degree_panel(axes[1], df)
    add_panel_label(axes[1], "b")

    _draw_knn_line_panel(axes[2], df)
    add_panel_label(axes[2], "c")

    plt.tight_layout()
    save_figure(fig, "figS7/combined")
    plt.close(fig)


if __name__ == "__main__":
    main()
