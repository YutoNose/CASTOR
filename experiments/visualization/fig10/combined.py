"""
Figure 10: Anomaly Prevalence Sensitivity

Shows how detection performance degrades as anomaly prevalence decreases.

Panels:
(a) Line plot: AUC vs prevalence rate, one line per method, bootstrap 95% CI
(b) Same for AUPRC
(c) Heatmap: AUC degradation relative to highest prevalence (delta AUC)

Data source: exp21_prevalence_sensitivity.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, SINGLE_COL, DOUBLE_COL, RESULTS_DIR,
)


# Methods to plot
METHODS = {
    'pos': ('Inv_PosError', COLORS['inv_pos']),
    'pca': ('PCA_Error', COLORS['pca_error']),
    'neighbor': ('Neighbor_Diff', COLORS['neighbor_diff']),
    'lisa': ('LISA', COLORS['lisa']),
    'lof': ('LOF', COLORS['lof']),
    'if': ('Isolation_Forest', COLORS['isolation_forest']),
}


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
    """Load prevalence sensitivity results."""
    path = RESULTS_DIR / "exp21_prevalence_sensitivity.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    return pd.read_csv(path)


def _draw_line_panel(ax, df, metric_prefix, ylabel):
    """Draw a line plot with bootstrap 95% CI shading."""
    prevalence_rates = sorted(df["prevalence_rate"].unique())

    for method_key, (label, color) in METHODS.items():
        col = f"{metric_prefix}_{method_key}"
        if col not in df.columns:
            continue

        means, ci_lows, ci_highs = [], [], []
        for prev_rate in prevalence_rates:
            subset = df[df["prevalence_rate"] == prev_rate]
            mean, lo, hi = _bootstrap_ci(subset[col])
            means.append(mean)
            ci_lows.append(mean - lo)
            ci_highs.append(mean + hi)

        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        ax.plot(prevalence_rates, means, 'o-', color=color, label=label,
                markersize=3, linewidth=1.0)
        ax.fill_between(prevalence_rates, ci_lows, ci_highs,
                        alpha=0.15, color=color)

    ax.set_xlabel("Anomaly prevalence (fraction)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.legend(fontsize=5, loc="lower right", framealpha=0.8)


def _draw_heatmap(ax, df):
    """Draw AUC degradation heatmap (delta from highest prevalence)."""
    prevalence_rates = sorted(df["prevalence_rate"].unique())
    max_prev = max(prevalence_rates)

    method_keys = list(METHODS.keys())
    method_labels = [METHODS[k][0] for k in method_keys]

    delta_matrix = np.zeros((len(method_keys), len(prevalence_rates)))

    for j, prev_rate in enumerate(prevalence_rates):
        subset = df[df["prevalence_rate"] == prev_rate]
        ref_subset = df[df["prevalence_rate"] == max_prev]
        for i, method_key in enumerate(method_keys):
            col = f"auc_ectopic_{method_key}"
            if col in df.columns:
                current = subset[col].mean()
                reference = ref_subset[col].mean()
                delta_matrix[i, j] = current - reference

    im = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto",
                   vmin=-0.3, vmax=0.05)
    ax.set_xticks(range(len(prevalence_rates)))
    ax.set_xticklabels([f"{p:.1%}" for p in prevalence_rates], fontsize=5, rotation=45)
    ax.set_yticks(range(len(method_labels)))
    ax.set_yticklabels(method_labels, fontsize=6)
    ax.set_xlabel("Prevalence")
    ax.set_title(r"$\Delta$AUC (vs. highest prevalence)", fontsize=7)

    # Add text annotations
    for i in range(len(method_keys)):
        for j in range(len(prevalence_rates)):
            val = delta_matrix[i, j]
            color = "white" if abs(val) > 0.15 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=4, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)


def main():
    """Generate combined Figure 10."""
    set_nature_style()
    df = _load_data()

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.9))

    # (a) AUC vs prevalence
    _draw_line_panel(axes[0], df, "auc_ectopic", "AUROC (ectopic detection)")
    add_panel_label(axes[0], "a")

    # (b) AUPRC vs prevalence
    _draw_line_panel(axes[1], df, "ap_ectopic", "AUPRC (ectopic detection)")
    add_panel_label(axes[1], "b")

    # (c) Delta AUC heatmap
    _draw_heatmap(axes[2])
    add_panel_label(axes[2], "c")

    plt.tight_layout()
    save_figure(fig, "fig10/combined")
    plt.close(fig)


if __name__ == "__main__":
    main()
