"""
Figure S8: False Positive Spatial Characterization

Shows that FP spots from Inv_PosError occur at spatial transition zones.

Panels:
(a) Violin plot: expression gradient magnitude (FP vs TN)
(b) Violin plot: neighbor heterogeneity (FP vs TN)
(c) Bar chart: effect sizes with bootstrap 95% CI
(d) Summary table of FDR-corrected p-values across seeds

Data source: exp24_fp_characterization.csv
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
    """Load FP characterization results."""
    path = RESULTS_DIR / "exp24_fp_characterization.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    df = pd.read_csv(path)
    # Keep only valid seeds (with sufficient FP)
    return df.dropna(subset=["gradient_pval_fdr"])


def _draw_violin_panel(ax, df, feature, ylabel):
    """Draw violin plot comparing FP vs TN for a feature."""
    fp_vals = df[f"{feature}_mean_fp"].values
    tn_vals = df[f"{feature}_mean_tn"].values

    parts = ax.violinplot(
        [fp_vals, tn_vals],
        positions=[0, 1],
        showmeans=True,
        showextrema=False,
    )

    fp_color = COLORS["ectopic"]
    tn_color = COLORS["normal"]

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(fp_color if i == 0 else tn_color)
        pc.set_alpha(0.7)

    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(0.8)

    # Overlay individual points
    jitter = 0.05
    rng = np.random.RandomState(42)
    ax.scatter(
        rng.normal(0, jitter, len(fp_vals)), fp_vals,
        s=8, alpha=0.4, color=fp_color, edgecolors="white", linewidths=0.3,
    )
    ax.scatter(
        rng.normal(1, jitter, len(tn_vals)), tn_vals,
        s=8, alpha=0.4, color=tn_color, edgecolors="white", linewidths=0.3,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["FP", "TN"])
    ax.set_ylabel(ylabel)

    # Add p-value annotation
    median_p = df[f"{feature}_pval_fdr"].median()
    sig_marker = ""
    if median_p < 0.001:
        sig_marker = "***"
    elif median_p < 0.01:
        sig_marker = "**"
    elif median_p < 0.05:
        sig_marker = "*"

    if sig_marker:
        y_max = max(fp_vals.max(), tn_vals.max())
        ax.plot([0, 0, 1, 1], [y_max * 1.02, y_max * 1.05, y_max * 1.05, y_max * 1.02],
                color="black", linewidth=0.5)
        ax.text(0.5, y_max * 1.06, sig_marker, ha="center", fontsize=7)


def _draw_effect_size_panel(ax, df):
    """Panel (c): Effect sizes with bootstrap 95% CI."""
    features = {
        "gradient": "Gradient\nmagnitude",
        "heterogeneity": "Neighbor\nheterogeneity",
        "moran": "Local\nMoran's I",
    }

    x = np.arange(len(features))
    means, errs_lo, errs_hi = [], [], []
    bar_colors = [COLORS["ectopic"], COLORS["intrinsic"], COLORS["inv_pos"]]

    for feat_key in features:
        col = f"{feat_key}_effect_size"
        m, lo, hi = _bootstrap_ci(df[col])
        means.append(m)
        errs_lo.append(lo)
        errs_hi.append(hi)

    ax.bar(x, means, color=bar_colors,
           yerr=[errs_lo, errs_hi], capsize=3, error_kw={"linewidth": 0.5})
    ax.set_xticks(x)
    ax.set_xticklabels(list(features.values()), fontsize=6)
    ax.set_ylabel("Effect size (rank-biserial)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Add effect size interpretation lines
    for threshold, label in [(0.1, "small"), (0.3, "medium"), (0.5, "large")]:
        ax.axhline(threshold, color="gray", linewidth=0.3, linestyle=":", alpha=0.5)
        ax.axhline(-threshold, color="gray", linewidth=0.3, linestyle=":", alpha=0.5)


def _draw_summary_panel(ax, df):
    """Panel (d): Summary table of statistics."""
    ax.axis("off")

    features = ["gradient", "heterogeneity", "moran"]
    labels = ["Gradient mag.", "Neighbor het.", "Local Moran's I"]

    table_data = []
    for feat, label in zip(features, labels):
        effect = df[f"{feat}_effect_size"].mean()
        p_med = df[f"{feat}_pval_fdr"].median()
        n_sig = (df[f"{feat}_pval_fdr"] < 0.05).sum()
        n_total = len(df)
        table_data.append([
            label,
            f"{effect:.3f}",
            f"{p_med:.2e}",
            f"{n_sig}/{n_total}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=["Feature", "Effect size", "Median p (FDR)", "Sig. seeds"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.4)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor("#E8E8E8")
        table[0, j].set_text_props(fontweight="bold")

    ax.set_title("Summary across seeds", fontsize=7, pad=10)


def main():
    """Generate combined Figure S8."""
    set_nature_style()
    df = _load_data()

    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL, SINGLE_COL * 0.9),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})

    _draw_violin_panel(axes[0], df, "gradient", "Gradient magnitude\n(mean per seed)")
    add_panel_label(axes[0], "a")

    _draw_violin_panel(axes[1], df, "heterogeneity", "Neighbor heterogeneity\n(mean per seed)")
    add_panel_label(axes[1], "b")

    _draw_effect_size_panel(axes[2], df)
    add_panel_label(axes[2], "c")

    _draw_summary_panel(axes[3], df)
    add_panel_label(axes[3], "d")

    plt.tight_layout()
    save_figure(fig, "figS8/combined")
    plt.close(fig)


if __name__ == "__main__":
    main()
