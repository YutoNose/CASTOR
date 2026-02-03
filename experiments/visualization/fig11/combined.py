"""
Figure 11: Mixed Anomaly Overlap

Shows how the dual-axis framework behaves when ectopic and intrinsic
anomalies co-occur.

Panels:
(a) Grouped bar chart: Ectopic AUC by overlap fraction
(b) Stacked bar: dominance fraction in mixed spots
(c) Scatter: s_pos vs s_pca at overlap=0.5 (representative seed)

Data source: exp22_mixed_anomaly.csv
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
    """Load mixed anomaly results."""
    path = RESULTS_DIR / "exp22_mixed_anomaly.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    return pd.read_csv(path)


def _draw_bar_panel(ax, df):
    """Panel (a): Ectopic AUC by overlap fraction."""
    overlaps = sorted(df["overlap_fraction"].unique())
    x = np.arange(len(overlaps))
    width = 0.35

    # Inv_PosError ectopic AUC
    means_pos, errs_lo_pos, errs_hi_pos = [], [], []
    means_pca, errs_lo_pca, errs_hi_pca = [], [], []

    for overlap in overlaps:
        subset = df[df["overlap_fraction"] == overlap]
        m, lo, hi = _bootstrap_ci(subset["auc_ectopic_pos"])
        means_pos.append(m)
        errs_lo_pos.append(lo)
        errs_hi_pos.append(hi)

        m, lo, hi = _bootstrap_ci(subset["auc_intrinsic_pca"])
        means_pca.append(m)
        errs_lo_pca.append(lo)
        errs_hi_pca.append(hi)

    ax.bar(x - width / 2, means_pos, width, label="Inv_PosError (ectopic)",
           color=COLORS["inv_pos"],
           yerr=[errs_lo_pos, errs_hi_pos], capsize=2, error_kw={"linewidth": 0.5})
    ax.bar(x + width / 2, means_pca, width, label="PCA_Error (intrinsic)",
           color=COLORS["pca_error"],
           yerr=[errs_lo_pca, errs_hi_pca], capsize=2, error_kw={"linewidth": 0.5})

    ax.set_xticks(x)
    ax.set_xticklabels([f"{o:.0%}" for o in overlaps], fontsize=6)
    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=5, loc="lower left")


def _draw_dominance_panel(ax, df):
    """Panel (b): Dominance fraction in mixed spots."""
    overlaps = sorted(df["overlap_fraction"].unique())
    # Skip overlap=0 (no mixed spots)
    overlaps = [o for o in overlaps if o > 0]

    x = np.arange(len(overlaps))
    frac_ect = []
    frac_int = []

    for overlap in overlaps:
        subset = df[df["overlap_fraction"] == overlap]
        frac_ect.append(subset["frac_ectopic_dominant"].mean())
        frac_int.append(subset["frac_intrinsic_dominant"].mean())

    frac_ect = np.array(frac_ect)
    frac_int = np.array(frac_int)

    ax.bar(x, frac_ect, label="Ectopic-dominant", color=COLORS["ectopic"])
    ax.bar(x, frac_int, bottom=frac_ect, label="Intrinsic-dominant",
           color=COLORS["intrinsic"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{o:.0%}" for o in overlaps], fontsize=6)
    ax.set_xlabel("Overlap fraction")
    ax.set_ylabel("Fraction of mixed spots")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=5)


def _draw_scatter_panel(ax, df):
    """Panel (c): Placeholder scatter showing AUC relationship at overlap=0.5."""
    # Use per-seed data to show correlation between pos and pca AUC
    subset = df[df["overlap_fraction"] == 0.5]
    if len(subset) == 0:
        ax.text(0.5, 0.5, "No data for overlap=0.5", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        return

    ax.scatter(
        subset["auc_ectopic_pos"], subset["auc_intrinsic_pca"],
        s=15, alpha=0.6, color=COLORS["inv_pos"], edgecolors="white",
        linewidths=0.3,
    )

    ax.set_xlabel("Ectopic AUC (Inv_PosError)")
    ax.set_ylabel("Intrinsic AUC (PCA_Error)")
    ax.set_title("overlap = 50%", fontsize=7)

    # Add reference line
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)


def main():
    """Generate combined Figure 11."""
    set_nature_style()
    df = _load_data()

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, SINGLE_COL * 0.9))

    _draw_bar_panel(axes[0], df)
    add_panel_label(axes[0], "a")

    _draw_dominance_panel(axes[1], df)
    add_panel_label(axes[1], "b")

    _draw_scatter_panel(axes[2], df)
    add_panel_label(axes[2], "c")

    plt.tight_layout()
    save_figure(fig, "fig11/combined")
    plt.close(fig)


if __name__ == "__main__":
    main()
