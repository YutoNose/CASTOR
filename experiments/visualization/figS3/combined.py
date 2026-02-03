"""
Supplementary Figure S3: Predicted Position Interpretability

Validates that for transplanted spots (tumor expression at normal positions),
the model's predicted position displacement vector points toward the
donor (tumor) location.

Key metric: cosine similarity between
  - displacement vector (predicted - actual)
  - donor direction vector (donor - actual)

Panels:
(a) Spatial map: arrows showing predicted displacement vs donor direction
(b) Cosine similarity distribution with Wilcoxon signed-rank test vs 0
(c) Per-sample mean cosine similarity
(d) Cosine similarity vs transplant distance (donor-actual distance)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
VIS_DIR = Path(__file__).resolve().parent.parent
EXP_DIR = VIS_DIR.parent  # .../experiments/
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR.parent / "figures" / "figS3"

sys.path.insert(0, str(VIS_DIR))
from common import set_nature_style, save_figure, add_panel_label, COLORS

from figS3 import panel_a, panel_b, panel_c, panel_d


def load_data():
    csv_path = RESULTS_DIR / "exp18_interpretability.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run exp18_interpretability.py first: {csv_path}")
    df = pd.read_csv(csv_path)

    # Compute cosine similarity between displacement and donor direction
    disp_x = df["pred_x"] - df["actual_x"]
    disp_y = df["pred_y"] - df["actual_y"]
    donor_dir_x = df["donor_x"] - df["actual_x"]
    donor_dir_y = df["donor_y"] - df["actual_y"]

    dot = disp_x * donor_dir_x + disp_y * donor_dir_y
    norm_disp = np.sqrt(disp_x**2 + disp_y**2) + 1e-12
    norm_donor = np.sqrt(donor_dir_x**2 + donor_dir_y**2) + 1e-12
    df["cosine_similarity"] = dot / (norm_disp * norm_donor)

    # Displacement magnitude (how far the prediction moved from actual)
    df["displacement_mag"] = norm_disp

    return df


def create_combined(df=None, save_dir=None):
    """Combined 2x2 figure."""
    set_nature_style()

    if df is None:
        df = load_data()

    fig = plt.figure(figsize=(7.08, 6.5))

    ax_a = fig.add_subplot(2, 2, 1)
    panel_a.draw(ax_a, df)
    ax_a.text(-0.12, 1.08, "a", transform=ax_a.transAxes,
              fontsize=8, fontweight="bold")

    ax_b = fig.add_subplot(2, 2, 2)
    panel_b.draw(ax_b, df)
    ax_b.text(-0.12, 1.08, "b", transform=ax_b.transAxes,
              fontsize=8, fontweight="bold")

    ax_c = fig.add_subplot(2, 2, 3)
    panel_c.draw(ax_c, df)
    ax_c.text(-0.12, 1.08, "c", transform=ax_c.transAxes,
              fontsize=8, fontweight="bold")

    ax_d = fig.add_subplot(2, 2, 4)
    panel_d.draw(ax_d, df)
    ax_d.text(-0.12, 1.08, "d", transform=ax_d.transAxes,
              fontsize=8, fontweight="bold")

    plt.tight_layout(h_pad=2.5, w_pad=2.0)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "combined.pdf"),
                    bbox_inches="tight", dpi=300)
        fig.savefig(os.path.join(save_dir, "combined.png"),
                    bbox_inches="tight", dpi=200)
        print(f"Saved: {os.path.join(save_dir, 'combined.png')}")
    return fig


def main():
    """Generate all Figure S3 outputs."""
    print("Generating Supplementary Figure S3 panels...")
    try:
        df = load_data()
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        print("  Skipping Figure S3 (requires exp18_interpretability.csv)")
        return

    # Print summary statistics
    cs = df["cosine_similarity"].values
    print(f"\n--- Cosine Similarity Summary ---")
    print(f"  Mean: {np.mean(cs):.4f}")
    print(f"  Median: {np.median(cs):.4f}")
    print(f"  Fraction > 0: {(cs > 0).sum()}/{len(cs)} ({100*(cs>0).mean():.1f}%)")
    w_stat, w_pval = stats.wilcoxon(cs, alternative='greater')
    print(f"  Wilcoxon signed-rank: W={w_stat:.0f}, p={w_pval:.2e} (one-sided)")

    # Per-sample
    print(f"\n--- Per-Sample Cosine Similarity ---")
    per_sample = df.groupby("sample_id")["cosine_similarity"].agg(["mean", "std", "count"])
    for sid, row in per_sample.iterrows():
        direction = "toward donor" if row["mean"] > 0 else "AWAY from donor"
        print(f"  {sid}: mean={row['mean']:.4f} +/- {row['std']:.4f} (n={int(row['count'])}) [{direction}]")

    # Individual panels
    fig_a = panel_a.create()
    save_figure(fig_a, 'figS3/panel_a')
    plt.close(fig_a)

    fig_b = panel_b.create()
    save_figure(fig_b, 'figS3/panel_b')
    plt.close(fig_b)

    fig_c = panel_c.create()
    save_figure(fig_c, 'figS3/panel_c')
    plt.close(fig_c)

    fig_d = panel_d.create()
    save_figure(fig_d, 'figS3/panel_d')
    plt.close(fig_d)

    # Combined
    fig_combined = create_combined(df)
    save_figure(fig_combined, 'figS3/combined')
    plt.close(fig_combined)

    print("\nFigure S3 complete.")


if __name__ == "__main__":
    main()
