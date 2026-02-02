"""
Figure S3 Panel B: Cosine Similarity Distribution

Cosine similarity distribution with one-sample t-test vs 0.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, COLORS, SINGLE_COL, RESULTS_DIR


def _load_data():
    """Load interpretability data from exp18."""
    csv_path = RESULTS_DIR / "exp18_interpretability.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run exp18_interpretability.py first: {csv_path}")
    df = pd.read_csv(csv_path)

    disp_x = df["pred_x"] - df["actual_x"]
    disp_y = df["pred_y"] - df["actual_y"]
    donor_dir_x = df["donor_x"] - df["actual_x"]
    donor_dir_y = df["donor_y"] - df["actual_y"]

    dot = disp_x * donor_dir_x + disp_y * donor_dir_y
    norm_disp = np.sqrt(disp_x**2 + disp_y**2) + 1e-12
    norm_donor = np.sqrt(donor_dir_x**2 + donor_dir_y**2) + 1e-12
    df["cosine_similarity"] = dot / (norm_disp * norm_donor)
    df["displacement_mag"] = norm_disp

    return df


def draw(ax, df=None):
    """Draw cosine similarity distribution panel on given axes."""
    if df is None:
        df = _load_data()

    cs = df["cosine_similarity"].values

    ax.hist(cs, bins=40, color=COLORS.get("inv_pos", "#0072B2"),
            alpha=0.7, edgecolor="white", linewidth=0.3, density=True)

    mean_cs = np.mean(cs)
    ax.axvline(mean_cs, color="#E74C3C", lw=2, ls="-",
               label=f"Mean = {mean_cs:.3f}")
    ax.axvline(0, color="gray", lw=1.5, ls="--", alpha=0.7,
               label="Random (0)")

    t_stat, pval = stats.ttest_1samp(cs, 0, alternative='greater')

    # Use scipy's built-in one-sided test (tests if distribution > 0)
    w_stat, w_pval = stats.wilcoxon(cs, alternative='greater')

    ax.set_xlabel("Cosine Similarity\n(displacement vs donor direction)", fontsize=7)
    ax.set_ylabel("Density", fontsize=7)

    n_positive = (cs > 0).sum()
    frac_positive = n_positive / len(cs)
    stats_text = (f"n = {len(cs)}\n"
                  f"{frac_positive:.1%} toward donor\n"
                  f"t-test p = {pval:.2e}\n"
                  f"Wilcoxon p = {w_pval:.2e}")
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            ha="right", va="top", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    ax.legend(fontsize=6, loc="upper left")


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel B."""
    fig = create()
    save_figure(fig, 'figS3/panel_b')
    plt.close(fig)
    print("Figure S3 panel B complete.")


if __name__ == '__main__':
    main()
