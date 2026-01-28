"""
Figure S3 Panel C: Per-Sample Mean Cosine Similarity

Per-sample mean cosine similarity with seed-based error bars.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """Draw per-sample cosine similarity panel on given axes."""
    if df is None:
        df = _load_data()

    per_ss = df.groupby(["sample_id", "seed"])["cosine_similarity"].mean().reset_index()
    per_ss.columns = ["sample_id", "seed", "cs"]
    sample_stats = per_ss.groupby("sample_id")["cs"].agg(["mean", "std"])
    sample_stats = sample_stats.sort_values("mean", ascending=False)

    x_pos = range(len(sample_stats))
    colors_bar = [("#2ecc71" if m > 0 else "#e74c3c") for m in sample_stats["mean"]]

    ax.bar(x_pos, sample_stats["mean"], yerr=sample_stats["std"],
           color=colors_bar, alpha=0.8, capsize=3,
           edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.7, label="Random (0)")

    overall_mean = df["cosine_similarity"].mean()
    ax.axhline(overall_mean, color="#0072B2", ls="-", lw=1.5, alpha=0.8,
               label=f"Overall mean = {overall_mean:.3f}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(sample_stats.index, fontsize=6)
    ax.set_xlabel("HER2ST Sample", fontsize=7)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=7)
    ax.legend(fontsize=5, loc="lower right")

    n_positive = (sample_stats["mean"] > 0).sum()
    ax.text(0.98, 0.98, f"{n_positive}/{len(sample_stats)} samples\ntoward donor",
            transform=ax.transAxes, ha="right", va="top", fontsize=6,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ecc71", alpha=0.2))


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel C."""
    fig = create()
    save_figure(fig, 'figS3/panel_c')
    plt.close(fig)
    print("Figure S3 panel C complete.")


if __name__ == '__main__':
    main()
