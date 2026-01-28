"""
Figure S3 Panel A: Spatial Displacement Arrows

Arrows showing predicted displacement vs donor direction for the
sample with highest mean cosine similarity.
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
    """Draw spatial arrows panel on given axes."""
    if df is None:
        df = _load_data()

    sample_means = df.groupby("sample_id")["cosine_similarity"].mean()
    sample = sample_means.idxmax()
    seed = df[df["sample_id"] == sample]["seed"].iloc[0]
    subset = df[(df["sample_id"] == sample) & (df["seed"] == seed)].copy()

    subset = subset.sort_values("cosine_similarity")

    for _, row in subset.iterrows():
        cs = row["cosine_similarity"]
        if cs > 0.5:
            color = "#2ecc71"
            alpha = 0.8
        elif cs > 0:
            color = "#82e0aa"
            alpha = 0.6
        elif cs > -0.5:
            color = "#f1948a"
            alpha = 0.6
        else:
            color = "#e74c3c"
            alpha = 0.8

        ax.annotate(
            "", xy=(row["pred_x"], row["pred_y"]),
            xytext=(row["actual_x"], row["actual_y"]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                            alpha=alpha, mutation_scale=10),
        )

    ax.scatter(subset["donor_x"], subset["donor_y"],
               c="#E74C3C", marker="D", s=25, alpha=0.5,
               edgecolors="black", linewidths=0.3, label="Donor (tumor)", zorder=5)

    ax.scatter(subset["actual_x"], subset["actual_y"],
               c="#3498DB", marker="o", s=30, alpha=0.7,
               edgecolors="black", linewidths=0.3, label="Actual (recipient)", zorder=5)

    mean_cs = subset["cosine_similarity"].mean()
    n_pos = (subset["cosine_similarity"] > 0).sum()
    n_total = len(subset)
    ax.text(0.02, 0.98, f"Sample {sample}\nMean cos sim = {mean_cs:.2f}\n{n_pos}/{n_total} toward donor",
            transform=ax.transAxes, fontsize=6, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.legend(fontsize=5, loc="best", framealpha=0.9)
    ax.set_xlabel("Normalized x", fontsize=7)
    ax.set_ylabel("Normalized y", fontsize=7)
    ax.set_aspect("equal")


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 1.0))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel A."""
    fig = create()
    save_figure(fig, 'figS3/panel_a')
    plt.close(fig)
    print("Figure S3 panel A complete.")


if __name__ == '__main__':
    main()
