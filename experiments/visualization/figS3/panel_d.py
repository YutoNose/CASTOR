"""
Figure S3 Panel D: Cosine Similarity vs Transplant Distance

Cosine similarity vs transplant distance with binned median trend.
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
    """Draw cosine similarity vs distance panel on given axes."""
    if df is None:
        df = _load_data()

    x = df["dist_donor_to_actual"].values
    y = df["cosine_similarity"].values

    pos_mask = y > 0
    ax.scatter(x[pos_mask], y[pos_mask], c="#2ecc71", alpha=0.15, s=8,
               label="Toward donor", rasterized=True)
    ax.scatter(x[~pos_mask], y[~pos_mask], c="#e74c3c", alpha=0.15, s=8,
               label="Away from donor", rasterized=True)

    ax.axhline(0, color="gray", ls="--", lw=1.5, alpha=0.7)

    bins = np.linspace(x.min(), x.max(), 10)
    bin_centers, bin_medians, bin_cis = [], [], []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        if mask.sum() > 10:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            vals = y[mask]
            bin_medians.append(np.median(vals))
            boot = np.array([np.median(np.random.choice(vals, len(vals)))
                             for _ in range(200)])
            bin_cis.append((np.percentile(boot, 2.5), np.percentile(boot, 97.5)))

    if bin_centers:
        bin_centers = np.array(bin_centers)
        bin_medians = np.array(bin_medians)
        ci_low = np.array([c[0] for c in bin_cis])
        ci_high = np.array([c[1] for c in bin_cis])
        ax.plot(bin_centers, bin_medians, "k-o", lw=2, ms=4,
                zorder=10, label="Binned median")
        ax.fill_between(bin_centers, ci_low, ci_high,
                        color="black", alpha=0.1, zorder=9)

    rho, p_rho = stats.spearmanr(x, y)
    ax.text(0.03, 0.03, f"Spearman r = {rho:.3f}\np = {p_rho:.2e}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    ax.set_xlabel("Transplant Distance\n(donor-actual distance)", fontsize=7)
    ax.set_ylabel("Cosine Similarity\n(>0 = toward donor)", fontsize=7)
    ax.legend(fontsize=5, loc="upper right")


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel D."""
    fig = create()
    save_figure(fig, 'figS3/panel_d')
    plt.close(fig)
    print("Figure S3 panel D complete.")


if __name__ == '__main__':
    main()
