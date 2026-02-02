"""
Figure S4 Panel B: Pairwise Statistical Significance

Pairwise t-test significance matrix (Bonferroni-corrected).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL, RESULTS_DIR

DISPLAY_NAMES = {
    'Inv_PosError': 'Inv_Pos',
    'Inv_SelfRecon': 'SelfRecon',
    'Inv_NeighborRecon': 'NeighRecon',
    'PCA_Error': 'PCA',
    'Neighbor_Diff': 'NbrDiff',
    'LISA': 'LISA',
    'LOF': 'LOF',
    'IF': 'IF',
}


def _load_per_seed_data():
    """Load per-seed AUC data from exp01."""
    return pd.read_csv(RESULTS_DIR / 'exp01_cross-detection_auc.csv')


def _pivot_by_method(df):
    """Pivot data to get per-method columns of ectopic AUC values by seed."""
    return df.pivot(index='seed', columns='score', values='auc_ectopic')


def draw(ax, df=None):
    """Draw pairwise significance matrix on given axes."""
    if df is None:
        df = _load_per_seed_data()

    pivot = _pivot_by_method(df)
    methods = list(pivot.columns)
    n = len(methods)

    pval_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vals_i = pivot[methods[i]].dropna()
            vals_j = pivot[methods[j]].dropna()
            common = vals_i.index.intersection(vals_j.index)
            if len(common) > 2:
                _, p = stats.ttest_rel(vals_i.loc[common], vals_j.loc[common])
            else:
                p = np.nan  # Insufficient paired observations
            pval_matrix[i, j] = p
            pval_matrix[j, i] = p

    n_tests = n * (n - 1) / 2
    pval_corrected = np.minimum(pval_matrix * n_tests, 1.0)

    # Always use Bonferroni correction (no fallback to avoid HARKing)
    pvals = pval_corrected
    correction_label = "Bonferroni"

    sig_matrix = np.where(pvals < 0.001, 3,
                          np.where(pvals < 0.01, 2,
                                   np.where(pvals < 0.05, 1, 0)))
    np.fill_diagonal(sig_matrix, -1)

    cmap = plt.cm.colors.ListedColormap(
        ['#E8E8E8', 'white', '#FFF9C4', '#FFCC80', '#EF5350'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    display_names = [DISPLAY_NAMES.get(m, m)[:8] for m in methods]

    ax.imshow(sig_matrix, cmap=cmap, norm=norm)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=5)
    ax.set_yticklabels(display_names, fontsize=5)
    ax.set_title(f'Pairwise t-test ({correction_label})', fontsize=6, pad=4)

    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='n.s.'),
        Patch(facecolor='#FFF9C4', label='p<0.05'),
        Patch(facecolor='#FFCC80', label='p<0.01'),
        Patch(facecolor='#EF5350', label='p<0.001'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=5,
              bbox_to_anchor=(1.35, 1))


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.2))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel B."""
    fig = create()
    save_figure(fig, 'figS4/panel_b')
    plt.close(fig)
    print("Figure S4 panel B complete.")


if __name__ == '__main__':
    main()
