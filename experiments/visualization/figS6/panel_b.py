"""
Figure S6 Panel B: Gene Expression Statistics

Expression distribution (ZINB counts) and mean-variance relationship.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common import set_nature_style, save_figure, COLORS, SINGLE_COL


def _generate_zinb_data():
    """Generate ZINB count data from the actual data generation pipeline."""
    from core.data_generation import generate_raw_counts
    X_counts, X_norm, coords, labels, _, _ = generate_raw_counts(
        n_spots=3000, n_genes=500, n_ectopic=100, n_intrinsic=300,
        random_state=42,
    )
    return X_counts, X_norm


def draw(ax_or_axes, df=None):
    """Draw gene expression statistics panel on given axes.

    Can accept either a single axes (for combined figure with inset)
    or a tuple of two axes (for standalone figure).
    """
    X_counts, X_norm = _generate_zinb_data()

    if hasattr(ax_or_axes, '__len__') and len(ax_or_axes) == 2:
        ax_left, ax_right = ax_or_axes
    else:
        ax_left = ax_or_axes
        ax_right = ax_left.inset_axes([0.55, 0.15, 0.4, 0.7])

    # Left: log1p expression distribution
    ax_left.hist(X_norm.flatten(), bins=50, density=True, alpha=0.7,
                 color=COLORS['inv_pos'], edgecolor='black', linewidth=0.3)
    ax_left.set_xlabel('Log1p Expression', fontsize=7)
    ax_left.set_ylabel('Density', fontsize=7)

    # Right: mean-variance relationship (on raw counts — characteristic of ZINB)
    gene_means = X_counts.mean(axis=0)
    gene_vars = X_counts.var(axis=0)
    ax_right.scatter(gene_means, gene_vars, alpha=0.3, s=5, c=COLORS['inv_pos'])
    ax_right.set_xlabel('Gene Mean', fontsize=6)
    ax_right.set_ylabel('Gene Variance', fontsize=6)

    # Poisson reference line (var = mean)
    x_line = np.linspace(gene_means.min(), gene_means.max(), 100)
    ax_right.plot(x_line, x_line, 'k--', linewidth=0.8, alpha=0.5, label='Poisson')
    # Quadratic fit (NB: var = mean + mean^2/r)
    z = np.polyfit(gene_means, gene_vars, 2)
    p = np.poly1d(z)
    ax_right.plot(x_line, p(x_line), 'r-', linewidth=1, alpha=0.7, label='NB fit')
    ax_right.legend(fontsize=5, loc='upper left')


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 1.6, SINGLE_COL * 0.8))
    draw(axes)
    plt.tight_layout()
    return fig


def main():
    """Generate panel B."""
    fig = create()
    save_figure(fig, 'figS6/panel_b')
    plt.close(fig)
    print("Figure S6 panel B complete.")


if __name__ == '__main__':
    main()
