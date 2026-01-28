"""
Figure S6 Panel A: Spatial Distribution

Spatial distribution of synthetic cells from ZINB data generation pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common import set_nature_style, save_figure, COLORS, SINGLE_COL


def _load_synthetic_layout():
    """Load spatial layout from actual data generation pipeline."""
    from core.data_generation import generate_synthetic_data
    # Use fewer spots for readability, but match DEFAULT_CONFIG proportions
    # (ectopic 3.3%, intrinsic 10%)
    X, coords, labels, ectopic_idx, intrinsic_idx = generate_synthetic_data(
        n_spots=500, n_genes=500, n_ectopic=17, n_intrinsic=50,
        random_state=42,
    )
    # Normalize coords to [0, 1]
    # Note: np.ptp() is deprecated in NumPy 2.0, use max-min instead
    coord_range = coords.max(axis=0) - coords.min(axis=0)
    coords_norm = (coords - coords.min(axis=0)) / (coord_range + 1e-8)
    return coords_norm, labels, ectopic_idx, intrinsic_idx


def draw(ax, df=None):
    """Draw spatial distribution panel on given axes."""
    coords, labels, ectopic_idx, intrinsic_idx = _load_synthetic_layout()

    normal_mask = labels == 0
    ectopic_mask = labels == 1
    intrinsic_mask = labels == 2

    n_cells = len(labels)
    n_ectopic = int(ectopic_mask.sum())
    n_intrinsic = int(intrinsic_mask.sum())

    ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1],
               c=COLORS['normal'], s=20, alpha=0.5, label='Normal')
    ax.scatter(coords[ectopic_mask, 0], coords[ectopic_mask, 1],
               c=COLORS['ectopic'], s=50, marker='*',
               edgecolors='black', linewidths=0.5, label='Ectopic')
    ax.scatter(coords[intrinsic_mask, 0], coords[intrinsic_mask, 1],
               c=COLORS['intrinsic'], s=40, marker='s',
               edgecolors='black', linewidths=0.5, label='Intrinsic')

    ax.set_xlabel('X coordinate', fontsize=8)
    ax.set_ylabel('Y coordinate', fontsize=8)
    ax.legend(fontsize=6, loc='upper right')
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    pct_ect = 100 * n_ectopic / n_cells
    pct_int = 100 * n_intrinsic / n_cells
    stats_text = f'n = {n_cells}\nEctopic: {n_ectopic} ({pct_ect:.0f}%)\nIntrinsic: {n_intrinsic} ({pct_int:.0f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=6, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


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
    save_figure(fig, 'figS6/panel_a')
    plt.close(fig)
    print("Figure S6 panel A complete.")


if __name__ == '__main__':
    main()
