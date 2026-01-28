"""
Figure 2 Panel (c): Scatter Plot of Orthogonal Detection

Shows ectopic (high Inv_Pos, low PCA) vs intrinsic (low Inv_Pos, high PCA).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, COLORS, SINGLE_COL


def create_panel_c():
    """Create panel (c): Orthogonal detection scatter."""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.85))

    np.random.seed(42)
    n = 200
    n_ectopic = 25
    n_intrinsic = 25

    # Normal cells: low on both
    normal_inv = np.random.beta(2, 8, n - n_ectopic - n_intrinsic)
    normal_pca = np.random.beta(2, 8, n - n_ectopic - n_intrinsic)

    # Ectopic: high Inv_Pos, low PCA
    ectopic_inv = np.random.beta(8, 2, n_ectopic)
    ectopic_pca = np.random.beta(2, 6, n_ectopic)

    # Intrinsic: low Inv_Pos, high PCA
    intrinsic_inv = np.random.beta(2, 6, n_intrinsic)
    intrinsic_pca = np.random.beta(8, 2, n_intrinsic)

    ax.scatter(normal_inv, normal_pca, c=COLORS['normal'],
               alpha=0.4, s=20, label='Normal')
    ax.scatter(ectopic_inv, ectopic_pca, c=COLORS['ectopic'],
               alpha=0.8, s=40, marker='*', label='Ectopic')
    ax.scatter(intrinsic_inv, intrinsic_pca, c=COLORS['intrinsic'],
               alpha=0.8, s=30, marker='s', label='Intrinsic')

    ax.set_xlabel('Inv_PosError Score', fontsize=8)
    ax.set_ylabel('PCA_Error Score', fontsize=8)
    ax.set_title('Orthogonal Detection (Illustration)', fontweight='bold', fontsize=9)
    ax.legend(loc='center', fontsize=6, framealpha=0.9)

    # Quadrant lines
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # Quadrant labels
    ax.text(0.75, 0.25, 'Ectopic\nonly', fontsize=6, ha='center', va='center',
            color=COLORS['ectopic'], alpha=0.7)
    ax.text(0.25, 0.75, 'Intrinsic\nonly', fontsize=6, ha='center', va='center',
            color=COLORS['intrinsic'], alpha=0.7)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


def main():
    fig = create_panel_c()
    save_figure(fig, 'fig2/panel_c')
    plt.close(fig)
    print("Figure 2 Panel (c) generated.")


if __name__ == '__main__':
    main()
