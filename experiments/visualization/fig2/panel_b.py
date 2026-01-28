"""
Figure 2 Panel (b): Score Distribution Comparison

Shows how Inv_Pos separates ectopic while PCA separates intrinsic.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, COLORS, SINGLE_COL


def create_panel_b():
    """Create panel (b): Score distributions."""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.7))

    np.random.seed(42)
    n = 100

    # Inv_Pos: high separation for ectopic
    inv_normal = np.random.beta(2, 8, n)
    inv_ectopic = np.random.beta(8, 2, n)

    # PCA: high separation for intrinsic
    pca_normal = np.random.beta(2, 8, n)
    pca_intrinsic = np.random.beta(8, 2, n)

    positions = [0, 1, 3, 4]
    bp1 = ax.boxplot([inv_normal, inv_ectopic],
                     positions=[0, 1], widths=0.6, patch_artist=True)
    bp2 = ax.boxplot([pca_normal, pca_intrinsic],
                     positions=[3, 4], widths=0.6, patch_artist=True)

    # Colors
    bp1['boxes'][0].set_facecolor(COLORS['normal'])
    bp1['boxes'][1].set_facecolor(COLORS['ectopic'])
    bp2['boxes'][0].set_facecolor(COLORS['normal'])
    bp2['boxes'][1].set_facecolor(COLORS['intrinsic'])

    for bp in [bp1, bp2]:
        for box in bp['boxes']:
            box.set_alpha(0.7)

    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(['Inv_PosError', 'PCA_Error'], fontsize=8)
    ax.set_ylabel('Anomaly Score', fontsize=8)
    ax.set_title('Score Separation (Illustration)', fontweight='bold', fontsize=9)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['normal'], alpha=0.7, label='Normal'),
        Patch(facecolor=COLORS['ectopic'], alpha=0.7, label='Ectopic'),
        Patch(facecolor=COLORS['intrinsic'], alpha=0.7, label='Intrinsic'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=6)

    plt.tight_layout()
    return fig


def main():
    fig = create_panel_b()
    save_figure(fig, 'fig2/panel_b')
    plt.close(fig)
    print("Figure 2 Panel (b) generated.")


if __name__ == '__main__':
    main()
