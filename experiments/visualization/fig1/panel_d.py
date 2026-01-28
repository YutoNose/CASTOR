"""
Figure 1 Panel (d): Expected Detection Pattern

Shows which methods detect which anomaly types.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, add_panel_label, COLORS, SINGLE_COL


def create_panel_d():
    """Create panel (d): Expected detection pattern."""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.7))

    # Data: methods and their detection capabilities
    methods = ['Inv_PosError\n(Ours)', 'PCA_Error', 'LOF', 'LISA']
    ectopic_detect = [True, False, False, True]  # LISA detects spatial patterns
    intrinsic_detect = [False, True, True, False]

    n_methods = len(methods)
    x = np.arange(n_methods)
    width = 0.35

    # Create grouped bar-like visualization
    for i, method in enumerate(methods):
        # Ectopic detection
        color_e = COLORS['ectopic'] if ectopic_detect[i] else '#E0E0E0'
        ax.scatter([i - 0.15], [1], c=color_e, s=200, marker='*',
                   edgecolors='black' if ectopic_detect[i] else '#BDBDBD',
                   linewidths=0.8)

        # Intrinsic detection
        color_i = COLORS['intrinsic'] if intrinsic_detect[i] else '#E0E0E0'
        ax.scatter([i + 0.15], [1], c=color_i, s=150, marker='s',
                   edgecolors='black' if intrinsic_detect[i] else '#BDBDBD',
                   linewidths=0.8)

    # Method labels
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=6)

    # Highlight our method
    ax.axvspan(-0.5, 0.5, alpha=0.1, color=COLORS['inv_pos'])

    # Legend
    legend_elements = [
        plt.scatter([], [], c=COLORS['ectopic'], s=100, marker='*',
                    edgecolors='black', label='Detects Ectopic'),
        plt.scatter([], [], c=COLORS['intrinsic'], s=80, marker='s',
                    edgecolors='black', label='Detects Intrinsic'),
        plt.scatter([], [], c='#E0E0E0', s=80, marker='o',
                    edgecolors='#BDBDBD', label='Does not detect'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=5,
              framealpha=0.9)

    ax.set_ylim(0.5, 1.5)
    ax.set_xlim(-0.6, n_methods - 0.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_title('Expected Detection Pattern', fontweight='bold', pad=10)

    plt.tight_layout()
    return fig


def main():
    fig = create_panel_d()
    save_figure(fig, 'fig1/panel_d')
    plt.close(fig)
    print("Figure 1 Panel (d) generated.")


if __name__ == '__main__':
    main()
