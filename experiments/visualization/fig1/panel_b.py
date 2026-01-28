"""
Figure 1 Panel (b): Forward vs Inverse Prediction Paradigm

Shows the conceptual difference between predicting expression from position
versus predicting position from expression.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, add_panel_label, COLORS, SINGLE_COL


def create_panel_b():
    """Create panel (b): Forward vs Inverse prediction."""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.7))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # ===== Forward prediction (top row) =====
    y_forward = 4.5

    # Position box
    ax.add_patch(FancyBboxPatch((0.5, y_forward - 0.7), 2.2, 1.4,
                                boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor='#1976D2',
                                linewidth=1.2))
    ax.text(1.6, y_forward, 'Position\n(x, y)', fontsize=7, ha='center', va='center')

    # Arrow
    ax.annotate('', xy=(4.2, y_forward), xytext=(2.9, y_forward),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.5))
    ax.text(3.55, y_forward + 0.5, 'Forward', fontsize=6, ha='center',
            color='#666666')

    # Expression box
    ax.add_patch(FancyBboxPatch((4.2, y_forward - 0.7), 2.2, 1.4,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor='#C62828',
                                linewidth=1.2))
    ax.text(5.3, y_forward, 'Expression\nĝ(x,y)', fontsize=7, ha='center', va='center')

    # Detection type
    ax.text(7.5, y_forward, '→ Detects\n   Intrinsic', fontsize=6, ha='left',
            color=COLORS['intrinsic'], fontweight='bold')

    # ===== Inverse prediction (bottom row) =====
    y_inverse = 1.8

    # Expression box
    ax.add_patch(FancyBboxPatch((0.5, y_inverse - 0.7), 2.2, 1.4,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor='#C62828',
                                linewidth=1.2))
    ax.text(1.6, y_inverse, 'Expression\ng', fontsize=7, ha='center', va='center')

    # Arrow (emphasized)
    ax.annotate('', xy=(4.2, y_inverse), xytext=(2.9, y_inverse),
                arrowprops=dict(arrowstyle='->', color=COLORS['inv_pos'], lw=2.5))
    ax.text(3.55, y_inverse + 0.5, 'Inverse', fontsize=7, ha='center',
            color=COLORS['inv_pos'], fontweight='bold')

    # Position box
    ax.add_patch(FancyBboxPatch((4.2, y_inverse - 0.7), 2.2, 1.4,
                                boxstyle="round,pad=0.1",
                                facecolor='#E3F2FD', edgecolor='#1976D2',
                                linewidth=1.2))
    ax.text(5.3, y_inverse, 'Position\nf(g)', fontsize=7, ha='center', va='center')

    # Detection type (emphasized)
    ax.text(7.5, y_inverse, '→ Detects\n   Ectopic', fontsize=6, ha='left',
            color=COLORS['ectopic'], fontweight='bold')

    # Highlight box around inverse
    ax.add_patch(FancyBboxPatch((0.2, y_inverse - 1.0), 6.6, 2.0,
                                boxstyle="round,pad=0.1",
                                facecolor='none', edgecolor=COLORS['inv_pos'],
                                linewidth=1.5, linestyle='--', alpha=0.5))

    ax.set_title('Forward vs Inverse Prediction', fontweight='bold', pad=10)
    ax.axis('off')

    plt.tight_layout()
    return fig


def main():
    fig = create_panel_b()
    save_figure(fig, 'fig1/panel_b')
    plt.close(fig)
    print("Figure 1 Panel (b) generated.")


if __name__ == '__main__':
    main()
