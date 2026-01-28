"""
Figure 1 Panel (c): Method Workflow

Shows the pipeline: ST Data → GNN Encoder → Position Decoder → Anomaly Score
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, add_panel_label, COLORS, SINGLE_COL


def create_panel_c():
    """Create panel (c): Method workflow."""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.5))

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)

    y_center = 2.0
    box_h = 1.2

    # Step 1: ST Data (Input)
    ax.add_patch(FancyBboxPatch((0.3, y_center - box_h/2), 2.0, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#F5F5F5', edgecolor='#424242',
                                linewidth=1))
    ax.text(1.3, y_center, 'ST Data\n(X, coords)', fontsize=6, ha='center', va='center')

    # Arrow 1
    ax.annotate('', xy=(3.0, y_center), xytext=(2.5, y_center),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))

    # Step 2: GNN Encoder
    ax.add_patch(FancyBboxPatch((3.0, y_center - box_h/2), 2.2, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#E3F2FD', edgecolor=COLORS['inv_pos'],
                                linewidth=1.5))
    ax.text(4.1, y_center, 'GNN\nEncoder', fontsize=6, ha='center', va='center',
            fontweight='bold')

    # Arrow 2
    ax.annotate('', xy=(6.0, y_center), xytext=(5.4, y_center),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))

    # Step 3: Position Decoder
    ax.add_patch(FancyBboxPatch((6.0, y_center - box_h/2), 2.2, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#E8F5E9', edgecolor='#2E7D32',
                                linewidth=1.5))
    ax.text(7.1, y_center, 'Position\nDecoder', fontsize=6, ha='center', va='center',
            fontweight='bold')

    # Arrow 3
    ax.annotate('', xy=(9.0, y_center), xytext=(8.4, y_center),
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))

    # Step 4: Anomaly Score (Output)
    ax.add_patch(FancyBboxPatch((9.0, y_center - box_h/2), 3.0, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#FFEBEE', edgecolor=COLORS['ectopic'],
                                linewidth=1.5))
    ax.text(10.5, y_center + 0.15, 'Position Error', fontsize=6, ha='center', va='center',
            fontweight='bold')
    ax.text(10.5, y_center - 0.25, '||pred - actual||', fontsize=5, ha='center', va='center',
            family='monospace')

    # Key insight box at bottom
    ax.text(7, 0.4, 'Key: Ectopic cells have expression that maps to donor position',
            fontsize=5, ha='center', style='italic', color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1',
                      edgecolor='#FFB300', alpha=0.8))

    ax.set_title('Inverse Position Prediction Pipeline', fontweight='bold', pad=8)
    ax.axis('off')

    plt.tight_layout()
    return fig


def main():
    fig = create_panel_c()
    save_figure(fig, 'fig1/panel_c')
    plt.close(fig)
    print("Figure 1 Panel (c) generated.")


if __name__ == '__main__':
    main()
