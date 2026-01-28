"""
Figure 1 Panel (a): Two Types of Spatial Anomalies

Illustrates ectopic (spatially misplaced) vs intrinsic (expression outlier) anomalies.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, add_panel_label, COLORS, SINGLE_COL


def create_panel_a():
    """Create panel (a): Two types of anomalies."""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.8))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    # ===== Left side: Ectopic anomaly =====
    np.random.seed(42)

    # Region A cells (top-left)
    x_a = np.random.normal(2.5, 0.6, 12)
    y_a = np.random.normal(5, 0.5, 12)
    ax.scatter(x_a, y_a, c='#B8D4E8', s=50, alpha=0.7, zorder=2,
               edgecolors='white', linewidths=0.3)

    # Region B cells (bottom-left)
    x_b = np.random.normal(2.5, 0.6, 12)
    y_b = np.random.normal(2, 0.5, 12)
    ax.scatter(x_b, y_b, c='#A8D5BA', s=50, alpha=0.7, zorder=2,
               edgecolors='white', linewidths=0.3)

    # Ectopic cell: from B placed in A
    ectopic_x, ectopic_y = 3.0, 4.8
    donor_x, donor_y = 2.5, 2.2
    ax.scatter([ectopic_x], [ectopic_y], c=COLORS['ectopic'], s=120, marker='*',
               edgecolors='black', linewidths=1, zorder=10)

    # Arrow from donor to ectopic position
    ax.annotate('', xy=(ectopic_x, ectopic_y), xytext=(donor_x, donor_y),
                arrowprops=dict(arrowstyle='->', color=COLORS['ectopic'],
                                lw=2, ls='--', connectionstyle='arc3,rad=0.2'))

    # Region labels
    ax.text(0.8, 5.8, 'Region A', fontsize=7, fontweight='bold', color='#1565C0')
    ax.text(0.8, 2.8, 'Region B', fontsize=7, fontweight='bold', color='#2E7D32')

    # Dividing line
    ax.axhline(3.5, xmin=0.02, xmax=0.42, color='gray', linestyle=':', linewidth=1)

    # Ectopic label
    ax.text(2.5, 0.5, 'Ectopic Anomaly', fontsize=8, fontweight='bold',
            ha='center', color='#333333')
    ax.text(2.5, 0.1, '(spatially misplaced)', fontsize=6, ha='center',
            color='#666666', style='italic')

    # ===== Right side: Intrinsic anomaly =====
    # Normal cells
    x_c = np.random.normal(9, 1.0, 25)
    y_c = np.random.normal(3.5, 1.2, 25)
    ax.scatter(x_c, y_c, c=COLORS['normal'], s=50, alpha=0.6, zorder=2,
               edgecolors='white', linewidths=0.3)

    # Intrinsic anomaly: expression outlier at correct position
    intrinsic_x, intrinsic_y = 9.2, 3.8
    ax.scatter([intrinsic_x], [intrinsic_y], c=COLORS['intrinsic'], s=100, marker='s',
               edgecolors='black', linewidths=1, zorder=10)

    # Expression burst indicator
    burst_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in burst_angles:
        dx, dy = 0.4 * np.cos(angle), 0.4 * np.sin(angle)
        ax.plot([intrinsic_x, intrinsic_x + dx], [intrinsic_y, intrinsic_y + dy],
                color=COLORS['intrinsic'], lw=1.5, alpha=0.6)

    # Intrinsic label
    ax.text(9, 0.5, 'Intrinsic Anomaly', fontsize=8, fontweight='bold',
            ha='center', color='#333333')
    ax.text(9, 0.1, '(expression outlier)', fontsize=6, ha='center',
            color='#666666', style='italic')

    # Vertical divider
    ax.axvline(5.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_title('Two Types of Spatial Anomalies', fontweight='bold', pad=10)
    ax.axis('off')

    plt.tight_layout()
    return fig


def main():
    fig = create_panel_a()
    save_figure(fig, 'fig1/panel_a')
    plt.close(fig)
    print("Figure 1 Panel (a) generated.")


if __name__ == '__main__':
    main()
