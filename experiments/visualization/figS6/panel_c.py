"""
Figure S6 Panel C: Anomaly Injection Visualization

Ectopic and intrinsic anomaly injection diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, COLORS, SINGLE_COL


def draw(ax_or_axes, df=None):
    """Draw anomaly injection visualization on given axes.

    Can accept either a single axes (for combined figure)
    or a tuple of two axes (for standalone figure).
    """
    np.random.seed(42)

    if hasattr(ax_or_axes, '__len__') and len(ax_or_axes) == 2:
        ax_left, ax_right = ax_or_axes
    else:
        # For combined figure, split into left/right using inset
        ax_left = ax_or_axes
        ax_right = None

    n = 20
    x = np.random.rand(n)
    y = np.random.rand(n)

    # Left: Ectopic injection
    region_a = y > 0.5
    region_b = y <= 0.5

    ax_left.scatter(x[region_a], y[region_a], c='#A8D5BA', s=40, alpha=0.6, label='Region A')
    ax_left.scatter(x[region_b], y[region_b], c='#B8D4E8', s=40, alpha=0.6, label='Region B')

    ectopic_actual = [0.3, 0.7]
    ectopic_donor = [0.3, 0.3]
    ax_left.scatter([ectopic_actual[0]], [ectopic_actual[1]], c=COLORS['ectopic'],
               s=100, marker='*', edgecolors='black', linewidths=0.8, zorder=10)
    ax_left.scatter([ectopic_donor[0]], [ectopic_donor[1]], c=COLORS['ectopic'],
               s=60, marker='*', alpha=0.3, zorder=5)
    ax_left.annotate('', xy=ectopic_actual, xytext=ectopic_donor,
                arrowprops=dict(arrowstyle='->', color=COLORS['ectopic'], lw=1.5))
    ax_left.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)

    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.set_aspect('equal')
    ax_left.text(0.5, 0.05, 'Expression copied\nfrom donor', fontsize=6, ha='center',
            transform=ax_left.transAxes)
    ax_left.axis('off')

    if ax_right is not None:
        # Right: Intrinsic injection
        ax_right.scatter(x, y, c=COLORS['normal'], s=40, alpha=0.6)

        intrinsic_pos = [0.5, 0.5]
        ax_right.scatter([intrinsic_pos[0]], [intrinsic_pos[1]], c=COLORS['intrinsic'],
                   s=100, marker='s', edgecolors='black', linewidths=0.8, zorder=10)

        for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
            dx = 0.08 * np.cos(angle)
            dy = 0.08 * np.sin(angle)
            ax_right.annotate('', xy=(intrinsic_pos[0] + dx, intrinsic_pos[1] + dy),
                        xytext=intrinsic_pos,
                        arrowprops=dict(arrowstyle='->', color=COLORS['intrinsic'],
                                        lw=0.8, alpha=0.6))

        ax_right.set_xlim(0, 1)
        ax_right.set_ylim(0, 1)
        ax_right.set_aspect('equal')
        ax_right.text(0.5, 0.05, 'Expression\nperturbed', fontsize=6, ha='center',
                transform=ax_right.transAxes)
        ax_right.axis('off')


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 1.6, SINGLE_COL * 0.8))
    draw(axes)
    plt.tight_layout()
    return fig


def main():
    """Generate panel C."""
    fig = create()
    save_figure(fig, 'figS6/panel_c')
    plt.close(fig)
    print("Figure S6 panel C complete.")


if __name__ == '__main__':
    main()
