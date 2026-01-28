"""
Supplementary Figure S6: Synthetic Data Statistics

Caption: Characteristics of the synthetic spatial transcriptomics
data used for benchmarking.

Panels:
(a) Spatial distribution of cells
(b) Gene expression statistics
(c) Anomaly injection visualization
(d) Parameter summary table
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label, load_results,
    COLORS, METHOD_NAMES, SINGLE_COL, DOUBLE_COL, FIGURE_DIR, RESULTS_DIR
)

from figS6 import panel_a, panel_b, panel_c, panel_d


def create_combined():
    """Create Supplementary Figure S6: Synthetic data statistics."""
    set_nature_style()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.7))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.45)

    # Panel (a): Spatial distribution
    ax_a = fig.add_subplot(gs[0, 0])
    panel_a.draw(ax_a)
    add_panel_label(ax_a, 'a', x=-0.15)

    # Panel (b): Expression statistics
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b_inset = ax_b.inset_axes([0.55, 0.15, 0.4, 0.7])
    panel_b.draw((ax_b, ax_b_inset))
    add_panel_label(ax_b, 'b', x=-0.1)

    # Panel (c): Anomaly injection
    ax_c = fig.add_subplot(gs[1, 0])
    # For the combined figure, we draw ectopic only on this axes
    # and use a simplified version
    _draw_injection_combined(ax_c)
    add_panel_label(ax_c, 'c', x=-0.15)

    # Panel (d): Parameter table
    ax_d = fig.add_subplot(gs[1, 1])
    _draw_table_combined(ax_d)
    add_panel_label(ax_d, 'd', x=-0.1)

    plt.tight_layout()
    return fig


def _draw_injection_combined(ax):
    """Draw combined injection panel (simplified for 2x2 layout)."""
    np.random.seed(42)
    n = 30
    x = np.random.rand(n)
    y = np.random.rand(n)

    region_a = y > 0.5
    region_b = y <= 0.5

    ax.scatter(x[region_a], y[region_a], c='#A8D5BA', s=25, alpha=0.5)
    ax.scatter(x[region_b], y[region_b], c='#B8D4E8', s=25, alpha=0.5)

    # Ectopic
    ectopic_actual = [0.25, 0.75]
    ectopic_donor = [0.25, 0.25]
    ax.scatter([ectopic_actual[0]], [ectopic_actual[1]], c=COLORS['ectopic'],
               s=80, marker='*', edgecolors='black', linewidths=0.8, zorder=10)
    ax.scatter([ectopic_donor[0]], [ectopic_donor[1]], c=COLORS['ectopic'],
               s=40, marker='*', alpha=0.3, zorder=5)
    ax.annotate('', xy=ectopic_actual, xytext=ectopic_donor,
                arrowprops=dict(arrowstyle='->', color=COLORS['ectopic'], lw=1.5))

    # Intrinsic
    intrinsic_pos = [0.75, 0.5]
    ax.scatter([intrinsic_pos[0]], [intrinsic_pos[1]], c=COLORS['intrinsic'],
               s=60, marker='s', edgecolors='black', linewidths=0.8, zorder=10)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
    ax.text(0.05, 0.9, 'Region A', fontsize=6, color='#2E7D32')
    ax.text(0.05, 0.1, 'Region B', fontsize=6, color='#1565C0')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')


def _draw_table_combined(ax):
    """Draw compact parameter table for combined figure."""
    params = [
        ['Spots', '3,000'],
        ['Genes', '500'],
        ['Ectopic', '100 (3.3%)'],
        ['Intrinsic', '300 (10%)'],
        ['Donor dist.', '> 0.5'],
        ['Seeds', '30'],
    ]

    ax.axis('off')

    table = ax.table(
        cellText=params,
        colLabels=['Parameter', 'Value'],
        cellLoc='left',
        loc='center',
        colWidths=[0.6, 0.4]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.1, 1.3)

    for i in range(2):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(fontweight='bold')


# Alias for backward compatibility
create_figS6 = create_combined


def main():
    """Generate all Figure S6 outputs."""
    print("Generating Figure S6 panels...")

    fig_a = panel_a.create()
    save_figure(fig_a, 'figS6/panel_a')
    plt.close(fig_a)

    fig_b = panel_b.create()
    save_figure(fig_b, 'figS6/panel_b')
    plt.close(fig_b)

    fig_c = panel_c.create()
    save_figure(fig_c, 'figS6/panel_c')
    plt.close(fig_c)

    fig_d = panel_d.create()
    save_figure(fig_d, 'figS6/panel_d')
    plt.close(fig_d)

    fig_combined = create_combined()
    save_figure(fig_combined, 'figS6/combined')
    plt.close(fig_combined)

    print("Figure S6 complete.")


if __name__ == '__main__':
    main()
