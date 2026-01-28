"""
Figure 1: Concept and Method Overview

Caption: Inverse spatial prediction enables selective detection
of ectopic anomalies in spatial transcriptomics.

Panels:
(a) Two types of anomalies: Ectopic vs Intrinsic
(b) Forward vs Inverse prediction paradigm
(c) Method workflow schematic

Outputs:
- figures/fig1/panel_a.pdf/png
- figures/fig1/panel_b.pdf/png
- figures/fig1/panel_c.pdf/png
- figures/fig1/combined.pdf/png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, SINGLE_COL, DOUBLE_COL
)


# =============================================================================
# Individual Panel Functions
# =============================================================================

def create_panel_a():
    """Panel (a): Two types of anomalies."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

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

    # Ectopic cell
    ax.scatter([3.0], [4.8], c=COLORS['ectopic'], s=120, marker='*',
               edgecolors='black', linewidths=0.8, zorder=10)
    ax.annotate('', xy=(3.0, 4.8), xytext=(2.5, 2.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['ectopic'],
                                lw=1.8, ls='--', connectionstyle='arc3,rad=0.2'))

    ax.text(0.8, 5.6, 'Region A', fontsize=7, fontweight='bold', color='#1565C0')
    ax.text(0.8, 2.6, 'Region B', fontsize=7, fontweight='bold', color='#2E7D32')
    ax.axhline(3.5, xmin=0.02, xmax=0.42, color='gray', linestyle=':', linewidth=0.8)
    ax.text(2.5, 0.5, 'Ectopic\n(spatially misplaced)', fontsize=7, fontweight='bold',
            ha='center', color='#333')

    # Right side: Intrinsic
    x_c = np.random.normal(9, 1.0, 20)
    y_c = np.random.normal(3.5, 1.0, 20)
    ax.scatter(x_c, y_c, c=COLORS['normal'], s=50, alpha=0.6, zorder=2,
               edgecolors='white', linewidths=0.3)
    ax.scatter([9.2], [3.8], c=COLORS['intrinsic'], s=100, marker='s',
               edgecolors='black', linewidths=0.8, zorder=10)

    # Expression burst
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        dx, dy = 0.4 * np.cos(angle), 0.4 * np.sin(angle)
        ax.plot([9.2, 9.2 + dx], [3.8, 3.8 + dy],
                color=COLORS['intrinsic'], lw=1.5, alpha=0.6)

    ax.text(9, 0.5, 'Intrinsic\n(expression outlier)', fontsize=7, fontweight='bold',
            ha='center', color='#333')
    ax.axvline(5.5, color='gray', linestyle='-', linewidth=0.4, alpha=0.3)

    ax.axis('off')
    plt.tight_layout()
    return fig


def create_panel_b():
    """Panel (b): Forward vs Inverse prediction - FIXED ALIGNMENT."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 0.85))

    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)

    # Box dimensions
    box_w = 2.0
    box_h = 1.2

    # Forward (top row)
    y_f = 4.3
    left_box_x = 2.0
    right_box_x = 6.0
    arrow_start = left_box_x + box_w + 0.15
    arrow_end = right_box_x - 0.15
    arrow_mid = (arrow_start + arrow_end) / 2

    # Left box: Position
    ax.add_patch(FancyBboxPatch((left_box_x, y_f - box_h/2), box_w, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#E3F2FD', edgecolor='#1976D2', lw=1.2))
    ax.text(left_box_x + box_w/2, y_f, 'Position\n(x, y)', fontsize=7, ha='center', va='center')

    # Arrow centered between boxes
    ax.annotate('', xy=(arrow_end, y_f), xytext=(arrow_start, y_f),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))
    ax.text(arrow_mid, y_f + 0.5, 'Forward', fontsize=6, ha='center', va='bottom', color='#666')

    # Right box: Expression
    ax.add_patch(FancyBboxPatch((right_box_x, y_f - box_h/2), box_w, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#FFEBEE', edgecolor='#C62828', lw=1.2))
    ax.text(right_box_x + box_w/2, y_f, 'Expression\nĝ(x,y)', fontsize=7, ha='center', va='center')

    # Result text
    ax.text(right_box_x + box_w + 0.5, y_f, '  Detects\n  Intrinsic', fontsize=7,
            color=COLORS['intrinsic'], fontweight='bold', va='center')

    # Inverse (bottom row, emphasized)
    y_i = 1.6

    # Left box: Expression
    ax.add_patch(FancyBboxPatch((left_box_x, y_i - box_h/2), box_w, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#FFEBEE', edgecolor='#C62828', lw=1.2))
    ax.text(left_box_x + box_w/2, y_i, 'Expression\ng', fontsize=7, ha='center', va='center')

    # Arrow centered between boxes (emphasized)
    ax.annotate('', xy=(arrow_end, y_i), xytext=(arrow_start, y_i),
                arrowprops=dict(arrowstyle='->', color=COLORS['inv_pos'], lw=2.5))
    ax.text(arrow_mid, y_i + 0.55, 'Inverse', fontsize=7, ha='center', va='bottom',
            color=COLORS['inv_pos'], fontweight='bold')

    # Right box: Position
    ax.add_patch(FancyBboxPatch((right_box_x, y_i - box_h/2), box_w, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#E3F2FD', edgecolor='#1976D2', lw=1.2))
    ax.text(right_box_x + box_w/2, y_i, 'Position\nf(g)', fontsize=7, ha='center', va='center')

    # Result text
    ax.text(right_box_x + box_w + 0.5, y_i, '  Detects\n  Ectopic', fontsize=7,
            color=COLORS['ectopic'], fontweight='bold', va='center')

    # Highlight box around inverse row
    ax.add_patch(FancyBboxPatch((left_box_x - 0.3, y_i - box_h/2 - 0.25),
                                right_box_x + box_w - left_box_x + 0.6, box_h + 0.5,
                                boxstyle="round,pad=0.05",
                                facecolor='none', edgecolor=COLORS['inv_pos'],
                                lw=1.5, ls='--', alpha=0.5))

    ax.axis('off')
    plt.tight_layout()
    return fig


def create_panel_c():
    """Panel (c): Method workflow - FIXED ALIGNMENT."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.9, SINGLE_COL * 0.5))

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)

    y = 2.0
    bh = 1.1

    # Box positions and widths
    boxes = [
        (2.0, 2.0, 'ST Data\n(X, coords)', '#F5F5F5', '#424242'),
        (5.0, 2.2, 'GNN\nEncoder', '#E3F2FD', COLORS['inv_pos']),
        (8.3, 2.2, 'Position\nDecoder', '#E8F5E9', '#2E7D32'),
        (11.7, 3.0, 'Anomaly Score\n||pred - actual||', '#FFEBEE', COLORS['ectopic']),
    ]

    for x, w, text, fc, ec in boxes:
        ax.add_patch(FancyBboxPatch((x, y - bh/2), w, bh,
                                    boxstyle="round,pad=0.06",
                                    facecolor=fc, edgecolor=ec, lw=1.3))
        ax.text(x + w/2, y, text, fontsize=7, ha='center', va='center')

    # Arrows centered between boxes
    arrow_pairs = [
        (2.0 + 2.0 + 0.1, 5.0 - 0.1),   # Data -> GNN
        (5.0 + 2.2 + 0.1, 8.3 - 0.1),   # GNN -> Decoder
        (8.3 + 2.2 + 0.1, 11.7 - 0.1),  # Decoder -> Score
    ]
    for x1, x2 in arrow_pairs:
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.2))

    # Key insight
    ax.text(9.5, 0.4, 'Key: Ectopic cells have expression that maps to donor location',
            fontsize=6, ha='center', style='italic', color='#555',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF8E1', ec='#FFB300', alpha=0.9))

    ax.axis('off')
    plt.tight_layout()
    return fig


# =============================================================================
# Combined Figure
# =============================================================================

def create_combined():
    """Create combined Figure 1 with all panels."""
    set_nature_style()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.7],
                          hspace=0.6, wspace=0.45,
                          left=0.06, right=0.97, top=0.93, bottom=0.08)

    # Panel (a)
    ax_a = fig.add_subplot(gs[0, 0])
    _draw_panel_a(ax_a)
    add_panel_label(ax_a, 'a', x=-0.1, y=1.08)

    # Panel (b)
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_panel_b(ax_b)
    add_panel_label(ax_b, 'b', x=-0.08, y=1.08)

    # Panel (c)
    ax_c = fig.add_subplot(gs[1, :])
    _draw_panel_c(ax_c)
    add_panel_label(ax_c, 'c', x=-0.04, y=1.15)

    return fig


def _draw_panel_a(ax):
    """Draw panel (a) content on given axes."""
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    np.random.seed(42)

    x_a = np.random.normal(2.5, 0.6, 12)
    y_a = np.random.normal(5, 0.5, 12)
    ax.scatter(x_a, y_a, c='#B8D4E8', s=40, alpha=0.7, edgecolors='white', linewidths=0.3)

    x_b = np.random.normal(2.5, 0.6, 12)
    y_b = np.random.normal(2, 0.5, 12)
    ax.scatter(x_b, y_b, c='#A8D5BA', s=40, alpha=0.7, edgecolors='white', linewidths=0.3)

    ax.scatter([3.0], [4.8], c=COLORS['ectopic'], s=100, marker='*',
               edgecolors='black', linewidths=0.8, zorder=10)
    ax.annotate('', xy=(3.0, 4.8), xytext=(2.5, 2.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['ectopic'],
                                lw=1.5, ls='--', connectionstyle='arc3,rad=0.2'))

    ax.text(0.8, 5.6, 'Region A', fontsize=6, fontweight='bold', color='#1565C0')
    ax.text(0.8, 2.6, 'Region B', fontsize=6, fontweight='bold', color='#2E7D32')
    ax.axhline(3.5, xmin=0.02, xmax=0.42, color='gray', linestyle=':', linewidth=0.8)
    ax.text(2.5, 0.5, 'Ectopic', fontsize=7, fontweight='bold', ha='center')

    x_c = np.random.normal(9, 1.0, 20)
    y_c = np.random.normal(3.5, 1.0, 20)
    ax.scatter(x_c, y_c, c=COLORS['normal'], s=40, alpha=0.6, edgecolors='white', linewidths=0.3)
    ax.scatter([9.2], [3.8], c=COLORS['intrinsic'], s=80, marker='s',
               edgecolors='black', linewidths=0.8, zorder=10)

    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        dx, dy = 0.35 * np.cos(angle), 0.35 * np.sin(angle)
        ax.plot([9.2, 9.2 + dx], [3.8, 3.8 + dy], color=COLORS['intrinsic'], lw=1.2, alpha=0.5)

    ax.text(9, 0.5, 'Intrinsic', fontsize=7, fontweight='bold', ha='center')
    ax.axvline(5.5, color='gray', linestyle='-', linewidth=0.4, alpha=0.3)
    ax.axis('off')


def _draw_panel_b(ax):
    """Draw panel (b) content on given axes - FIXED ALIGNMENT."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    # Box dimensions
    box_w = 1.8
    box_h = 1.1

    # Forward (top row)
    y_f = 4.2
    left_box_x = 1.9
    right_box_x = 5.5
    arrow_start = left_box_x + box_w + 0.1
    arrow_end = right_box_x - 0.1
    arrow_mid = (arrow_start + arrow_end) / 2

    # Left box: Position
    ax.add_patch(FancyBboxPatch((left_box_x, y_f - box_h/2), box_w, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#E3F2FD', edgecolor='#1976D2', lw=1))
    ax.text(left_box_x + box_w/2, y_f, 'Position\n(x,y)', fontsize=6, ha='center', va='center')

    # Arrow centered
    ax.annotate('', xy=(arrow_end, y_f), xytext=(arrow_start, y_f),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))
    ax.text(arrow_mid, y_f + 0.45, 'Forward', fontsize=5, ha='center', va='bottom', color='#666')

    # Right box: Expression
    ax.add_patch(FancyBboxPatch((right_box_x, y_f - box_h/2), box_w, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#FFEBEE', edgecolor='#C62828', lw=1))
    ax.text(right_box_x + box_w/2, y_f, 'Expression\nĝ(x,y)', fontsize=6, ha='center', va='center')

    # Result
    ax.text(right_box_x + box_w + 0.3, y_f, ' Intrinsic', fontsize=6,
            color=COLORS['intrinsic'], fontweight='bold', va='center')

    # Inverse (bottom row)
    y_i = 1.7

    # Left box: Expression
    ax.add_patch(FancyBboxPatch((left_box_x, y_i - box_h/2), box_w, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#FFEBEE', edgecolor='#C62828', lw=1))
    ax.text(left_box_x + box_w/2, y_i, 'Expression\ng', fontsize=6, ha='center', va='center')

    # Arrow centered (emphasized)
    ax.annotate('', xy=(arrow_end, y_i), xytext=(arrow_start, y_i),
                arrowprops=dict(arrowstyle='->', color=COLORS['inv_pos'], lw=2))
    ax.text(arrow_mid, y_i + 0.5, 'Inverse', fontsize=6, ha='center', va='bottom',
            color=COLORS['inv_pos'], fontweight='bold')

    # Right box: Position
    ax.add_patch(FancyBboxPatch((right_box_x, y_i - box_h/2), box_w, box_h,
                                boxstyle="round,pad=0.08",
                                facecolor='#E3F2FD', edgecolor='#1976D2', lw=1))
    ax.text(right_box_x + box_w/2, y_i, 'Position\nf(g)', fontsize=6, ha='center', va='center')

    # Result
    ax.text(right_box_x + box_w + 0.3, y_i, ' Ectopic', fontsize=6,
            color=COLORS['ectopic'], fontweight='bold', va='center')

    # Highlight box
    ax.add_patch(FancyBboxPatch((left_box_x - 0.2, y_i - box_h/2 - 0.2),
                                right_box_x + box_w - left_box_x + 0.4, box_h + 0.4,
                                boxstyle="round,pad=0.05",
                                facecolor='none', edgecolor=COLORS['inv_pos'],
                                lw=1.2, ls='--', alpha=0.4))

    ax.axis('off')


def _draw_panel_c(ax):
    """Draw panel (c) content on given axes - FIXED ALIGNMENT."""
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 3.5)

    y = 1.8
    bh = 1.0

    # Boxes with their x positions and widths
    boxes = [
        (2.0, 2.0, 'ST Data\n(X, coords)', '#F5F5F5', '#424242'),
        (4.8, 2.2, 'GNN\nEncoder', '#E3F2FD', COLORS['inv_pos']),
        (7.9, 2.2, 'Position\nDecoder', '#E8F5E9', '#2E7D32'),
        (11.1, 3.0, 'Anomaly Score\n||pred - actual||', '#FFEBEE', COLORS['ectopic']),
    ]

    for x, w, text, fc, ec in boxes:
        ax.add_patch(FancyBboxPatch((x, y - bh/2), w, bh,
                                    boxstyle="round,pad=0.06",
                                    facecolor=fc, edgecolor=ec, lw=1.2))
        ax.text(x + w/2, y, text, fontsize=6, ha='center', va='center')

    # Arrows between boxes (centered)
    arrow_data = [
        (2.0 + 2.0, 4.8),      # Data -> GNN
        (4.8 + 2.2, 7.9),      # GNN -> Decoder
        (7.9 + 2.2, 11.1),     # Decoder -> Score
    ]
    for x1, x2 in arrow_data:
        mid = (x1 + x2) / 2
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1))

    ax.text(9.0, 0.4, 'Ectopic cells: predicted position points to donor location',
            fontsize=5, ha='center', style='italic', color='#555',
            bbox=dict(boxstyle='round,pad=0.3', fc='#FFF8E1', ec='#FFB300', alpha=0.9))

    ax.axis('off')


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all Figure 1 outputs."""
    print("Generating Figure 1 panels...")

    fig_a = create_panel_a()
    save_figure(fig_a, 'fig1/panel_a')
    plt.close(fig_a)

    fig_b = create_panel_b()
    save_figure(fig_b, 'fig1/panel_b')
    plt.close(fig_b)

    fig_c = create_panel_c()
    save_figure(fig_c, 'fig1/panel_c')
    plt.close(fig_c)

    fig_combined = create_combined()
    save_figure(fig_combined, 'fig1/combined')
    plt.close(fig_combined)

    print("Figure 1 complete.")


if __name__ == '__main__':
    main()
