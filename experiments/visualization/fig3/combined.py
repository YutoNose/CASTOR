"""
Figure 3: Robustness to Hard Ectopic Scenarios (Combined)

Caption: Position prediction error is the only method robust to
spatially proximal ectopic anomalies.

Panels:
(a) Scenario illustration: Easy vs Hard ectopic
(b) Method performance across scenarios
(c) Hard ectopic detail

Data source: exp10_multi-scenario_validation.csv (unsupervised training)
Columns: scenario, scenario_full_name, seed,
         auc_ectopic_pos, auc_ectopic_pca, auc_ectopic_neighbor,
         auc_intrinsic_pos, auc_intrinsic_pca
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, SINGLE_COL, DOUBLE_COL, RESULTS_DIR
)

# Method mapping for exp10 columns
SCENARIO_METHODS = {
    'auc_ectopic_pos': ('Inv_PosError', COLORS['inv_pos']),
    'auc_ectopic_pca': ('PCA_Error', COLORS['pca_error']),
    'auc_ectopic_neighbor': ('Neighbor_Recon', COLORS['neighbor_diff']),
}

# Scenario display order and labels
SCENARIO_ORDER = [
    'baseline', 'noisy_ectopic', 'partial_ectopic',
    'hard_ectopic', 'hardest'
]
SCENARIO_LABELS = {
    'baseline': 'Baseline',
    'noisy_ectopic': 'Noisy\nCopy',
    'partial_ectopic': 'Partial Mix\n($\\alpha$=0.7)',
    'hard_ectopic': 'Partial Mix\n($\\alpha$=0.5)',
    'hardest': 'Partial Mix\n+ Noise + Subtle',
    'realistic_counts': 'Realistic\nCounts',
    'medium_intrinsic': 'Medium\nIntrinsic',
    'hard_intrinsic': 'Hard\nIntrinsic',
    'cell_type_based': 'Cell Type\nBased',
}


def _load_scenario_data():
    """Load multi-scenario validation results from exp10."""
    for filename in ['exp10_multi-scenario_validation.csv', 'exp10_multi_scenario.csv']:
        path = RESULTS_DIR / filename
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No multi-scenario data found")


# =============================================================================
# Individual Panel Functions
# =============================================================================

def create_panel_a():
    """Panel (a): Easy vs Hard ectopic illustration - standalone figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 1.0))
    _draw_scenario_panel(ax)
    plt.tight_layout()
    return fig


def create_panel_b():
    """Panel (b): Method comparison across scenarios - standalone figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.0))
    df = _load_scenario_data()
    _draw_bar_chart_panel(ax, df)
    plt.tight_layout()
    return fig


def create_panel_c():
    """Panel (c): Hard ectopic detail - standalone figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    df = _load_scenario_data()
    _draw_hard_ectopic_panel(ax, df)
    plt.tight_layout()
    return fig


# =============================================================================
# Panel Drawing Functions
# =============================================================================

def _draw_scenario_panel(ax):
    """Draw panel (a) content: expression mixing schematic for scenarios."""
    ax.set_xlim(0, 3.3)
    ax.set_ylim(0, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Gene profile bar heights for donor and original
    donor_profile = [0.9, 0.3, 0.7, 0.5]
    original_profile = [0.2, 0.8, 0.4, 0.6]
    noise_offsets = [0.1, -0.15, 0.12, -0.08]

    bar_w = 0.06
    n_genes = len(donor_profile)

    def _draw_mini_bars(x0, y0, heights, color, max_h=0.55):
        """Draw a mini bar chart at (x0, y0)."""
        for j, h in enumerate(heights):
            bx = x0 + j * (bar_w + 0.02)
            bh = max(0.02, min(h, 1.0)) * max_h
            ax.add_patch(plt.Rectangle((bx, y0), bar_w, bh,
                                       facecolor=color, edgecolor='black',
                                       linewidth=0.4))

    section_xs = [0.15, 1.15, 2.15]
    labels = ['Baseline', 'Partial Mix\n($\\alpha$=0.5)', 'Hardest']
    sublabels = ['100% donor', '50% donor +\n50% original', 'mixed +\nnoise + subtle']
    bar_colors = [COLORS['ectopic'], '#b07cd8', '#d45f5f']

    for idx, (sx, lbl, slbl, clr) in enumerate(zip(
            section_xs, labels, sublabels, bar_colors)):
        # Compute profile
        if idx == 0:
            profile = donor_profile
        elif idx == 1:
            profile = [0.5 * d + 0.5 * o
                       for d, o in zip(donor_profile, original_profile)]
        else:
            profile = [np.clip(0.5 * d + 0.5 * o + n, 0, 1)
                       for d, o, n in zip(donor_profile, original_profile,
                                          noise_offsets)]

        # Draw bars
        _draw_mini_bars(sx, 0.35, profile, clr)

        # Arrow pointing down to a recipient spot
        arrow_cx = sx + (n_genes - 1) * (bar_w + 0.02) / 2 + bar_w / 2
        ax.annotate('', xy=(arrow_cx, 0.18),
                    xytext=(arrow_cx, 0.33),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.0))

        # Recipient spot
        ax.scatter([arrow_cx], [0.1], s=50, c=clr,
                   edgecolors='black', linewidths=0.6, zorder=5)

        # Labels
        ax.text(arrow_cx, 1.05, lbl, fontsize=6, fontweight='bold',
                ha='center', va='bottom')
        ax.text(arrow_cx, 0.92, slbl, fontsize=5, ha='center',
                va='top', color='#444444')

    # Difficulty arrow across the bottom
    ax.annotate('', xy=(2.9, -0.05), xytext=(0.4, -0.05),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.0))
    ax.text(1.65, -0.12, 'Increasing detection difficulty', fontsize=5,
            ha='center', va='top', color='gray', fontstyle='italic')


def _draw_bar_chart_panel(ax, df):
    """Draw panel (b) content: grouped bar chart across scenarios."""
    # Filter to scenarios that exist in the data
    available = [s for s in SCENARIO_ORDER if s in df['scenario'].unique()]
    if not available:
        available = sorted(df['scenario'].unique())

    # Collect method data
    method_cols = [c for c in SCENARIO_METHODS if c in df.columns]

    x_pos = np.arange(len(available))
    width = 0.8 / len(method_cols)

    for i, col in enumerate(method_cols):
        name, color = SCENARIO_METHODS[col]
        means = []
        stds = []
        for scenario in available:
            sub = df[df['scenario'] == scenario]
            means.append(sub[col].mean())
            stds.append(sub[col].std())

        offsets = x_pos + (i - len(method_cols)/2 + 0.5) * width
        ax.bar(offsets, means, width, yerr=stds, capsize=2,
               label=name, color=color,
               edgecolor='black', linewidth=0.3)

    ax.set_xticks(x_pos)
    labels = [SCENARIO_LABELS.get(s, s) for s in available]
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel('Ectopic Detection AUC', fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper right', fontsize=5)

    # Highlight hard ectopic scenarios
    for scenario_name in ['hard_ectopic', 'hardest']:
        if scenario_name in available:
            idx = available.index(scenario_name)
            ax.axvspan(idx - 0.45, idx + 0.45, alpha=0.08, color='red')


def _draw_hard_ectopic_panel(ax, df):
    """Draw panel (c) content: hard ectopic scenario detail."""
    hard_scenarios = ['hard_ectopic', 'hardest', 'partial_ectopic']
    hard_df = df[df['scenario'].isin(hard_scenarios)]

    if len(hard_df) == 0:
        ax.text(0.5, 0.5, 'No hard ectopic data', ha='center', va='center',
                transform=ax.transAxes)
        return

    method_cols = [c for c in SCENARIO_METHODS if c in df.columns]

    method_data = []
    for col in method_cols:
        name, color = SCENARIO_METHODS[col]
        vals = hard_df[col].dropna()
        if len(vals) > 0:
            method_data.append({
                'name': name, 'color': color,
                'mean': vals.mean(), 'std': vals.std(),
            })

    # Sort by mean AUC
    method_data.sort(key=lambda x: x['mean'])

    names = [m['name'] for m in method_data]
    means = [m['mean'] for m in method_data]
    stds = [m['std'] for m in method_data]
    colors = [m['color'] for m in method_data]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=stds, capsize=3,
            color=colors, edgecolor='black', linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('Ectopic Detection AUC', fontsize=7)
    ax.set_xlim(0, 1.1)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(min(m + s + 0.03, 1.05), i, f'{m:.2f}', va='center', fontsize=5)


# =============================================================================
# Combined Figure
# =============================================================================

def create_combined():
    """Create combined Figure 3 with all panels."""
    set_nature_style()

    df = _load_scenario_data()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.7, 1], width_ratios=[1, 1.4],
                          hspace=0.55, wspace=0.45,
                          left=0.08, right=0.95, top=0.92, bottom=0.1)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_scenario_panel(ax_a)
    add_panel_label(ax_a, 'a', x=-0.12, y=1.1)

    ax_b = fig.add_subplot(gs[:, 1])
    _draw_bar_chart_panel(ax_b, df)
    add_panel_label(ax_b, 'b', x=-0.08, y=1.02)

    ax_c = fig.add_subplot(gs[1, 0])
    _draw_hard_ectopic_panel(ax_c, df)
    add_panel_label(ax_c, 'c', x=-0.2, y=1.08)

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all Figure 3 outputs."""
    print("Generating Figure 3 panels...")

    fig_a = create_panel_a()
    save_figure(fig_a, 'fig3/panel_a')
    plt.close(fig_a)

    fig_b = create_panel_b()
    save_figure(fig_b, 'fig3/panel_b')
    plt.close(fig_b)

    fig_c = create_panel_c()
    save_figure(fig_c, 'fig3/panel_c')
    plt.close(fig_c)

    fig_combined = create_combined()
    save_figure(fig_combined, 'fig3/combined')
    plt.close(fig_combined)

    print("Figure 3 complete.")


if __name__ == '__main__':
    main()
