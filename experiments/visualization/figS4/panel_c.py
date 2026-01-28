"""
Figure S4 Panel C: Effect Sizes

Cohen's d comparing Inv_PosError ectopic AUC to other methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, COLORS, SINGLE_COL, RESULTS_DIR

DISPLAY_NAMES = {
    'Inv_PosError': 'Inv_Pos',
    'Inv_SelfRecon': 'SelfRecon',
    'Inv_NeighborRecon': 'NeighRecon',
    'PCA_Error': 'PCA',
    'Neighbor_Diff': 'NbrDiff',
    'LISA': 'LISA',
    'LOF': 'LOF',
    'IF': 'IF',
}


def _load_per_seed_data():
    """Load per-seed AUC data from exp01."""
    return pd.read_csv(RESULTS_DIR / 'exp01_cross-detection_auc.csv')


def _pivot_by_method(df):
    """Pivot data to get per-method columns of ectopic AUC values by seed."""
    return df.pivot(index='seed', columns='score', values='auc_ectopic')


def draw(ax, df=None):
    """Draw effect sizes panel on given axes."""
    if df is None:
        df = _load_per_seed_data()

    pivot = _pivot_by_method(df)

    if 'Inv_PosError' not in pivot.columns:
        ax.text(0.5, 0.5, 'No Inv_PosError data', ha='center', va='center',
                transform=ax.transAxes)
        return

    inv_pos = pivot['Inv_PosError'].dropna()

    methods = []
    effect_sizes = []
    for col in pivot.columns:
        if col == 'Inv_PosError':
            continue
        other = pivot[col].dropna()
        common = inv_pos.index.intersection(other.index)
        if len(common) < 3:
            continue

        # Paired Cohen's d (d_z): appropriate for paired comparisons
        diff = inv_pos.loc[common] - other.loc[common]
        d = diff.mean() / diff.std(ddof=1) if diff.std() > 0 else 0

        methods.append(DISPLAY_NAMES.get(col, col))
        effect_sizes.append(d)

    sorted_pairs = sorted(zip(effect_sizes, methods), reverse=True)
    effect_sizes = [p[0] for p in sorted_pairs]
    methods = [p[1] for p in sorted_pairs]

    colors = [COLORS['inv_pos'] if d > 0.8 else
              COLORS['ectopic'] if d > 0.5 else
              COLORS['normal'] for d in effect_sizes]

    y_pos = range(len(methods))
    ax.barh(y_pos, effect_sizes, color=colors, edgecolor='black', linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=6)
    ax.set_xlabel("Cohen's d (Inv_Pos - Method)", fontsize=7)
    ax.axvline(0.8, color='gray', linestyle='--', linewidth=0.5, alpha=0.7,
               label='Large effect')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7,
               label='Medium effect')

    for i, d in enumerate(effect_sizes):
        ax.text(d + 0.1, i, f'{d:.1f}', va='center', fontsize=5)

    ax.legend(fontsize=4, loc='lower right')


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel C."""
    fig = create()
    save_figure(fig, 'figS4/panel_c')
    plt.close(fig)
    print("Figure S4 panel C complete.")


if __name__ == '__main__':
    main()
