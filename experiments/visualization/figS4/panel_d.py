"""
Figure S4 Panel D: Confidence Intervals

95% confidence intervals for each method's ectopic AUC.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
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


def draw(ax, df=None):
    """Draw confidence intervals panel on given axes."""
    if df is None:
        df = _load_per_seed_data()

    summary = df.groupby('score')['auc_ectopic'].agg(['mean', 'std', 'count'])
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci'] = stats.t.ppf(0.975, summary['count'] - 1) * summary['sem']
    summary['ci_low'] = summary['mean'] - summary['ci']
    summary['ci_high'] = summary['mean'] + summary['ci']

    summary = summary.sort_values('mean', ascending=True)

    display_names = [DISPLAY_NAMES.get(m, m) for m in summary.index]

    y_pos = range(len(summary))

    ax.errorbar(summary['mean'], y_pos,
                xerr=[[m - l for m, l in zip(summary['mean'], summary['ci_low'])],
                      [h - m for m, h in zip(summary['mean'], summary['ci_high'])]],
                fmt='o', capsize=4, markersize=5, linewidth=1.5,
                color=COLORS['inv_pos'])

    for i, (idx, row) in enumerate(summary.iterrows()):
        color = COLORS['inv_pos'] if 'Inv_Pos' in DISPLAY_NAMES.get(idx, idx) else COLORS['normal']
        ax.plot(row['mean'], i, 'o', color=color, markersize=5, zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=6)
    ax.set_xlabel('Ectopic Detection AUC', fontsize=7)
    ax.set_xlim(0, 1.1)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)

    for i, (idx, row) in enumerate(summary.iterrows()):
        ax.text(1.02, i, f'{row["mean"]:.2f}\n[{row["ci_low"]:.2f}-{row["ci_high"]:.2f}]',
                va='center', fontsize=5)


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 0.9))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel D."""
    fig = create()
    save_figure(fig, 'figS4/panel_d')
    plt.close(fig)
    print("Figure S4 panel D complete.")


if __name__ == '__main__':
    main()
