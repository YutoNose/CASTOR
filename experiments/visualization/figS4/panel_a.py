"""
Figure S4 Panel A: Method Comparison Heatmap

Full method comparison heatmap (Ectopic + Intrinsic AUC).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL, RESULTS_DIR

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
    """Draw method comparison heatmap on given axes."""
    if df is None:
        df = _load_per_seed_data()

    summary = df.groupby('score')[['auc_ectopic', 'auc_intrinsic']].mean()
    summary.columns = ['Ectopic AUC', 'Intrinsic AUC']
    summary['Selectivity'] = summary['Ectopic AUC'] - summary['Intrinsic AUC']

    summary.index = [DISPLAY_NAMES.get(m, m) for m in summary.index]
    summary = summary.sort_values('Selectivity', ascending=False)

    # Note: Selectivity column can be negative, so we don't constrain vmin/vmax
    # AUC columns are [0,1], Selectivity is [-1,1]
    sns.heatmap(summary, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0.5, ax=ax, cbar_kws={'shrink': 0.6},
                annot_kws={'size': 6})

    ax.tick_params(axis='both', labelsize=6)


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.2))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel A."""
    fig = create()
    save_figure(fig, 'figS4/panel_a')
    plt.close(fig)
    print("Figure S4 panel A complete.")


if __name__ == '__main__':
    main()
