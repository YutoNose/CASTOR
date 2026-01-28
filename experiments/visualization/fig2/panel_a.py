"""
Figure 2 Panel (a): Cross-Detection AUC Heatmap

Shows which methods detect which anomaly types.
Data source: exp01_cross-detection_auc.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, METHOD_NAMES, SINGLE_COL, FIGURE_DIR, RESULTS_DIR
)


def create_panel_a():
    """Create panel (a): Cross-detection AUC heatmap from real exp01 data."""
    set_nature_style()

    df = pd.read_csv(RESULTS_DIR / 'exp01_cross-detection_auc.csv')
    summary = df.groupby('score')[['auc_ectopic', 'auc_intrinsic']].mean()
    summary.columns = ['Ectopic', 'Intrinsic']

    # Rename for display
    summary.index = [METHOD_NAMES.get(m, m) if m in METHOD_NAMES
                     else m for m in summary.index]

    # Sort by ectopic AUC
    summary = summary.sort_values('Ectopic', ascending=False)

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 0.9, SINGLE_COL * 1.1))

    sns.heatmap(summary, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax,
                cbar_kws={'shrink': 0.7, 'label': 'AUC'},
                annot_kws={'size': 7})

    ax.set_title('Cross-Detection AUC', fontweight='bold', fontsize=9)
    ax.set_xlabel('Anomaly Type', fontsize=8)
    ax.set_ylabel('')
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    return fig


def main():
    fig = create_panel_a()
    save_figure(fig, 'fig2/panel_a')
    plt.close(fig)
    print("Figure 2 Panel (a) generated.")


if __name__ == '__main__':
    main()
