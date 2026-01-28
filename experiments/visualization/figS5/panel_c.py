"""
Figure S5 Panel C: AUC Stability

AUC stability across dataset sizes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, COLORS, SINGLE_COL, RESULTS_DIR


def _load_scalability_data():
    """Load scalability analysis results from exp13."""
    for filename in ['exp13_scalability_analysis.csv', 'exp13_scalability.csv']:
        path = RESULTS_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            df = df[df['status'] == 'success'] if 'status' in df.columns else df
            return df
    raise FileNotFoundError("No scalability data found")


def draw(ax, df=None):
    """Draw AUC stability panel on given axes."""
    if df is None:
        df = _load_scalability_data()

    auc_summary = df.groupby('n_spots')['auc_ectopic_Inv_PosError'].agg(['mean', 'std']).reset_index()
    auc_summary = auc_summary.sort_values('n_spots')

    ax.errorbar(auc_summary['n_spots'], auc_summary['mean'],
                yerr=auc_summary['std'], fmt='o-',
                color=COLORS['inv_pos'], capsize=3,
                label='Inv_PosError (Ectopic)', linewidth=1.5, markersize=4)

    if 'auc_intrinsic_Inv_PosError' in df.columns:
        int_summary = df.groupby('n_spots')['auc_intrinsic_Inv_PosError'].agg(['mean', 'std']).reset_index()
        int_summary = int_summary.sort_values('n_spots')
        ax.errorbar(int_summary['n_spots'], int_summary['mean'],
                    yerr=int_summary['std'], fmt='s--',
                    color=COLORS['intrinsic'], capsize=3,
                    label='Inv_PosError (Intrinsic)', linewidth=1, markersize=3)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Spots', fontsize=8)
    ax.set_ylabel('Detection AUC', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=5, loc='lower right')


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel C."""
    fig = create()
    save_figure(fig, 'figS5/panel_c')
    plt.close(fig)
    print("Figure S5 panel C complete.")


if __name__ == '__main__':
    main()
