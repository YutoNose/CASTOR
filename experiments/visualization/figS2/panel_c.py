"""
Figure S2 Panel C: K Neighbors Ablation

Number of neighbors (k) ablation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, COLORS, SINGLE_COL, RESULTS_DIR
)


def _load_ablation_data():
    """Load ablation study results from exp07."""
    for filename in ['exp07_ablation_studies.csv', 'exp07_ablation.csv']:
        path = RESULTS_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            if 'auc_ectopic_pos' in df.columns:
                df['auc_ectopic'] = df['auc_ectopic_pos']
            if 'auc_intrinsic_pos' in df.columns:
                df['auc_intrinsic'] = df['auc_intrinsic_pos']
            return df
    raise FileNotFoundError("No ablation data found")


def draw(ax, df=None):
    """Draw k_neighbors ablation panel on given axes."""
    if df is None:
        df = _load_ablation_data()

    k_df = df[df['ablation_type'] == 'k_neighbors'] if 'ablation_type' in df.columns else df

    summary = k_df.groupby('k_neighbors').agg({
        'auc_ectopic': ['mean', 'std']
    }).reset_index()
    summary.columns = ['k_neighbors', 'mean', 'std']
    summary = summary.sort_values('k_neighbors')

    ax.errorbar(summary['k_neighbors'], summary['mean'],
                yerr=summary['std'], marker='o', capsize=3,
                color=COLORS['inv_pos'], linewidth=1.5, markersize=5)

    ax.set_xlabel('Number of Neighbors (k)', fontsize=8)
    ax.set_ylabel('Ectopic Detection AUC', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)


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
    save_figure(fig, 'figS2/panel_c')
    plt.close(fig)
    print("Figure S2 panel C complete.")


if __name__ == '__main__':
    main()
