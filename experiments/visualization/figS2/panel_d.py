"""
Figure S2 Panel D: Selectivity Summary

Selectivity (Ectopic - Intrinsic AUC) summary across ablation conditions.
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
    """Draw selectivity summary panel on given axes."""
    if df is None:
        df = _load_ablation_data()

    df = df.copy()
    df['selectivity'] = df['auc_ectopic'] - df['auc_intrinsic']

    ablation_configs = []

    # Lambda ablation
    lambda_df = df[df['ablation_type'] == 'lambda_pos'] if 'ablation_type' in df.columns else df
    for val in sorted(lambda_df['lambda_pos'].unique()):
        sub = lambda_df[lambda_df['lambda_pos'] == val]
        ablation_configs.append({
            'label': f'$\\lambda$={val}',
            'mean': sub['selectivity'].mean(),
            'std': sub['selectivity'].std(),
        })

    names = [c['label'] for c in ablation_configs]
    means = [c['mean'] for c in ablation_configs]
    stds = [c['std'] for c in ablation_configs]

    colors = [COLORS['inv_pos'] if m > 0 else COLORS['pca_error'] for m in means]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=stds, capsize=3,
            color=colors, edgecolor='black', linewidth=0.3, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('Selectivity (Ectopic - Intrinsic AUC)', fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.02, i, f'{m:.2f}', va='center', fontsize=5)


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel D."""
    fig = create()
    save_figure(fig, 'figS2/panel_d')
    plt.close(fig)
    print("Figure S2 panel D complete.")


if __name__ == '__main__':
    main()
