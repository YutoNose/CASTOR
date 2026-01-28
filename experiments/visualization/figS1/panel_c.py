"""
Figure S1 Panel C: Summary Comparison at High Noise

Comparison at 50% dropout.
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

# Method display config
METHOD_CONFIG = {
    'Inv_PosError': {'color': COLORS['inv_pos'], 'label': 'Inv_PosError'},
    'PCA_Error': {'color': COLORS['pca_error'], 'label': 'PCA_Error'},
}


def _load_noise_data():
    """Load noise robustness data from exp04."""
    return pd.read_csv(RESULTS_DIR / 'exp04_noise_robustness.csv')


def draw(ax, df=None):
    """Draw summary comparison panel on given axes."""
    if df is None:
        df = _load_noise_data()

    high_dropout = df[
        (df['perturbation_type'] == 'dropout') &
        (df['perturbation_level'] == 0.5)
    ]

    if len(high_dropout) > 0:
        method_aucs = []
        for method, config in METHOD_CONFIG.items():
            col = f'{method}_auc_ectopic'
            if col in high_dropout.columns:
                mean_auc = high_dropout[col].mean()
                std_auc = high_dropout[col].std()
                method_aucs.append({
                    'method': config['label'],
                    'method_key': method,
                    'auc': mean_auc,
                    'std': std_auc
                })

        auc_df = pd.DataFrame(method_aucs)
        auc_df = auc_df.sort_values('auc', ascending=True)

        colors = [METHOD_CONFIG[m]['color'] for m in auc_df['method_key']]
        ax.barh(range(len(auc_df)), auc_df['auc'],
                xerr=auc_df['std'], capsize=3,
                color=colors, edgecolor='black', linewidth=0.3)

        ax.set_yticks(range(len(auc_df)))
        ax.set_yticklabels(auc_df['method'], fontsize=6)

        for i, (_, row) in enumerate(auc_df.iterrows()):
            ax.text(row['auc'] + row['std'] + 0.02, i, f"{row['auc']:.2f}",
                    va='center', fontsize=5)
    else:
        ax.text(0.5, 0.5, 'No dropout=0.5 data', ha='center', va='center',
                transform=ax.transAxes)

    ax.set_xlabel('Ectopic Detection AUC', fontsize=7)
    ax.set_xlim(0, 1.1)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)


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
    save_figure(fig, 'figS1/panel_c')
    plt.close(fig)
    print("Figure S1 panel C complete.")


if __name__ == '__main__':
    main()
