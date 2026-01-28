"""
Figure S1 Panel B: Expression Noise Robustness

Performance vs expression noise level.
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


def _get_perturbation_data(df, perturbation_type):
    """Get data for a specific perturbation type, including baseline (none)."""
    baseline = df[df['perturbation_type'] == 'none'].copy()
    baseline['perturbation_level'] = 0.0
    perturbed = df[df['perturbation_type'] == perturbation_type]
    return pd.concat([baseline, perturbed])


def draw(ax, df=None):
    """Draw expression noise robustness panel on given axes."""
    if df is None:
        df = _load_noise_data()

    noise_data = _get_perturbation_data(df, 'expression_noise')

    for method, config in METHOD_CONFIG.items():
        col = f'{method}_auc_ectopic'
        if col in noise_data.columns:
            summary = noise_data.groupby('perturbation_level')[col].agg(['mean', 'std'])
            ax.errorbar(summary.index, summary['mean'], yerr=summary['std'],
                        marker='s', capsize=3, linewidth=1.5, markersize=4,
                        color=config['color'], label=config['label'])

    # Also plot intrinsic AUC as dashed lines
    for method, config in METHOD_CONFIG.items():
        col_int = f'{method}_auc_intrinsic'
        if col_int in noise_data.columns:
            summary_int = noise_data.groupby('perturbation_level')[col_int].agg(['mean', 'std'])
            ax.errorbar(summary_int.index, summary_int['mean'], yerr=summary_int['std'],
                        marker='o', capsize=3, linewidth=1.0, markersize=3,
                        color=config['color'], linestyle='--', alpha=0.5,
                        label=f"{config['label']} (Intrinsic)")

    ax.set_xlabel('Expression Noise Level', fontsize=7)
    ax.set_ylabel('Detection AUC', fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=5, loc='upper right', bbox_to_anchor=(1.0, 1.0))


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel B."""
    fig = create()
    save_figure(fig, 'figS1/panel_b')
    plt.close(fig)
    print("Figure S1 panel B complete.")


if __name__ == '__main__':
    main()
