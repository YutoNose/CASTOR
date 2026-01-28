"""
Figure S5 Panel B: GPU Memory Usage

GPU memory usage vs dataset size.
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
    """Draw GPU memory panel on given axes."""
    if df is None:
        df = _load_scalability_data()

    summary = df.groupby('n_spots')['gpu_mem_peak_train_mb'].agg(['mean', 'std']).reset_index()
    summary = summary.sort_values('n_spots')

    ax.plot(summary['n_spots'], summary['mean'], 'o-',
            color=COLORS['inv_pos'], linewidth=1.5, markersize=4,
            label='Peak GPU (training)')
    ax.fill_between(summary['n_spots'],
                    summary['mean'] - summary['std'],
                    summary['mean'] + summary['std'],
                    alpha=0.2, color=COLORS['inv_pos'])

    if 'gpu_mem_peak_inference_mb' in df.columns:
        inf_summary = df.groupby('n_spots')['gpu_mem_peak_inference_mb'].agg(['mean', 'std']).reset_index()
        inf_summary = inf_summary.sort_values('n_spots')
        ax.plot(inf_summary['n_spots'], inf_summary['mean'], 's--',
                color=COLORS['pca_error'], linewidth=1, markersize=3,
                alpha=0.7, label='Peak GPU (inference)')

    ax.set_xscale('log')
    ax.set_xlabel('Number of Spots', fontsize=8)
    ax.set_ylabel('GPU Memory (MB)', fontsize=8)
    ax.legend(fontsize=5, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    for mem, label in [(1000, '1 GB'), (8000, '8 GB')]:
        if summary['mean'].max() > mem * 0.3:
            ax.axhline(mem, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.text(summary['n_spots'].iloc[0], mem * 1.05, label,
                    fontsize=5, color='gray')


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
    save_figure(fig, 'figS5/panel_b')
    plt.close(fig)
    print("Figure S5 panel B complete.")


if __name__ == '__main__':
    main()
