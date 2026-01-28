"""
Figure S5 Panel A: Runtime vs Dataset Size

Runtime scaling analysis.
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
    """Draw runtime panel on given axes."""
    if df is None:
        df = _load_scalability_data()

    summary = df.groupby('n_spots')['time_total_sec'].agg(['mean', 'std']).reset_index()
    summary = summary.sort_values('n_spots')

    ax.plot(summary['n_spots'], summary['mean'], 'o-',
            color=COLORS['inv_pos'], linewidth=1.5, markersize=4,
            label='Inv_PosError')
    ax.fill_between(summary['n_spots'],
                    summary['mean'] - summary['std'],
                    summary['mean'] + summary['std'],
                    alpha=0.2, color=COLORS['inv_pos'])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Spots', fontsize=8)
    ax.set_ylabel('Total Runtime (seconds)', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    for t, label in [(60, '1 min'), (600, '10 min')]:
        if summary['mean'].max() > t * 0.5:
            ax.axhline(t, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.text(summary['n_spots'].iloc[0], t * 1.15, label,
                    fontsize=5, color='gray')

    if 'time_train_sec' in df.columns and 'time_inference_sec' in df.columns:
        train_summary = df.groupby('n_spots')['time_train_sec'].mean().reset_index()
        train_summary = train_summary.sort_values('n_spots')
        ax.plot(train_summary['n_spots'], train_summary['time_train_sec'],
                's--', color=COLORS['pca_error'], linewidth=1, markersize=3,
                alpha=0.7, label='Training only')
        ax.legend(fontsize=5, loc='upper left')


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.9))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel A."""
    fig = create()
    save_figure(fig, 'figS5/panel_a')
    plt.close(fig)
    print("Figure S5 panel A complete.")


if __name__ == '__main__':
    main()
