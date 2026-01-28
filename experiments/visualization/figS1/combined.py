"""
Supplementary Figure S1: Noise Robustness Analysis

Caption: Inverse position prediction maintains high ectopic detection
performance across varying noise levels.

Panels:
(a) Performance vs dropout rate
(b) Performance vs expression noise level
(c) Comparison at high noise (50% dropout)

Data source: exp04_noise_robustness.csv
"""

import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    DOUBLE_COL, RESULTS_DIR
)

from figS1 import panel_a, panel_b, panel_c


def _load_noise_data():
    """Load noise robustness data from exp04."""
    import pandas as pd
    return pd.read_csv(RESULTS_DIR / 'exp04_noise_robustness.csv')


def create_combined():
    """Create combined Figure S1 with all panels."""
    set_nature_style()

    df = _load_noise_data()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.45)

    ax_a = fig.add_subplot(gs[0])
    panel_a.draw(ax_a, df)
    add_panel_label(ax_a, 'a', x=-0.15)

    ax_b = fig.add_subplot(gs[1])
    panel_b.draw(ax_b, df)
    add_panel_label(ax_b, 'b')

    ax_c = fig.add_subplot(gs[2])
    panel_c.draw(ax_c, df)
    add_panel_label(ax_c, 'c', x=-0.25)

    plt.tight_layout()
    return fig


def main():
    """Generate all Figure S1 outputs."""
    print("Generating Figure S1 panels...")

    fig_a = panel_a.create()
    save_figure(fig_a, 'figS1/panel_a')
    plt.close(fig_a)

    fig_b = panel_b.create()
    save_figure(fig_b, 'figS1/panel_b')
    plt.close(fig_b)

    fig_c = panel_c.create()
    save_figure(fig_c, 'figS1/panel_c')
    plt.close(fig_c)

    fig_combined = create_combined()
    save_figure(fig_combined, 'figS1/combined')
    plt.close(fig_combined)

    print("Figure S1 complete.")


if __name__ == '__main__':
    main()
