"""
Supplementary Figure S5: Scalability Analysis

Caption: Inverse position prediction scales efficiently
to large spatial transcriptomics datasets.

Panels:
(a) Runtime vs number of spots
(b) GPU memory usage vs number of spots
(c) AUC stability across dataset sizes

Data source: exp13_scalability_analysis.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, SINGLE_COL, DOUBLE_COL, RESULTS_DIR
)

from figS5 import panel_a, panel_b, panel_c


def _load_scalability_data():
    """Load scalability analysis results from exp13."""
    for filename in ['exp13_scalability_analysis.csv', 'exp13_scalability.csv']:
        path = RESULTS_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            df = df[df['status'] == 'success'] if 'status' in df.columns else df
            return df
    raise FileNotFoundError("No scalability data found")


def create_combined():
    """Create combined Figure S5 with all panels."""
    set_nature_style()

    df = _load_scalability_data()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.45)

    ax_a = fig.add_subplot(gs[0])
    panel_a.draw(ax_a, df)
    add_panel_label(ax_a, 'a', x=-0.15)

    ax_b = fig.add_subplot(gs[1])
    panel_b.draw(ax_b, df)
    add_panel_label(ax_b, 'b')

    ax_c = fig.add_subplot(gs[2])
    panel_c.draw(ax_c, df)
    add_panel_label(ax_c, 'c')

    plt.tight_layout()
    return fig


def main():
    """Generate all Figure S5 outputs."""
    print("Generating Figure S5 panels...")

    try:
        _load_scalability_data()
    except FileNotFoundError:
        print("  Warning: No scalability data found (exp13_scalability.csv)")
        print("  Skipping Figure S5 (requires exp13 to be run first)")
        return

    fig_a = panel_a.create()
    save_figure(fig_a, 'figS5/panel_a')
    plt.close(fig_a)

    fig_b = panel_b.create()
    save_figure(fig_b, 'figS5/panel_b')
    plt.close(fig_b)

    fig_c = panel_c.create()
    save_figure(fig_c, 'figS5/panel_c')
    plt.close(fig_c)

    fig_combined = create_combined()
    save_figure(fig_combined, 'figS5/combined')
    plt.close(fig_combined)

    print("Figure S5 complete.")


if __name__ == '__main__':
    main()
