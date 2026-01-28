"""
Supplementary Figure S4: Statistical Analysis

Caption: Statistical comparison across methods with
multiple testing correction and effect sizes.

Panels:
(a) Full method comparison heatmap (Ectopic + Intrinsic AUC)
(b) Pairwise statistical significance (Bonferroni-corrected)
(c) Effect sizes (Cohen's d) vs Inv_PosError
(d) 95% Confidence intervals

Data source: exp01_cross-detection_auc.csv (per-seed AUCs)
             exp02_competitor_comparison.csv (competitor methods)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, SINGLE_COL, DOUBLE_COL, RESULTS_DIR
)

from figS4 import panel_a, panel_b, panel_c, panel_d


def _load_per_seed_data():
    """Load per-seed AUC data from exp01."""
    return pd.read_csv(RESULTS_DIR / 'exp01_cross-detection_auc.csv')


def create_combined():
    """Create combined Figure S4 with all panels."""
    set_nature_style()

    df = _load_per_seed_data()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.6))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a.draw(ax_a, df)
    add_panel_label(ax_a, 'a', x=-0.2)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_b.draw(ax_b, df)
    add_panel_label(ax_b, 'b', x=-0.15)

    ax_c = fig.add_subplot(gs[1, 0])
    panel_c.draw(ax_c, df)
    add_panel_label(ax_c, 'c', x=-0.2)

    ax_d = fig.add_subplot(gs[1, 1])
    panel_d.draw(ax_d, df)
    add_panel_label(ax_d, 'd', x=-0.15)

    plt.tight_layout()
    return fig


# Alias for backward compatibility
create_figS4 = create_combined


def main():
    """Generate all Figure S4 outputs."""
    print("Generating Figure S4 panels...")

    fig_a = panel_a.create()
    save_figure(fig_a, 'figS4/panel_a')
    plt.close(fig_a)

    fig_b = panel_b.create()
    save_figure(fig_b, 'figS4/panel_b')
    plt.close(fig_b)

    fig_c = panel_c.create()
    save_figure(fig_c, 'figS4/panel_c')
    plt.close(fig_c)

    fig_d = panel_d.create()
    save_figure(fig_d, 'figS4/panel_d')
    plt.close(fig_d)

    fig_combined = create_combined()
    save_figure(fig_combined, 'figS4/combined')
    plt.close(fig_combined)

    print("Figure S4 complete.")


if __name__ == '__main__':
    main()
