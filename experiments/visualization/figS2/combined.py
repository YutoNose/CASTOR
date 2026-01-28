"""
Supplementary Figure S2: Ablation Study

Caption: Ablation study shows the contribution of each component
to the inverse position prediction performance.

Panels:
(a) Position loss weight (lambda_pos) ablation
(b) Hidden dimension ablation
(c) Number of neighbors (k) ablation
(d) Selectivity (Ectopic - Intrinsic AUC) summary

Data source: exp07_ablation_studies.csv
"""

import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    DOUBLE_COL, RESULTS_DIR
)

from figS2 import panel_a, panel_b, panel_c, panel_d


def _load_ablation_data():
    """Load ablation study results from exp07."""
    import pandas as pd
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


def create_combined():
    """Create combined Figure S2 with all panels."""
    set_nature_style()

    df = _load_ablation_data()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.45)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a.draw(ax_a, df)
    add_panel_label(ax_a, 'a', x=-0.15)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_b.draw(ax_b, df)
    add_panel_label(ax_b, 'b')

    ax_c = fig.add_subplot(gs[1, 0])
    panel_c.draw(ax_c, df)
    add_panel_label(ax_c, 'c', x=-0.15)

    ax_d = fig.add_subplot(gs[1, 1])
    panel_d.draw(ax_d, df)
    add_panel_label(ax_d, 'd')

    plt.tight_layout()
    return fig


def main():
    """Generate all Figure S2 outputs."""
    print("Generating Figure S2 panels...")

    fig_a = panel_a.create()
    save_figure(fig_a, 'figS2/panel_a')
    plt.close(fig_a)

    fig_b = panel_b.create()
    save_figure(fig_b, 'figS2/panel_b')
    plt.close(fig_b)

    fig_c = panel_c.create()
    save_figure(fig_c, 'figS2/panel_c')
    plt.close(fig_c)

    fig_d = panel_d.create()
    save_figure(fig_d, 'figS2/panel_d')
    plt.close(fig_d)

    fig_combined = create_combined()
    save_figure(fig_combined, 'figS2/combined')
    plt.close(fig_combined)

    print("Figure S2 complete.")


if __name__ == '__main__':
    main()
