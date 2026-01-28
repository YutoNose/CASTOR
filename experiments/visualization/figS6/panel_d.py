"""
Figure S6 Panel D: Parameter Summary Table

Synthetic data parameter summary.
"""

import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    """Draw parameter summary table on given axes."""
    params = [
        ['Number of spots', '3,000'],
        ['Number of genes', '500'],
        ['Spatial dimensions', '2D (unit square)'],
        ['Ectopic anomalies', '100 (3.3%)'],
        ['Intrinsic anomalies', '300 (10%)'],
        ['Ectopic donor distance', '> 0.5 (normalized)'],
        ['Intrinsic perturbation', '3\u201310x gene upregulation'],
        ['Expression distribution', 'ZINB (r=2, dropout=0.3)'],
        ['Spatial graph', 'k=15 neighbors'],
        ['Number of seeds', '30'],
    ]

    ax.axis('off')

    table = ax.table(
        cellText=params,
        colLabels=['Parameter', 'Value'],
        cellLoc='left',
        loc='center',
        colWidths=[0.55, 0.45]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.5)

    for i in range(2):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(fontweight='bold')


def create():
    """Create standalone panel figure."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.0))
    draw(ax)
    plt.tight_layout()
    return fig


def main():
    """Generate panel D."""
    fig = create()
    save_figure(fig, 'figS6/panel_d')
    plt.close(fig)
    print("Figure S6 panel D complete.")


if __name__ == '__main__':
    main()
