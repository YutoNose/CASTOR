"""Figure 4 Panel B: Inv_Pos vs PCA Scatter."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig4.combined import _draw_scatter_panel
    _draw_scatter_panel(ax)


def create():
    from fig4.combined import create_panel_b
    return create_panel_b()


def main():
    fig = create()
    save_figure(fig, 'fig4/panel_b')
    plt.close(fig)
    print("Figure 4 panel B complete.")


if __name__ == '__main__':
    main()
