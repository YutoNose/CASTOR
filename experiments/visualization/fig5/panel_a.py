"""Figure 5 Panel A: Spatial Visualization."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig5.combined import _draw_spatial_panel
    _draw_spatial_panel(ax)


def create():
    from fig5.combined import create_panel_a
    return create_panel_a()


def main():
    fig = create()
    save_figure(fig, 'fig5/panel_a')
    plt.close(fig)
    print("Figure 5 panel A complete.")


if __name__ == '__main__':
    main()
