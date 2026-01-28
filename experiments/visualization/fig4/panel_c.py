"""Figure 4 Panel C: Combined Detection Performance."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig4.combined import _draw_combined_panel
    _draw_combined_panel(ax)


def create():
    from fig4.combined import create_panel_c
    return create_panel_c()


def main():
    fig = create()
    save_figure(fig, 'fig4/panel_c')
    plt.close(fig)
    print("Figure 4 panel C complete.")


if __name__ == '__main__':
    main()
