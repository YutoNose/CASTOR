"""Figure 3 Panel A: Scenario Illustration."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig3.combined import _draw_scenario_panel
    _draw_scenario_panel(ax)


def create():
    from fig3.combined import create_panel_a
    return create_panel_a()


def main():
    fig = create()
    save_figure(fig, 'fig3/panel_a')
    plt.close(fig)
    print("Figure 3 panel A complete.")


if __name__ == '__main__':
    main()
