"""Figure 3 Panel B: Method Performance Across Scenarios."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig3.combined import _draw_bar_chart_panel, _load_scenario_data
    if df is None:
        df = _load_scenario_data()
    _draw_bar_chart_panel(ax, df)


def create():
    from fig3.combined import create_panel_b
    return create_panel_b()


def main():
    fig = create()
    save_figure(fig, 'fig3/panel_b')
    plt.close(fig)
    print("Figure 3 panel B complete.")


if __name__ == '__main__':
    main()
