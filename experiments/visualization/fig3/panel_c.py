"""Figure 3 Panel C: Hard Ectopic Detail."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig3.combined import _draw_hard_ectopic_panel, _load_scenario_data
    if df is None:
        df = _load_scenario_data()
    _draw_hard_ectopic_panel(ax, df)


def create():
    from fig3.combined import create_panel_c
    return create_panel_c()


def main():
    fig = create()
    save_figure(fig, 'fig3/panel_c')
    plt.close(fig)
    print("Figure 3 panel C complete.")


if __name__ == '__main__':
    main()
