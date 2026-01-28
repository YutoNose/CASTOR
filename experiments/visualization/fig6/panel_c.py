"""Figure 6 Panel C: Method Comparison."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig6.combined import _draw_method_comparison, _load_data
    if df is None:
        df = _load_data()
    _draw_method_comparison(ax, df)


def create():
    from fig6.combined import create_panel_c
    return create_panel_c()


def main():
    fig = create()
    save_figure(fig, 'fig6/panel_c')
    plt.close(fig)
    print("Figure 6 panel C complete.")


if __name__ == '__main__':
    main()
