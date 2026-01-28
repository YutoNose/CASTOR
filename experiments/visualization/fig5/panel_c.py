"""Figure 5 Panel C: Per-Sample AUC Comparison."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig5.combined import _draw_persample_panel, _load_her2st_data
    if df is None:
        df, _ = _load_her2st_data()
    _draw_persample_panel(ax, df)


def create():
    from fig5.combined import create_panel_c
    return create_panel_c()


def main():
    fig = create()
    save_figure(fig, 'fig5/panel_c')
    plt.close(fig)
    print("Figure 5 panel C complete.")


if __name__ == '__main__':
    main()
