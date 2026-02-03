"""Figure 9 Panel D: delta-AUC (Inv_PosError - LISA) vs Region Size."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig9.combined import _load_results, _draw_delta_auc
    if df is None:
        df = _load_results()
    _draw_delta_auc(ax, df)


def create():
    from fig9.combined import create_panel_d
    return create_panel_d()


def main():
    fig = create()
    save_figure(fig, 'fig9/panel_d')
    plt.close(fig)
    print("Figure 9 panel D complete.")


if __name__ == '__main__':
    main()
