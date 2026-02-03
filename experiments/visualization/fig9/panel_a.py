"""Figure 9 Panel A: H&E Tissue with Transplant Overlay."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig9.combined import _load_results, _draw_tissue_panel
    if df is None:
        df = _load_results()
    _draw_tissue_panel(ax, df)


def create():
    from fig9.combined import create_panel_a
    return create_panel_a()


def main():
    fig = create()
    save_figure(fig, 'fig9/panel_a')
    plt.close(fig)
    print("Figure 9 panel A complete.")


if __name__ == '__main__':
    main()
