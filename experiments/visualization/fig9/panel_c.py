"""Figure 9 Panel C: Detection AUC at region_size=30."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None, region_size=30):
    from fig9.combined import _load_results, _draw_bar_at_max_region
    if df is None:
        df = _load_results()
    _draw_bar_at_max_region(ax, df, region_size=region_size)


def create(region_size=30):
    from fig9.combined import create_panel_c
    return create_panel_c(region_size=region_size)


def main():
    fig = create()
    save_figure(fig, 'fig9/panel_c')
    plt.close(fig)
    print("Figure 9 panel C complete.")


if __name__ == '__main__':
    main()
