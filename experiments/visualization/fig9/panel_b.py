"""Figure 9 Panel B: AUC vs Region Size."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None):
    from fig9.combined import _load_results, _draw_auc_vs_region_size
    if df is None:
        df = _load_results()
    _draw_auc_vs_region_size(ax, df)


def create():
    from fig9.combined import create_panel_b
    return create_panel_b()


def main():
    fig = create()
    save_figure(fig, 'fig9/panel_b')
    plt.close(fig)
    print("Figure 9 panel B complete.")


if __name__ == '__main__':
    main()
