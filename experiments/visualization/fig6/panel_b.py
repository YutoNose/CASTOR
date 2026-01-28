"""Figure 6 Panel B: Transplantation Design."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None, sample_id='G2'):
    from fig6.combined import _load_sample_data, _draw_transplant_overlay
    sample_data = _load_sample_data(sample_id)
    _draw_transplant_overlay(ax, sample_data, sample_id)


def create(sample_id='G2'):
    from fig6.combined import create_panel_b
    return create_panel_b(sample_id)


def main():
    fig = create()
    save_figure(fig, 'fig6/panel_b')
    plt.close(fig)
    print("Figure 6 panel B complete.")


if __name__ == '__main__':
    main()
