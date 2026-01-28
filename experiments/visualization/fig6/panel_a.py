"""Figure 6 Panel A: H&E Tissue with Annotations."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None, sample_id='G2'):
    from fig6.combined import _load_sample_data, _draw_tissue_with_labels
    sample_data = _load_sample_data(sample_id)
    _draw_tissue_with_labels(ax, sample_data, sample_id)


def create(sample_id='G2'):
    from fig6.combined import create_panel_a
    return create_panel_a(sample_id)


def main():
    fig = create()
    save_figure(fig, 'fig6/panel_a')
    plt.close(fig)
    print("Figure 6 panel A complete.")


if __name__ == '__main__':
    main()
