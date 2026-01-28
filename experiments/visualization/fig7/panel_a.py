"""Figure 7 Panel A: Tissue Error Map."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None, sample_id=None):
    from fig7.combined import _find_results_dir, _load_best_sample, _draw_tissue_error_map
    results_dir = _find_results_dir()
    if sample_id is None:
        sample_id = _load_best_sample(results_dir)
    _draw_tissue_error_map(ax, results_dir, sample_id)


def create(sample_id=None):
    from fig7.combined import create_panel_a
    return create_panel_a(sample_id)


def main():
    fig = create()
    save_figure(fig, 'fig7/panel_a')
    plt.close(fig)
    print("Figure 7 panel A complete.")


if __name__ == '__main__':
    main()
