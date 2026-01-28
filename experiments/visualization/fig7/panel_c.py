"""Figure 7 Panel C: Top Gene Barplot."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None, sample_id=None):
    from fig7.combined import _find_results_dir, _load_best_sample, _draw_top_genes
    import pandas as pd
    results_dir = _find_results_dir()
    if sample_id is None:
        sample_id = _load_best_sample(results_dir)
    if df is None:
        deg_file = results_dir / f"{sample_id}_combined_deg.csv"
        df = pd.read_csv(deg_file)
    _draw_top_genes(ax, df)


def create(sample_id=None):
    from fig7.combined import create_panel_c
    return create_panel_c(sample_id)


def main():
    fig = create()
    save_figure(fig, 'fig7/panel_c')
    plt.close(fig)
    print("Figure 7 panel C complete.")


if __name__ == '__main__':
    main()
