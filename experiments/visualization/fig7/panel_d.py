"""Figure 7 Panel D: Cancer Marker Enrichment."""
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import set_nature_style, save_figure, SINGLE_COL


def draw(ax, df=None, sample_id=None):
    from fig7.combined import _find_results_dir, _load_best_sample, _draw_cancer_gene_overlap
    import pandas as pd
    results_dir = _find_results_dir()
    if sample_id is None:
        sample_id = _load_best_sample(results_dir)
    if df is None:
        deg_file = results_dir / f"{sample_id}_combined_deg.csv"
        df = pd.read_csv(deg_file)
    _draw_cancer_gene_overlap(ax, df)


def create(sample_id=None):
    from fig7.combined import create_panel_d
    return create_panel_d(sample_id)


def main():
    fig = create()
    save_figure(fig, 'fig7/panel_d')
    plt.close(fig)
    print("Figure 7 panel D complete.")


if __name__ == '__main__':
    main()
