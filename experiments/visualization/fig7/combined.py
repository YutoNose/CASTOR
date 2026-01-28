"""
Figure 7: Real Data Gene Contribution Analysis

Applies inverse prediction to unmodified HER2ST data and analyzes which
genes are differentially expressed between spatially high-error vs low-error
spots. This reveals the biological signature of spatial anomalies.

Panels:
(a) H&E tissue image with position prediction error overlay
(b) Volcano plot of DEGs (high-error vs low-error spots)
(c) Top gene barplot
(d) Known cancer marker enrichment

Data source: results/exp16_gene_analysis_full/ (or exp16_gene_analysis/)
Tissue images: HER2ST H&E stained sections (Andersson et al. 2021)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from pathlib import Path
from PIL import Image
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
# Add project root for imports (core/, data/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, SINGLE_COL, DOUBLE_COL, RESULTS_DIR,
)

HER2ST_DIR = Path("/home/yutonose/Projects/her2st")
IMG_DIR = HER2ST_DIR / "data" / "ST-imgs"


def _find_results_dir():
    """Find gene analysis results directory."""
    for dirname in ['exp16_gene_analysis_full', 'exp16_gene_analysis']:
        d = RESULTS_DIR / dirname
        if d.exists() and any(d.glob("*_combined_deg.csv")):
            return d
    raise FileNotFoundError("No gene analysis results found")


def _load_best_sample(results_dir):
    """Load the sample with most significant DEGs."""
    best_sample = None
    best_n_sig = 0

    for f in results_dir.glob("*_combined_deg.csv"):
        sample_id = f.stem.replace("_combined_deg", "")
        df = pd.read_csv(f)
        n_sig = df["significant"].sum() if "significant" in df.columns else 0
        if n_sig > best_n_sig:
            best_n_sig = n_sig
            best_sample = sample_id

    if best_sample is None:
        files = list(results_dir.glob("*_combined_deg.csv"))
        if not files:
            raise FileNotFoundError("No combined DEG results found")
        best_sample = files[0].stem.replace("_combined_deg", "")

    return best_sample


def _load_tissue_image(sample_id):
    """Load H&E tissue image for a sample."""
    letter = sample_id[0]
    img_dir = IMG_DIR / letter / sample_id
    if img_dir.exists():
        for f in img_dir.glob("*.jpg"):
            return np.array(Image.open(f))
    return None


def _load_sample_with_coords(sample_id):
    """Load sample data including pixel coordinates."""
    from data.generators.her2st import HER2STDataLoader
    loader = HER2STDataLoader(str(HER2ST_DIR))
    X, coords, y_true, meta = loader.load(sample_id)
    return coords, y_true, meta


def _draw_tissue_error_map(ax, results_dir, sample_id):
    """Panel (a): H&E tissue image with position prediction error overlay."""
    spot_file = results_dir / f"{sample_id}_spot_scores.csv"
    if not spot_file.exists():
        ax.text(0.5, 0.5, f'No spot data for {sample_id}',
                ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return

    spot_df = pd.read_csv(spot_file)
    s_pos = spot_df["s_pos"].values
    high_mask = spot_df["high_error"].values.astype(bool)

    # Load tissue image and pixel coords
    img = _load_tissue_image(sample_id)
    coords, y_true, meta = _load_sample_with_coords(sample_id)
    pixel_coords = meta.get("pixel_coords")

    if img is not None and pixel_coords is not None:
        ax.imshow(img, aspect='equal')
        # Zoom to tissue region with padding
        pad = 500
        ax.set_xlim(pixel_coords[:, 0].min() - pad, pixel_coords[:, 0].max() + pad)
        ax.set_ylim(pixel_coords[:, 1].max() + pad, pixel_coords[:, 1].min() - pad)
        sc = ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1],
                        c=s_pos, s=20, cmap='YlOrRd', alpha=0.8,
                        edgecolors='gray', linewidths=0.3)
        # Highlight high-error spots
        ax.scatter(pixel_coords[high_mask, 0], pixel_coords[high_mask, 1],
                   s=35, facecolors='none', edgecolors='black', linewidths=1.2,
                   label=f'High error (top 10%)')
    else:
        # Use array coords
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=s_pos, s=8,
                        cmap='YlOrRd', alpha=0.8,
                        edgecolors='gray', linewidths=0.2)
        ax.scatter(coords[high_mask, 0], coords[high_mask, 1],
                   s=15, facecolors='none', edgecolors='black', linewidths=0.8,
                   label=f'High error (top 10%)')
        ax.set_aspect('equal')

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label('Position Error', fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    ax.legend(fontsize=4, loc='upper right', framealpha=0.8)
    ax.axis('off')


def _draw_volcano(ax, deg_df, sample_id):
    """Panel (b): Volcano plot of DEGs."""
    if len(deg_df) == 0:
        ax.text(0.5, 0.5, 'No DEG data', ha='center', va='center',
                transform=ax.transAxes, fontsize=7)
        return

    pcol = 'combined_pval' if 'combined_pval' in deg_df.columns else 'pval'
    fc_col = 'median_log2fc' if 'median_log2fc' in deg_df.columns else 'log2fc'

    log10p = -np.log10(deg_df[pcol].clip(lower=1e-300))
    fc = deg_df[fc_col]

    sig_mask = deg_df['significant'].values if 'significant' in deg_df.columns else np.zeros(len(deg_df), dtype=bool)
    up_mask = sig_mask & (fc > 0)
    down_mask = sig_mask & (fc < 0)
    ns_mask = ~sig_mask

    ax.scatter(fc[ns_mask], log10p[ns_mask], s=3, c='gray', alpha=0.2)
    ax.scatter(fc[up_mask], log10p[up_mask], s=5, c=COLORS['ectopic'], alpha=0.5,
               label=f'Up ({up_mask.sum()})')
    ax.scatter(fc[down_mask], log10p[down_mask], s=5, c=COLORS['inv_pos'], alpha=0.5,
               label=f'Down ({down_mask.sum()})')

    # Label top genes
    top_up = deg_df[up_mask].nsmallest(8, pcol)
    top_down = deg_df[down_mask].nsmallest(3, pcol)
    for _, row in pd.concat([top_up, top_down]).iterrows():
        gene = row['gene']
        x = row[fc_col]
        y = -np.log10(max(row[pcol], 1e-300))
        ax.annotate(gene, (x, y), fontsize=4, ha='center', va='bottom',
                    xytext=(0, 3), textcoords='offset points')

    ax.set_xlabel('log2 Fold Change\n(high error / low error)', fontsize=7)
    ax.set_ylabel('-log10(p-value)', fontsize=7)
    # Note: horizontal line shows raw p=0.05; point coloring uses adjusted p-values
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', linewidth=0.5, alpha=0.5,
               label='p=0.05 (raw)')
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=5, loc='upper right')


def _draw_top_genes(ax, deg_df, n_genes=15):
    """Panel (c): Top gene barplot."""
    fc_col = 'median_log2fc' if 'median_log2fc' in deg_df.columns else 'log2fc'
    pcol = 'combined_pval' if 'combined_pval' in deg_df.columns else 'pval'

    sig = deg_df[deg_df.get('significant', pd.Series(dtype=bool)).fillna(False)]
    if len(sig) == 0:
        sig = deg_df.nsmallest(n_genes, pcol)

    top = sig.nsmallest(n_genes, pcol)

    y_pos = np.arange(len(top))
    fc_values = top[fc_col].values
    colors = [COLORS['ectopic'] if fc > 0 else COLORS['inv_pos'] for fc in fc_values]

    ax.barh(y_pos, fc_values, color=colors, edgecolor='black', linewidth=0.3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top['gene'].values, fontsize=5)
    ax.set_xlabel('log2 Fold Change', fontsize=7)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()

    for i, (_, row) in enumerate(top.iterrows()):
        p = row[pcol]
        if p < 1e-10:
            label = f'p<1e-10'
        elif p < 0.001:
            label = f'p={p:.1e}'
        else:
            label = f'p={p:.3f}'
        xpos = row[fc_col]
        align = 'left' if xpos >= 0 else 'right'
        offset = 0.02 if xpos >= 0 else -0.02
        ax.text(xpos + offset, i, label, va='center', ha=align, fontsize=4, style='italic')


def _draw_cancer_gene_overlap(ax, deg_df):
    """Panel (d): Known cancer gene enrichment check."""
    cancer_markers = {
        'HER2/Growth': ['ERBB2', 'EGFR', 'GRB7', 'GNAS'],
        'Adhesion': ['SDC1', 'CD24', 'CD44', 'EPCAM', 'MUC1'],
        'Proliferation': ['MKI67', 'PCNA', 'TOP2A', 'CCNB1', 'CDK1'],
        'Immune': ['CD3D', 'CD8A', 'PTPRC', 'CD68', 'HLA-A'],
        'Structural': ['KRT8', 'KRT18', 'KRT19', 'VIM', 'ACTB'],
    }

    fc_col = 'median_log2fc' if 'median_log2fc' in deg_df.columns else 'log2fc'
    pcol = 'padj' if 'padj' in deg_df.columns else 'combined_pval'

    categories = []
    genes_found = []
    fc_values = []
    sig_values = []

    for cat, genes in cancer_markers.items():
        for gene in genes:
            match = deg_df[deg_df['gene'] == gene]
            if len(match) > 0:
                row = match.iloc[0]
                categories.append(cat)
                genes_found.append(gene)
                fc_values.append(row[fc_col])
                p = row[pcol] if pcol in row.index else 1.0
                sig_values.append(p < 0.05)

    if len(genes_found) == 0:
        ax.text(0.5, 0.5, 'No cancer markers found in DEG list',
                ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return

    result_df = pd.DataFrame({
        'category': categories, 'gene': genes_found,
        'log2fc': fc_values, 'significant': sig_values,
    })

    y_pos = np.arange(len(result_df))
    colors_list = []
    for _, row in result_df.iterrows():
        if row['significant']:
            colors_list.append(COLORS['ectopic'] if row['log2fc'] > 0 else COLORS['inv_pos'])
        else:
            colors_list.append('lightgray')

    ax.barh(y_pos, result_df['log2fc'].values, color=colors_list,
            edgecolor='black', linewidth=0.3, height=0.7)

    labels = [f"{row['gene']}" for _, row in result_df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel('log2 FC in high-error spots', fontsize=7)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.5)

    prev_cat = None
    for i, (_, row) in enumerate(result_df.iterrows()):
        if row['category'] != prev_cat:
            ax.text(ax.get_xlim()[1] * 0.95, i, row['category'],
                    fontsize=4, va='center', ha='right', style='italic', color='gray')
            prev_cat = row['category']

    ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor=COLORS['ectopic'], edgecolor='black', linewidth=0.3, label='Sig. Up'),
        Patch(facecolor=COLORS['inv_pos'], edgecolor='black', linewidth=0.3, label='Sig. Down'),
        Patch(facecolor='lightgray', edgecolor='black', linewidth=0.3, label='Not sig.'),
    ]
    ax.legend(handles=legend_elements, fontsize=4, loc='lower right')


# =============================================================================
# Individual Panel Functions
# =============================================================================

def create_panel_a(sample_id=None):
    """Panel (a): H&E with error overlay - standalone."""
    set_nature_style()
    results_dir = _find_results_dir()
    if sample_id is None:
        sample_id = _load_best_sample(results_dir)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.3))
    _draw_tissue_error_map(ax, results_dir, sample_id)
    plt.tight_layout()
    return fig


def create_panel_b(sample_id=None):
    """Panel (b): Volcano plot - standalone."""
    set_nature_style()
    results_dir = _find_results_dir()
    if sample_id is None:
        sample_id = _load_best_sample(results_dir)
    deg_file = results_dir / f"{sample_id}_combined_deg.csv"
    deg_df = pd.read_csv(deg_file)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.1))
    _draw_volcano(ax, deg_df, sample_id)
    plt.tight_layout()
    return fig


def create_panel_c(sample_id=None):
    """Panel (c): Top genes - standalone."""
    set_nature_style()
    results_dir = _find_results_dir()
    if sample_id is None:
        sample_id = _load_best_sample(results_dir)
    deg_file = results_dir / f"{sample_id}_combined_deg.csv"
    deg_df = pd.read_csv(deg_file)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 1.3))
    _draw_top_genes(ax, deg_df)
    plt.tight_layout()
    return fig


def create_panel_d(sample_id=None):
    """Panel (d): Cancer marker overlap - standalone."""
    set_nature_style()
    results_dir = _find_results_dir()
    if sample_id is None:
        sample_id = _load_best_sample(results_dir)
    deg_file = results_dir / f"{sample_id}_combined_deg.csv"
    deg_df = pd.read_csv(deg_file)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.5))
    _draw_cancer_gene_overlap(ax, deg_df)
    plt.tight_layout()
    return fig


# =============================================================================
# Combined Figure
# =============================================================================

def create_combined():
    """Create combined Figure 7 with tissue images."""
    set_nature_style()

    results_dir = _find_results_dir()
    sample_id = _load_best_sample(results_dir)

    deg_file = results_dir / f"{sample_id}_combined_deg.csv"
    deg_df = pd.read_csv(deg_file)

    print(f"  Using sample: {sample_id}")
    n_sig = deg_df['significant'].sum() if 'significant' in deg_df.columns else 0
    print(f"  Significant DEGs: {n_sig}")

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.65))
    gs = fig.add_gridspec(2, 2, hspace=0.65, wspace=0.5,
                          left=0.06, right=0.96, top=0.93, bottom=0.07)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_tissue_error_map(ax_a, results_dir, sample_id)
    add_panel_label(ax_a, 'a', x=-0.02, y=1.05)

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_volcano(ax_b, deg_df, sample_id)
    add_panel_label(ax_b, 'b', x=-0.15, y=1.05)

    ax_c = fig.add_subplot(gs[1, 0])
    _draw_top_genes(ax_c, deg_df, n_genes=15)
    add_panel_label(ax_c, 'c', x=-0.2, y=1.05)

    ax_d = fig.add_subplot(gs[1, 1])
    _draw_cancer_gene_overlap(ax_d, deg_df)
    add_panel_label(ax_d, 'd', x=-0.15, y=1.05)

    return fig


def main():
    """Generate all Figure 7 outputs."""
    print("Generating Figure 7 panels...")

    fig_a = create_panel_a()
    save_figure(fig_a, 'fig7/panel_a')
    plt.close(fig_a)

    fig_b = create_panel_b()
    save_figure(fig_b, 'fig7/panel_b')
    plt.close(fig_b)

    fig_c = create_panel_c()
    save_figure(fig_c, 'fig7/panel_c')
    plt.close(fig_c)

    fig_d = create_panel_d()
    save_figure(fig_d, 'fig7/panel_d')
    plt.close(fig_d)

    fig_combined = create_combined()
    save_figure(fig_combined, 'fig7/combined')
    plt.close(fig_combined)

    print("Figure 7 complete.")


if __name__ == '__main__':
    main()
