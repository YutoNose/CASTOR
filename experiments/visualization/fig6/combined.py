"""
Figure 6: HER2ST Tumor-to-Normal Transplantation Validation

Semi-synthetic validation using real HER2ST breast cancer data.
Tumor expression profiles are transplanted into normal tissue positions,
creating ground-truth ectopic anomalies for quantitative evaluation.

Panels:
(a) H&E tissue image with pathologist annotations (representative sample)
(b) Transplantation design: tumor spots -> normal positions
(c) Detection AUC across methods (all samples)
(d) Per-sample AUC comparison (top 3 methods)

Data source: exp17_transplantation_full.csv (or exp17_transplantation.csv)
Tissue images: HER2ST H&E stained sections (Andersson et al. 2021)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from PIL import Image
import os
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

HER2ST_DIR = Path(os.environ.get("HER2ST_DIR", "/home/yutonose/Projects/her2st"))
IMG_DIR = HER2ST_DIR / "data" / "ST-imgs"

# Method display config
TRANSPLANT_METHODS = {
    'auc_inv_pos': ('Inv_PosError', COLORS['inv_pos']),
    'auc_inv_self': ('Inv_Self', '#2ECC71'),
    'auc_inv_neighbor': ('Inv_Neighbor', '#8E44AD'),
    'auc_inv_if': ('Inv_IF', '#E67E22'),
    'auc_pca_error': ('PCA_Error', COLORS['pca_error']),
    'auc_neighbor_diff': ('Neighbor_Diff', COLORS['neighbor_diff']),
    'auc_lisa': ('LISA', COLORS['lisa']),
    'auc_lof': ('LOF', COLORS['lof']),
    'auc_isolation_forest': ('Isolation_Forest', COLORS['isolation_forest']),
}

# Tissue label colors
LABEL_COLORS = {
    'invasive cancer': '#E74C3C',
    'cancer in situ': '#FF6B6B',
    'connective tissue': '#3498DB',
    'adipose tissue': '#F39C12',
    'breast glands': '#27AE60',
    'immune infiltrate': '#9B59B6',
    'undetermined': '#BDC3C7',
}


def _load_data():
    """Load transplantation results."""
    for fname in ['exp17_transplantation_full.csv', 'exp17_transplantation.csv']:
        path = RESULTS_DIR / fname
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("No transplantation results found")


def _load_sample_data(sample_id):
    """Load HER2ST sample with tissue image and labels."""
    from data.generators.her2st import HER2STDataLoader

    loader = HER2STDataLoader(str(HER2ST_DIR))
    X, coords, y_true, meta = loader.load(sample_id)

    # Load tissue image directly from subdirectory
    letter = sample_id[0]
    img_dir = IMG_DIR / letter / sample_id
    img = None
    if img_dir.exists():
        for f in img_dir.glob("*.jpg"):
            img = np.array(Image.open(f))
            break

    # Load tissue labels
    labels_file = HER2ST_DIR / "data" / "ST-pat" / "lbl" / f"{sample_id}_labeled_coordinates.tsv"
    labels_df = pd.read_csv(labels_file, sep="\t")
    labels_df = labels_df.dropna(subset=["x", "y"])
    labels_df["array_x"] = labels_df["x"].round().astype(int)
    labels_df["array_y"] = labels_df["y"].round().astype(int)
    labels_df["spot_id"] = labels_df["array_x"].astype(str) + "x" + labels_df["array_y"].astype(str)

    # Align labels with loader output
    import gzip
    counts_file = HER2ST_DIR / "data" / "ST-cnts" / f"{sample_id}.tsv.gz"
    with gzip.open(counts_file, "rt") as f:
        counts_df = pd.read_csv(f, sep="\t", index_col=0, usecols=[0])
    common_spots = list(counts_df.index)[:len(coords)]

    labels_df_indexed = labels_df.set_index("spot_id")
    tissue_labels = []
    for sid in common_spots:
        if sid in labels_df_indexed.index:
            # Handle potential duplicates
            val = labels_df_indexed.loc[sid, "label"]
            if isinstance(val, pd.Series):
                tissue_labels.append(val.iloc[0])
            else:
                tissue_labels.append(val)
        else:
            tissue_labels.append("undetermined")

    return {
        "X": X, "coords": coords, "y_true": y_true,
        "meta": meta, "tissue_image": img,
        "tissue_labels": tissue_labels,
        "pixel_coords": meta.get("pixel_coords"),
    }


def _draw_tissue_with_labels(ax, sample_data, sample_id):
    """Panel (a): H&E tissue image with pathologist annotation overlay."""
    img = sample_data["tissue_image"]
    pixel_coords = sample_data["pixel_coords"]
    tissue_labels = sample_data["tissue_labels"]

    if img is not None and pixel_coords is not None:
        ax.imshow(img, aspect='equal')
        # Zoom to tissue region with padding
        pad = 500
        ax.set_xlim(pixel_coords[:, 0].min() - pad, pixel_coords[:, 0].max() + pad)
        ax.set_ylim(pixel_coords[:, 1].max() + pad, pixel_coords[:, 1].min() - pad)
        # Overlay spots colored by tissue label
        for label, color in LABEL_COLORS.items():
            mask = np.array([l == label for l in tissue_labels])
            if mask.sum() > 0:
                ax.scatter(pixel_coords[mask, 0], pixel_coords[mask, 1],
                           c=color, s=15, alpha=0.7, label=label,
                           edgecolors='white', linewidths=0.2)
        ax.legend(fontsize=5, loc='lower left', framealpha=0.9,
                  markerscale=1.5, handletextpad=0.3, labelspacing=0.3)
    else:
        # Fallback: plot with array coords
        coords = sample_data["coords"]
        for label, color in LABEL_COLORS.items():
            mask = np.array([l == label for l in tissue_labels])
            if mask.sum() > 0:
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           c=color, s=15, alpha=0.7, label=label,
                           edgecolors='gray', linewidths=0.3)
        ax.legend(fontsize=5, loc='lower left')

    ax.axis('off')


def _draw_transplant_overlay(ax, sample_data, sample_id, seed=42):
    """Panel (b): Tissue image with transplanted spots highlighted."""
    img = sample_data["tissue_image"]
    pixel_coords = sample_data["pixel_coords"]
    tissue_labels = sample_data["tissue_labels"]

    # Determine tumor and normal masks
    tumor_labels_set = {"invasive cancer", "cancer in situ", "dcis"}
    normal_labels_set = {"connective tissue", "adipose tissue", "breast glands"}

    tumor_mask = np.array([l in tumor_labels_set for l in tissue_labels], dtype=bool)
    normal_mask = np.array([l in normal_labels_set for l in tissue_labels], dtype=bool)

    # Simulate transplantation (same as exp17 logic)
    rng = np.random.RandomState(seed)
    tumor_idx = np.where(tumor_mask)[0]
    normal_idx = np.where(normal_mask)[0]
    n_transplant = min(30, len(normal_idx), len(tumor_idx))
    recipient_idx = rng.choice(normal_idx, n_transplant, replace=False)
    donor_idx = rng.choice(tumor_idx, n_transplant, replace=True)

    if img is not None and pixel_coords is not None:
        ax.imshow(img, aspect='equal')
        # Zoom to tissue region with padding
        pad = 500
        ax.set_xlim(pixel_coords[:, 0].min() - pad, pixel_coords[:, 0].max() + pad)
        ax.set_ylim(pixel_coords[:, 1].max() + pad, pixel_coords[:, 1].min() - pad)
        # All spots dim
        ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1],
                   c='gray', s=8, alpha=0.1)
        # Tumor region
        ax.scatter(pixel_coords[tumor_mask, 0], pixel_coords[tumor_mask, 1],
                   c=COLORS['ectopic'], s=15, alpha=0.45, label='Tumor')
        # Normal region
        ax.scatter(pixel_coords[normal_mask, 0], pixel_coords[normal_mask, 1],
                   c=COLORS['inv_pos'], s=15, alpha=0.45, label='Normal')
        # Transplanted spots
        ax.scatter(pixel_coords[recipient_idx, 0], pixel_coords[recipient_idx, 1],
                   c='yellow', s=120, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=10,
                   label=f'Transplanted (n={n_transplant})')
        # Select a few short-distance donor→recipient pairs for clean arrows
        dists = np.sqrt(
            (pixel_coords[donor_idx, 0] - pixel_coords[recipient_idx, 0])**2 +
            (pixel_coords[donor_idx, 1] - pixel_coords[recipient_idx, 1])**2
        )
        sorted_pairs = np.argsort(dists)
        for rank in sorted_pairs[:4]:
            ax.annotate(
                '', xy=(pixel_coords[recipient_idx[rank], 0],
                        pixel_coords[recipient_idx[rank], 1]),
                xytext=(pixel_coords[donor_idx[rank], 0],
                        pixel_coords[donor_idx[rank], 1]),
                arrowprops=dict(arrowstyle='-|>', color='#222222',
                                lw=2.0, alpha=0.85,
                                connectionstyle='arc3,rad=0.2',
                                mutation_scale=18,
                                shrinkA=5, shrinkB=5),
            )
    else:
        coords = sample_data["coords"]
        ax.scatter(coords[:, 0], coords[:, 1], c='gray', s=8, alpha=0.2)
        ax.scatter(coords[tumor_mask, 0], coords[tumor_mask, 1],
                   c=COLORS['ectopic'], s=10, alpha=0.5, label='Tumor')
        ax.scatter(coords[normal_mask, 0], coords[normal_mask, 1],
                   c=COLORS['inv_pos'], s=10, alpha=0.5, label='Normal')
        ax.scatter(coords[recipient_idx, 0], coords[recipient_idx, 1],
                   c=COLORS['ectopic'], s=40, marker='*',
                   edgecolors='black', linewidths=0.8, zorder=10,
                   label=f'Transplanted (n={n_transplant})')

    ax.legend(fontsize=5, loc='lower left', framealpha=0.8,
              markerscale=1.2, handletextpad=0.3, labelspacing=0.3)
    ax.axis('off')


def _draw_method_comparison(ax, df):
    """Panel (c): Overall method AUC comparison (bar chart) with AUPRC annotations."""
    auc_cols = [c for c in TRANSPLANT_METHODS if c in df.columns]

    method_stats = []
    for col in auc_cols:
        name, color = TRANSPLANT_METHODS[col]
        vals = df[col].dropna()
        # Try to get AUPRC
        auprc_col = col.replace('auc_', 'auprc_')
        auprc_val = None
        if auprc_col in df.columns:
            auprc_vals = df[auprc_col].dropna()
            if len(auprc_vals) > 0:
                auprc_val = auprc_vals.mean()
        method_stats.append({
            'name': name, 'color': color,
            'mean': vals.mean(), 'std': vals.std(),
            'auprc': auprc_val,
        })

    method_stats.sort(key=lambda x: x['mean'], reverse=True)

    y_pos = np.arange(len(method_stats))
    means = [m['mean'] for m in method_stats]
    stds = [m['std'] for m in method_stats]
    colors = [m['color'] for m in method_stats]
    names = [m['name'] for m in method_stats]

    ax.barh(y_pos, means, xerr=stds, capsize=3,
            color=colors, edgecolor='black', linewidth=0.3, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('Transplant Detection AUC', fontsize=7)
    ax.set_xlim(0, 1.15)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)

    for i, (m, s) in enumerate(zip(means, stds)):
        auprc = method_stats[i].get('auprc')
        if auprc is not None:
            label = f'{m:.3f} (PR:{auprc:.3f})'
        else:
            label = f'{m:.3f}'
        ax.text(min(m + s + 0.03, 1.1), i, label, va='center', fontsize=5)

    ax.invert_yaxis()


def _draw_per_sample(ax, df):
    """Panel (d): Per-sample AUC for top methods."""
    top_methods = ['auc_inv_pos', 'auc_lisa', 'auc_neighbor_diff']
    available = [m for m in top_methods if m in df.columns]

    samples = sorted(df['sample_id'].unique())
    x_pos = np.arange(len(samples))
    width = 0.8 / len(available)

    for i, col in enumerate(available):
        name, color = TRANSPLANT_METHODS[col]
        means = []
        stds = []
        for sample in samples:
            vals = df[df['sample_id'] == sample][col].dropna()
            means.append(vals.mean())
            stds.append(vals.std())

        offsets = x_pos + (i - len(available)/2 + 0.5) * width
        ax.bar(offsets, means, width, yerr=stds, capsize=2,
               label=name, color=color,
               edgecolor='black', linewidth=0.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(samples, fontsize=6)
    ax.set_ylabel('Transplant Detection AUC', fontsize=7)
    ax.set_xlabel('HER2ST Sample', fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=5, loc='lower right')


# =============================================================================
# Individual Panel Functions
# =============================================================================

def create_panel_a(sample_id='G2'):
    """Panel (a): H&E with annotations - standalone."""
    set_nature_style()
    sample_data = _load_sample_data(sample_id)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.3))
    _draw_tissue_with_labels(ax, sample_data, sample_id)
    plt.tight_layout()
    return fig


def create_panel_b(sample_id='G2'):
    """Panel (b): Transplant overlay - standalone."""
    set_nature_style()
    sample_data = _load_sample_data(sample_id)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.3))
    _draw_transplant_overlay(ax, sample_data, sample_id)
    plt.tight_layout()
    return fig


def create_panel_c():
    """Panel (c): Method comparison - standalone."""
    set_nature_style()
    df = _load_data()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.4, SINGLE_COL * 1.0))
    _draw_method_comparison(ax, df)
    plt.tight_layout()
    return fig


def create_panel_d():
    """Panel (d): Per-sample comparison - standalone."""
    set_nature_style()
    df = _load_data()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.0))
    _draw_per_sample(ax, df)
    plt.tight_layout()
    return fig


# =============================================================================
# Combined Figure
# =============================================================================

def create_combined(sample_id='G2'):
    """Create combined Figure 6 with tissue images."""
    set_nature_style()
    df = _load_data()
    sample_data = _load_sample_data(sample_id)

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.75))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.4,
                          left=0.06, right=0.97, top=0.94, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_tissue_with_labels(ax_a, sample_data, sample_id)
    add_panel_label(ax_a, 'a', x=-0.02, y=1.05)

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_transplant_overlay(ax_b, sample_data, sample_id)
    add_panel_label(ax_b, 'b', x=-0.02, y=1.05)

    ax_c = fig.add_subplot(gs[1, 0])
    _draw_method_comparison(ax_c, df)
    add_panel_label(ax_c, 'c', x=-0.15, y=1.05)

    ax_d = fig.add_subplot(gs[1, 1])
    _draw_per_sample(ax_d, df)
    add_panel_label(ax_d, 'd', x=-0.12, y=1.05)

    return fig


def main():
    """Generate all Figure 6 outputs."""
    print("Generating Figure 6 panels...")

    sample_id = 'G2'

    fig_a = create_panel_a(sample_id)
    save_figure(fig_a, 'fig6/panel_a')
    plt.close(fig_a)

    fig_b = create_panel_b(sample_id)
    save_figure(fig_b, 'fig6/panel_b')
    plt.close(fig_b)

    fig_c = create_panel_c()
    save_figure(fig_c, 'fig6/panel_c')
    plt.close(fig_c)

    fig_d = create_panel_d()
    save_figure(fig_d, 'fig6/panel_d')
    plt.close(fig_d)

    fig_combined = create_combined(sample_id)
    save_figure(fig_combined, 'fig6/combined')
    plt.close(fig_combined)

    print("Figure 6 complete.")


if __name__ == '__main__':
    main()
