"""
Figure 5: Real Data Validation - HER2ST (Combined)

Caption: Position prediction error applied to HER2-positive breast cancer
spatial transcriptomics data from her2st_full.csv.

Panels:
(a) Spatial visualization (conceptual)
(b) Method comparison boxplot across samples
(c) Per-sample AUC comparison

Data source: her2st_full.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import os

from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, SINGLE_COL, DOUBLE_COL, RESULTS_DIR
)

_default_her2st = Path(__file__).resolve().parent.parent.parent.parent.parent / "her2st"
HER2ST_DIR = Path(os.environ.get("HER2ST_DIR", str(_default_her2st)))

# Method display mapping for HER2ST data
# Support both old column names (auc_pos, auc_pca) and new names (auc_inv_pos, auc_pca_error)
HERST_METHODS = {
    # New column names
    'auc_inv_pos': ('Inv_PosError', COLORS['inv_pos']),
    'auc_pca_error': ('PCA_Error', COLORS['pca_error']),
    'auc_lisa': ('LISA', COLORS['lisa']),
    'auc_neighbor_diff': ('Neighbor_Diff', COLORS['neighbor_diff']),
    'auc_lof': ('LOF', COLORS['lof']),
    'auc_isolation_forest': ('IF', COLORS['isolation_forest']),
    'auc_spotsweeper': ('SpotSweeper', COLORS['spotsweeper']),
    'auc_squidpy': ('Squidpy', '#9467BD'),
    'auc_stlearn': ('STLearn', '#BCBD22'),
    'auc_stagate': ('STAGATE', '#17BECF'),
    'auc_graphst': ('GraphST', '#E377C2'),
    # Old column names (from exp11)
    'auc_pos': ('Inv_PosError', COLORS['inv_pos']),
    'auc_pca': ('PCA_Error', COLORS['pca_error']),
    'auc_neighbor': ('Neighbor_Diff', COLORS['neighbor_diff']),
}


def _load_her2st_data():
    """Load HER2ST real data validation results."""
    # Priority order: HER2ST-specific files first, then fallbacks
    # Note: exp14 is HER2ST breast cancer, exp11 is lymph node (different tissue)
    for filename in ['her2st_full.csv', 'her2st_quick.csv',
                     'exp14_her2st_validation.csv',
                     'exp11_real_data_validation.csv']:
        path = RESULTS_DIR / filename
        if path.exists():
            return pd.read_csv(path), filename
    raise FileNotFoundError("No HER2ST result file found")


# =============================================================================
# Individual Panel Functions
# =============================================================================

def create_panel_a():
    """Panel (a): Spatial visualization using real HER2ST data."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 1.0))
    _draw_spatial_panel(ax)
    plt.tight_layout()
    return fig


def create_panel_b():
    """Panel (b): Method comparison boxplot from real HER2ST data."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 2.0, SINGLE_COL * 1.0))

    df, _ = _load_her2st_data()

    # Collect AUC columns that exist
    method_data = []
    for col, (name, color) in HERST_METHODS.items():
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                method_data.append({
                    'name': name,
                    'color': color,
                    'values': vals.values,
                    'mean': vals.mean(),
                })

    # Sort by mean AUC
    method_data.sort(key=lambda x: x['mean'], reverse=True)

    bp = ax.boxplot([m['values'] for m in method_data],
                    labels=[m['name'] for m in method_data],
                    patch_artist=True, widths=0.6)

    for i, (patch, md) in enumerate(zip(bp['boxes'], method_data)):
        patch.set_facecolor(md['color'])
        patch.set_alpha(0.7)

    ax.set_ylabel('Cancer Detection AUC', fontsize=8)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    plt.tight_layout()
    return fig


def create_panel_c():
    """Panel (c): Per-sample AUC comparison from real HER2ST data."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.0))

    df, _ = _load_her2st_data()

    # Support both 'sample_id' and 'dataset' as grouping column
    group_col = 'sample_id' if 'sample_id' in df.columns else 'dataset' if 'dataset' in df.columns else None
    if group_col is None:
        ax.text(0.5, 0.5, 'No sample_id column', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    # Select key methods (support both old and new column names)
    key_cols_candidates = [
        'auc_inv_pos', 'auc_pos',  # Inv_PosError
        'auc_pca_error', 'auc_pca',  # PCA_Error
        'auc_lisa',
        'auc_neighbor_diff', 'auc_neighbor',  # Neighbor_Diff
        'auc_lof',
        'auc_squidpy',
    ]
    key_cols = [c for c in key_cols_candidates if c in df.columns]
    # Remove duplicates (prefer new names)
    seen_methods = set()
    unique_cols = []
    for col in key_cols:
        method_name = HERST_METHODS.get(col, (col,))[0]
        if method_name not in seen_methods:
            seen_methods.add(method_name)
            unique_cols.append(col)
    key_cols = unique_cols[:4]  # Limit to 4 methods

    sample_summary = df.groupby(group_col)[key_cols].mean()

    x = np.arange(len(sample_summary))
    width = 0.2

    for i, col in enumerate(key_cols):
        name, color = HERST_METHODS.get(col, (col, '#888888'))
        ax.bar(x + i * width, sample_summary[col], width,
               label=name, color=color, edgecolor='black', linewidth=0.3)

    ax.set_xticks(x + width * (len(key_cols) - 1) / 2)
    ax.set_xticklabels(sample_summary.index, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Cancer Detection AUC', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper right', fontsize=6)
    plt.tight_layout()
    return fig


# =============================================================================
# Panel Drawing Functions (for combined figure)
# =============================================================================

def _draw_spatial_panel(ax):
    """Panel (a): REAL HER2ST spatial score map."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    # Load spot scores
    spot_file = None
    for dirname in ['exp16_gene_analysis_full', 'exp16_gene_analysis']:
        candidate = RESULTS_DIR / dirname / 'G2_spot_scores.csv'
        if candidate.exists():
            spot_file = candidate
            break
    if spot_file is None:
        ax.text(0.5, 0.5, 'No spot score data', ha='center', va='center', transform=ax.transAxes)
        return

    spot_df = pd.read_csv(spot_file)
    s_pos = spot_df['s_pos'].values
    y_true = spot_df['y_true'].values

    # Load coordinates
    try:
        from data.generators.her2st import HER2STDataLoader
        loader = HER2STDataLoader(str(HER2ST_DIR))
        X, coords, _, meta = loader.load('G2')
        pixel_coords = meta.get('pixel_coords')
    except Exception:
        pixel_coords = None
        coords = np.column_stack([np.arange(len(s_pos)), np.zeros(len(s_pos))])

    # Load tissue image
    from PIL import Image
    img_dir = HER2ST_DIR / "data" / "ST-imgs" / "G" / "G2"
    img = None
    if img_dir.exists():
        for f in img_dir.glob("*.jpg"):
            img = np.array(Image.open(f))
            break

    if img is not None and pixel_coords is not None:
        ax.imshow(img, aspect='equal')
        pad = 500
        ax.set_xlim(pixel_coords[:, 0].min() - pad, pixel_coords[:, 0].max() + pad)
        ax.set_ylim(pixel_coords[:, 1].max() + pad, pixel_coords[:, 1].min() - pad)
        sc = ax.scatter(pixel_coords[:, 0], pixel_coords[:, 1],
                        c=s_pos, cmap='YlOrRd', s=20, alpha=0.8,
                        edgecolors='none')
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=s_pos, cmap='YlOrRd', s=15, alpha=0.8)
        ax.set_aspect('equal')

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label('Position Error', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    ax.axis('off')


def _draw_boxplot_panel(ax, df):
    """Draw panel (b) content on given axes."""
    method_data = []
    for col, (name, color) in HERST_METHODS.items():
        if col in df.columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                method_data.append({
                    'name': name, 'color': color,
                    'values': vals.values, 'mean': vals.mean(),
                })

    method_data.sort(key=lambda x: x['mean'], reverse=True)

    bp = ax.boxplot([m['values'] for m in method_data],
                    labels=[m['name'] for m in method_data],
                    patch_artist=True, widths=0.6)

    for patch, md in zip(bp['boxes'], method_data):
        patch.set_facecolor(md['color'])
        patch.set_alpha(0.7)

    ax.set_ylabel('Cancer Detection AUC', fontsize=7)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='x', rotation=45, labelsize=5)


def _draw_persample_panel(ax, df):
    """Draw panel (c) content on given axes."""
    # Support both 'sample_id' and 'dataset' as grouping column
    group_col = 'sample_id' if 'sample_id' in df.columns else 'dataset' if 'dataset' in df.columns else None
    if group_col is None:
        ax.text(0.5, 0.5, 'No sample data', ha='center', va='center',
                transform=ax.transAxes)
        return

    # Select key methods (support both old and new column names)
    key_cols_candidates = [
        'auc_inv_pos', 'auc_pos',
        'auc_pca_error', 'auc_pca',
        'auc_lisa',
        'auc_neighbor_diff', 'auc_neighbor',
        'auc_lof',
        'auc_squidpy',
    ]
    key_cols = [c for c in key_cols_candidates if c in df.columns]
    # Remove duplicates
    seen_methods = set()
    unique_cols = []
    for col in key_cols:
        method_name = HERST_METHODS.get(col, (col,))[0]
        if method_name not in seen_methods:
            seen_methods.add(method_name)
            unique_cols.append(col)
    key_cols = unique_cols[:4]

    sample_summary = df.groupby(group_col)[key_cols].mean()

    x = np.arange(len(sample_summary))
    width = 0.2

    for i, col in enumerate(key_cols):
        name, color = HERST_METHODS.get(col, (col, '#888888'))
        ax.bar(x + i * width, sample_summary[col], width,
               label=name, color=color, edgecolor='black', linewidth=0.3)

    ax.set_xticks(x + width * (len(key_cols) - 1) / 2)
    ax.set_xticklabels(sample_summary.index, rotation=45, ha='right', fontsize=5)
    ax.set_ylabel('Cancer Detection AUC', fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper right', fontsize=5)


# =============================================================================
# Combined Figure
# =============================================================================

def create_combined():
    """Create combined Figure 5 with all panels."""
    set_nature_style()

    try:
        df, filename = _load_her2st_data()
    except FileNotFoundError:
        # Can't make this figure without data
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.4))
        ax.text(0.5, 0.5, 'No HER2ST data found', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return fig

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.2, 1.0],
                          wspace=0.5, left=0.06, right=0.96, top=0.88, bottom=0.2)

    ax_a = fig.add_subplot(gs[0])
    _draw_spatial_panel(ax_a)
    add_panel_label(ax_a, 'a', x=-0.12, y=1.05)

    ax_b = fig.add_subplot(gs[1])
    _draw_boxplot_panel(ax_b, df)
    add_panel_label(ax_b, 'b', x=-0.1, y=1.05)

    ax_c = fig.add_subplot(gs[2])
    _draw_persample_panel(ax_c, df)
    add_panel_label(ax_c, 'c', x=-0.1, y=1.05)

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all Figure 5 outputs."""
    print("Generating Figure 5 panels...")

    fig_a = create_panel_a()
    save_figure(fig_a, 'fig5/panel_a')
    plt.close(fig_a)

    try:
        fig_b = create_panel_b()
        save_figure(fig_b, 'fig5/panel_b')
        plt.close(fig_b)

        fig_c = create_panel_c()
        save_figure(fig_c, 'fig5/panel_c')
        plt.close(fig_c)
    except FileNotFoundError:
        print("  Warning: No HER2ST data found for panels b,c")

    try:
        fig_combined = create_combined()
        save_figure(fig_combined, 'fig5/combined')
        plt.close(fig_combined)
    except Exception as e:
        print(f"  Warning: Could not create combined figure: {e}")

    print("Figure 5 complete.")


if __name__ == '__main__':
    main()
