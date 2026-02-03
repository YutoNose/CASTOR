"""
Figure 9: Region Transplantation on HER2ST

Demonstrates that CASTOR's inverse prediction maintains detection
performance when ectopic spots form contiguous regions, while
neighborhood-based methods (LISA, Neighbor_Diff) degrade.

Panels:
(a) H&E tissue image with transplanted contiguous region highlighted
(b) Detection AUC vs Region Size (line plot, mean +/- SD shaded bands)
(c) Detection AUC at region_size=30 (horizontal bar chart, 95% bootstrap CI)
(d) delta-AUC (Inv_PosError - LISA) vs Region Size with significance (95% CI)

Data source: results/exp20_region_transplantation.csv
Tissue images: HER2ST H&E stained sections (Andersson et al. 2021)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
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

_default_her2st = Path(__file__).resolve().parent.parent.parent.parent.parent / "her2st"
HER2ST_DIR = Path(os.environ.get("HER2ST_DIR", str(_default_her2st)))
IMG_DIR = HER2ST_DIR / "data" / "ST-imgs"

# Methods to show in panel b (line plot)
LINE_METHODS = {
    'auc_inv_pos': ('Inv_PosError', COLORS['inv_pos']),
    'auc_lisa': ('LISA', COLORS['lisa']),
    'auc_neighbor_diff': ('Neighbor_Diff', COLORS['neighbor_diff']),
    'auc_pca_error': ('PCA_Error', COLORS['pca_error']),
    'auc_lof': ('LOF', COLORS['lof']),
}

# All methods for bar chart (panel c)
BAR_METHODS = {
    'auc_inv_pos': ('Inv_PosError', COLORS['inv_pos']),
    'auc_inv_if': ('Inv_IF', COLORS['isolation_forest']),
    'auc_inv_neighbor': ('Inv_Neighbor', COLORS['neighbor_diff']),
    'auc_neighbor_diff': ('Neighbor_Diff', '#CC79A7'),
    'auc_isolation_forest': ('Isolation_Forest', '#E69F00'),
    'auc_inv_self': ('Inv_Self', '#56B4E9'),
    'auc_lof': ('LOF', COLORS['lof']),
    'auc_lisa': ('LISA', COLORS['lisa']),
    'auc_pca_error': ('PCA_Error', COLORS['pca_error']),
}

REGION_SIZES = [1, 3, 5, 10, 15, 30]


def _load_results():
    """Load experiment 20 results."""
    path = RESULTS_DIR / "exp20_region_transplantation.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Run exp20_region_transplantation.py first: {path}"
        )
    return pd.read_csv(path)


def _load_tissue_image(sample_id):
    """Load H&E tissue image for a sample."""
    letter = sample_id[0]
    img_dir = IMG_DIR / letter / sample_id
    if img_dir.exists():
        for f in img_dir.glob("*.jpg"):
            return np.array(Image.open(f))
    return None


def _draw_tissue_panel(ax, df):
    """
    Panel (a): H&E tissue image with transplant region overlay.

    Shows a representative sample with tumor/normal spots annotated
    and the transplanted contiguous region highlighted.
    """
    if df.empty or "sample_id" not in df.columns:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return

    # Pick the first sample in results for illustration
    sample_id = df["sample_id"].iloc[0]

    try:
        from data.generators.her2st import HER2STDataLoader
        loader = HER2STDataLoader(str(HER2ST_DIR))
        X_sparse, coords, y_true, meta = loader.load(sample_id)
        pixel_coords = meta.get("pixel_coords")
    except Exception:
        ax.text(0.5, 0.5, f'HER2ST data not available\n({sample_id})',
                ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return

    # Load tissue image
    img = _load_tissue_image(sample_id)

    # Use y_true from loader (already aligned to coords)
    # y_true: 1 = cancer, 0 = not cancer
    y_arr = np.asarray(y_true).ravel()
    is_tumor = y_arr == 1
    is_normal = y_arr == 0

    # Run one transplant to get donor/recipient locations for visualization
    transplant_mask = np.zeros(len(coords), dtype=bool)
    donor_indices = np.array([], dtype=int)
    recipient_indices = np.array([], dtype=int)
    try:
        from exp20_region_transplantation import transplant_region_to_normal
        from scipy import sparse
        X_raw = X_sparse.toarray() if sparse.issparse(X_sparse) else np.asarray(X_sparse)

        # Build labels_series from y_true (already aligned to coords)
        # transplant_region_to_normal uses str.contains() matching, so
        # "invasive cancer" / "connective tissue" satisfy its defaults.
        labels_series = pd.Series(
            np.where(y_arr == 1, "invasive cancer", "connective tissue"),
        )

        result = transplant_region_to_normal(
            X_raw, coords, labels_series,
            region_size=30, n_total_transplant=30, random_state=42,
        )
        transplant_mask = result["transplant_mask"]
        donor_indices = result.get("donor_indices", np.array([], dtype=int))
        recipient_indices = result.get("recipient_indices", np.array([], dtype=int))
    except Exception as e:
        print(f"  Warning: transplant visualization skipped: {e}")
        donor_indices = np.array([], dtype=int)
        recipient_indices = np.array([], dtype=int)

    use_pixel = img is not None and pixel_coords is not None
    plot_coords = pixel_coords if use_pixel else coords

    if use_pixel:
        ax.imshow(img, aspect='equal')
        pad = 500
        ax.set_xlim(plot_coords[:, 0].min() - pad, plot_coords[:, 0].max() + pad)
        ax.set_ylim(plot_coords[:, 1].max() + pad, plot_coords[:, 1].min() - pad)

    n = len(coords)

    # Draw spots by category
    other_mask = np.ones(n, dtype=bool)
    if len(is_tumor) == n:
        # Tumor spots (faded background)
        if is_tumor.any():
            ax.scatter(plot_coords[is_tumor, 0], plot_coords[is_tumor, 1],
                       s=8, c=COLORS['ectopic'], alpha=0.3, edgecolors='none',
                       label='Tumor')
            other_mask[is_tumor] = False
        # Normal spots (faded background)
        if is_normal.any():
            ax.scatter(plot_coords[is_normal, 0], plot_coords[is_normal, 1],
                       s=8, c=COLORS['inv_pos'], alpha=0.3, edgecolors='none',
                       label='Normal')
            other_mask[is_normal] = False

    # Remaining spots
    if other_mask.any():
        ax.scatter(plot_coords[other_mask, 0], plot_coords[other_mask, 1],
                   s=5, c='gray', alpha=0.2, edgecolors='none')

    # Highlight donor and recipient regions to show transplantation process
    if len(donor_indices) > 0 and len(recipient_indices) > 0:
        # Donor region (source of expression)
        donor_mask = np.zeros(n, dtype=bool)
        unique_donors = np.unique(donor_indices)
        donor_mask[unique_donors] = True
        ax.scatter(plot_coords[donor_mask, 0], plot_coords[donor_mask, 1],
                   s=25, marker='s', c=COLORS['ectopic'], edgecolors='black',
                   linewidths=0.5, zorder=8, alpha=0.9,
                   label=f'Donor region (n={len(unique_donors)})')

        # Recipient region (transplanted spots)
        ax.scatter(plot_coords[transplant_mask, 0], plot_coords[transplant_mask, 1],
                   s=25, marker='*', c='#FFD700', edgecolors='black',
                   linewidths=0.3, zorder=9, alpha=0.9,
                   label=f'Recipient region (n={transplant_mask.sum()})')

        # Arrow from donor center to recipient center
        if transplant_mask.sum() > 0:
            donor_center = plot_coords[unique_donors].mean(axis=0)
            recipient_center = plot_coords[transplant_mask].mean(axis=0)
            if not (np.isnan(donor_center).any() or np.isnan(recipient_center).any()):
                ax.annotate(
                    '', xy=recipient_center, xytext=donor_center,
                    arrowprops=dict(
                        arrowstyle='->', color='black', lw=1.5,
                        connectionstyle='arc3,rad=0.2',
                    ),
                    zorder=11,
                )
                # Label the arrow
                mid = (donor_center + recipient_center) / 2
                ax.text(mid[0], mid[1], 'expression\ntransplant',
                        fontsize=5, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='gray', alpha=0.9),
                        zorder=12)

    elif transplant_mask.any():
        # Fallback: only show recipient if donor info unavailable
        n_transplant = transplant_mask.sum()
        ax.scatter(plot_coords[transplant_mask, 0], plot_coords[transplant_mask, 1],
                   s=30, marker='*', c='#FFD700', edgecolors='black',
                   linewidths=0.3, zorder=10,
                   label=f'Transplanted (n={n_transplant})')

    ax.legend(fontsize=5, loc='lower left', framealpha=0.8, markerscale=1.5)
    ax.axis('off')
    if not use_pixel:
        ax.set_aspect('equal')


def _draw_auc_vs_region_size(ax, df):
    """
    Panel (b): Detection AUC vs Region Size (line plot).

    Lines show mean AUC across seeds and samples.
    Shaded bands show +/- 1 SD.
    """
    for col, (name, color) in LINE_METHODS.items():
        if col not in df.columns:
            continue
        means = []
        stds = []
        valid_sizes = []
        for rs in REGION_SIZES:
            sub = df[df["region_size"] == rs][col].dropna()
            if len(sub) > 0:
                means.append(sub.mean())
                stds.append(sub.std(ddof=1))
                valid_sizes.append(rs)
        if not valid_sizes:
            continue
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(valid_sizes, means, 'o-', color=color, label=name,
                markersize=4, linewidth=1.5)
        ax.fill_between(valid_sizes, means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.set_xlabel('Region Size', fontsize=7)
    ax.set_ylabel('Transplant Detection AUC', fontsize=7)
    ax.set_xscale('log')
    ax.set_xticks(REGION_SIZES)
    ax.set_xticklabels([str(s) for s in REGION_SIZES])
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=5, loc='lower left')

    # Annotation: bands = mean +/- 1 SD
    ax.text(0.98, 0.02, 'Bands: mean $\\pm$ 1 SD',
            transform=ax.transAxes, fontsize=5, ha='right', va='bottom',
            color='gray',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))


def _bootstrap_ci(values, confidence=0.95, n_bootstrap=10000, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    values = values.dropna().values if hasattr(values, 'dropna') else np.asarray(values)
    n = len(values)
    if n < 2:
        return values.mean() if n == 1 else np.nan, 0.0, 0.0
    boot_means = np.array([rng.choice(values, size=n, replace=True).mean()
                           for _ in range(n_bootstrap)])
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    mean_val = values.mean()
    return mean_val, mean_val - ci_lower, ci_upper - mean_val


def _draw_bar_at_max_region(ax, df, region_size=30):
    """
    Panel (c): Detection AUC at a fixed region size (bar chart).

    Shows mean AUC with 95% bootstrap CI error bars.
    """
    sub = df[df["region_size"] == region_size]
    if len(sub) == 0:
        ax.text(0.5, 0.5, f'No data for region_size={region_size}',
                ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return

    method_data = []
    for col, (name, color) in BAR_METHODS.items():
        if col not in sub.columns:
            continue
        vals = sub[col].dropna()
        if len(vals) == 0:
            continue
        mean_val, ci_lo, ci_hi = _bootstrap_ci(vals)
        method_data.append({
            'name': name, 'color': color,
            'mean': mean_val, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
        })

    method_data.sort(key=lambda x: x['mean'], reverse=True)

    y_pos = np.arange(len(method_data))
    means = [m['mean'] for m in method_data]
    ci_lo = [m['ci_lo'] for m in method_data]
    ci_hi = [m['ci_hi'] for m in method_data]
    colors = [m['color'] for m in method_data]
    names = [m['name'] for m in method_data]

    ax.barh(y_pos, means, xerr=[ci_lo, ci_hi], color=colors,
            edgecolor='black', linewidth=0.3, height=0.7,
            capsize=3, error_kw={'linewidth': 0.8})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel(f'Detection AUC (region\\_size={region_size})', fontsize=7)
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)

    n_obs = len(sub)
    for i, m in enumerate(means):
        ax.text(m + ci_hi[i] + 0.02, i, f'{m:.3f}', va='center', fontsize=5)

    ax.text(0.98, 0.02, f'Error bars: 95% CI (bootstrap, n={n_obs})',
            transform=ax.transAxes, fontsize=5, ha='right', va='bottom',
            color='gray',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))


def _draw_delta_auc(ax, df):
    """
    Panel (d): delta-AUC (Inv_PosError - LISA) vs Region Size.

    For each region size, compute the paired difference per (sample, seed)
    and test if mean != 0 via Wilcoxon signed-rank test (two-sided).
    Error bars show 95% bootstrap CI.
    """
    delta_means = []
    delta_ci_lo = []
    delta_ci_hi = []
    pvals = []
    valid_sizes = []

    for rs in REGION_SIZES:
        sub = df[df["region_size"] == rs]
        if len(sub) < 2:
            continue
        if "auc_inv_pos" not in sub.columns or "auc_lisa" not in sub.columns:
            continue

        delta = sub["auc_inv_pos"].values - sub["auc_lisa"].values
        delta = delta[~np.isnan(delta)]
        if len(delta) < 2:
            continue

        if delta.std() == 0:
            # All values identical; skip (no variance to test)
            continue

        valid_sizes.append(rs)
        mean_val, ci_lo, ci_hi = _bootstrap_ci(delta)
        delta_means.append(mean_val)
        delta_ci_lo.append(ci_lo)
        delta_ci_hi.append(ci_hi)
        # Wilcoxon signed-rank test (non-parametric, appropriate for bounded AUC data)
        _, p = stats.wilcoxon(delta, alternative='two-sided')
        pvals.append(p)

    if not valid_sizes:
        ax.text(0.5, 0.5, 'Insufficient data',
                ha='center', va='center', transform=ax.transAxes, fontsize=7)
        return

    delta_means = np.array(delta_means)
    delta_ci_lo = np.array(delta_ci_lo)
    delta_ci_hi = np.array(delta_ci_hi)
    pvals = np.array(pvals)

    # Bonferroni correction for the number of comparisons actually performed
    n_tests = len(valid_sizes)
    pvals_corrected = np.minimum(pvals * n_tests, 1.0)

    x_pos = np.arange(len(valid_sizes))
    ax.bar(x_pos, delta_means, yerr=[delta_ci_lo, delta_ci_hi],
           color=COLORS['inv_pos'], edgecolor='black', linewidth=0.3,
           capsize=4, error_kw={'linewidth': 0.8}, width=0.6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in valid_sizes])
    ax.set_xlabel('Region Size', fontsize=7)
    ax.set_ylabel('$\\Delta$AUC (Inv\\_Pos $-$ LISA)', fontsize=7)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)

    # Significance markers (Bonferroni-corrected p-values)
    for i, p in enumerate(pvals_corrected):
        if p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            marker = 'n.s.'
        y = delta_means[i] + delta_ci_hi[i] + 0.02
        ax.text(i, y, marker, ha='center', va='bottom', fontsize=6)

    ax.text(0.98, 0.98,
            f'Wilcoxon signed-rank (two-sided)\nBonferroni corrected ($n={n_tests}$)'
            '\nError bars: 95% CI (bootstrap)',
            transform=ax.transAxes, fontsize=5, ha='right', va='top',
            color='gray',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))


# =============================================================================
# Individual Panel Functions
# =============================================================================

def create_panel_a():
    """Panel (a): Tissue image with transplant overlay - standalone."""
    set_nature_style()
    df = _load_results()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.3))
    _draw_tissue_panel(ax, df)
    plt.tight_layout()
    return fig


def create_panel_b():
    """Panel (b): AUC vs Region Size - standalone."""
    set_nature_style()
    df = _load_results()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, SINGLE_COL * 1.0))
    _draw_auc_vs_region_size(ax, df)
    plt.tight_layout()
    return fig


def create_panel_c(region_size=30):
    """Panel (c): Bar chart at given region_size - standalone."""
    set_nature_style()
    df = _load_results()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.0))
    _draw_bar_at_max_region(ax, df, region_size=region_size)
    plt.tight_layout()
    return fig


def create_panel_d():
    """Panel (d): delta-AUC vs Region Size - standalone."""
    set_nature_style()
    df = _load_results()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.0))
    _draw_delta_auc(ax, df)
    plt.tight_layout()
    return fig


# =============================================================================
# Combined Figure
# =============================================================================

def create_combined():
    """Create combined Figure 9 with all panels."""
    set_nature_style()

    df = _load_results()

    print(f"  Samples: {df['sample_id'].unique().tolist()}")
    print(f"  Region sizes: {sorted(df['region_size'].unique().tolist())}")
    print(f"  Total rows: {len(df)}")

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.65))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.45,
                          left=0.08, right=0.96, top=0.95, bottom=0.08)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_tissue_panel(ax_a, df)
    add_panel_label(ax_a, 'a', x=-0.02, y=1.05)

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_auc_vs_region_size(ax_b, df)
    add_panel_label(ax_b, 'b', x=-0.15, y=1.05)

    ax_c = fig.add_subplot(gs[1, 0])
    _draw_bar_at_max_region(ax_c, df)
    add_panel_label(ax_c, 'c', x=-0.2, y=1.05)

    ax_d = fig.add_subplot(gs[1, 1])
    _draw_delta_auc(ax_d, df)
    add_panel_label(ax_d, 'd', x=-0.15, y=1.05)

    return fig


def main():
    """Generate all Figure 9 outputs."""
    print("Generating Figure 9 panels...")

    try:
        df = _load_results()
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        print("  Skipping Figure 9 (requires exp20_region_transplantation.csv)")
        return

    # Print summary
    print(f"\n--- Summary ---")
    for rs in REGION_SIZES:
        sub = df[df["region_size"] == rs]
        if len(sub) == 0:
            continue
        inv = sub["auc_inv_pos"].mean()
        lisa = sub["auc_lisa"].mean()
        print(f"  region_size={rs:2d}: Inv_Pos={inv:.3f}, LISA={lisa:.3f}, "
              f"delta={inv - lisa:.3f} (n={len(sub)})")

    fig_a = create_panel_a()
    save_figure(fig_a, 'fig9/panel_a')
    plt.close(fig_a)

    fig_b = create_panel_b()
    save_figure(fig_b, 'fig9/panel_b')
    plt.close(fig_b)

    fig_c = create_panel_c()
    save_figure(fig_c, 'fig9/panel_c')
    plt.close(fig_c)

    fig_d = create_panel_d()
    save_figure(fig_d, 'fig9/panel_d')
    plt.close(fig_d)

    fig_combined = create_combined()
    save_figure(fig_combined, 'fig9/combined')
    plt.close(fig_combined)

    print("\nFigure 9 complete.")


if __name__ == '__main__':
    main()
