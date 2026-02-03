"""
Figure 4: Independence from Expression-based Methods (Combined)

Caption: Position prediction error is orthogonal to global
expression-based anomaly scores, enabling complementary detection.

Panels:
(a) Correlation matrix — measured pairs only from exp05 (unmeasured = blank)
(b) Scatter: Inv_Pos vs PCA_Error — ILLUSTRATION with real r from exp05
(c) Combined detection performance (from exp02 TwoAxis data)

Data sources:
- exp05_independence_analysis.csv (correlation data)
- exp01_cross-detection_auc.csv (AUC values)
- exp02_competitor_comparison.csv (TwoAxis combined methods)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    set_nature_style, save_figure, add_panel_label,
    COLORS, SINGLE_COL, DOUBLE_COL, RESULTS_DIR
)


def _fisher_z_mean(r_values):
    """Average correlation coefficients using Fisher z-transformation.

    This is the statistically correct way to average Pearson correlations:
    z = arctanh(r), average z-values, then back-transform: r_avg = tanh(z_avg).
    """
    r_values = np.asarray(r_values)
    # Clamp to avoid arctanh(±1) = ±inf
    r_clamped = np.clip(r_values, -0.9999, 0.9999)
    z = np.arctanh(r_clamped)
    return float(np.tanh(z.mean()))


def _load_correlation_data():
    """Load real correlation data from exp05."""
    df = pd.read_csv(RESULTS_DIR / 'exp05_independence_analysis.csv')

    # Available pairwise correlations (averaged across seeds using Fisher z)
    corr_pairs = {
        ('Inv_PosError', 'PCA_Error'): _fisher_z_mean(df['corr_pos_pca_all'].dropna()),
        ('Inv_PosError', 'Neighbor_Diff'): _fisher_z_mean(df['corr_pos_neighbor_all'].dropna()),
        ('PCA_Error', 'Neighbor_Diff'): _fisher_z_mean(df['corr_pca_neighbor_all'].dropna()),
        ('Inv_PosError', 'LISA'): _fisher_z_mean(df['corr_pos_lisa'].dropna()),
        ('Inv_PosError', 'LOF'): _fisher_z_mean(df['corr_pos_lof'].dropna()),
        ('PCA_Error', 'LISA'): _fisher_z_mean(df['corr_pca_lisa'].dropna()),
    }

    # Add additional correlations if available (from updated exp05)
    optional_pairs = {
        ('PCA_Error', 'LOF'): 'corr_pca_lof',
        ('Neighbor_Diff', 'LISA'): 'corr_neighbor_lisa',
        ('Neighbor_Diff', 'LOF'): 'corr_neighbor_lof',
        ('LISA', 'LOF'): 'corr_lisa_lof',
    }
    for pair, col in optional_pairs.items():
        if col in df.columns:
            corr_pairs[pair] = _fisher_z_mean(df[col].dropna())

    # Also compute mean independence metric (Fisher z)
    mean_independence = _fisher_z_mean(df['corr_pos_pca_all'].dropna())

    return corr_pairs, mean_independence


def _build_correlation_matrix(corr_pairs):
    """Build a correlation matrix from available pairwise correlations.

    Only uses measured values. Unmeasured pairs are set to NaN.
    """
    methods = ['Inv_PosError', 'PCA_Error', 'Neighbor_Diff', 'LISA', 'LOF']
    n = len(methods)
    corr = np.full((n, n), np.nan)
    np.fill_diagonal(corr, 1.0)

    measured_mask = np.eye(n, dtype=bool)  # diagonal is always "measured"

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i == j:
                continue
            key = (m1, m2) if (m1, m2) in corr_pairs else (m2, m1)
            if key in corr_pairs:
                corr[i, j] = corr_pairs[key]
                corr[j, i] = corr_pairs[key]
                measured_mask[i, j] = True
                measured_mask[j, i] = True

    return corr, methods, measured_mask


def _load_combined_detection_data():
    """Load TwoAxis combined detection data from exp02."""
    df = pd.read_csv(RESULTS_DIR / 'exp02_competitor_comparison.csv')
    summary = df.groupby('method')[['auc_ectopic', 'auc_intrinsic']].mean()
    return summary


# =============================================================================
# Individual Panel Functions
# =============================================================================

def create_panel_a():
    """Panel (a): Correlation heatmap from real exp05 data (measured pairs only)."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.1))

    corr_pairs, _ = _load_correlation_data()
    corr, methods, measured_mask = _build_correlation_matrix(corr_pairs)

    corr_df = pd.DataFrame(corr, columns=methods, index=methods)
    # Mask upper triangle only
    display_mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)

    # Create annotation array
    annot_array = corr_df.copy().astype(str)
    for i in range(len(methods)):
        for j in range(len(methods)):
            if np.isnan(corr[i, j]) and not display_mask[i, j]:
                annot_array.iloc[i, j] = '-'
            elif not display_mask[i, j]:
                annot_array.iloc[i, j] = f'{corr[i, j]:.2f}'
            else:
                annot_array.iloc[i, j] = ''

    sns.heatmap(corr_df, annot=annot_array, fmt='', cmap='RdBu_r',
                vmin=-1, vmax=1, ax=ax, mask=display_mask,
                cbar_kws={'shrink': 0.6}, annot_kws={'size': 7})

    # Draw gray background for NaN (unmeasured) cells in the lower triangle
    for i in range(len(methods)):
        for j in range(len(methods)):
            if np.isnan(corr[i, j]) and not display_mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                             facecolor='#e0e0e0', edgecolor='white', linewidth=0.5))
                ax.text(j + 0.5, i + 0.5, '-', ha='center', va='center', fontsize=7, color='gray')

    ax.tick_params(axis='both', labelsize=6)

    # Note: correlations averaged using Fisher z-transformation (arctanh)
    ax.text(0.02, 0.02, 'Averaged via Fisher z-transform;\ngray: not measured',
            transform=ax.transAxes, fontsize=4.5, color='gray', va='bottom')

    plt.tight_layout()
    return fig


def _compute_real_scores():
    """Compute per-cell Inv_PosError and PCA_Error from synthetic data.

    Generates synthetic data with ectopic + intrinsic anomalies,
    trains the inverse prediction model, and computes real scores.

    Returns: (s_pos, pca_error, labels, real_r)
        - s_pos: position error scores (model-derived)
        - pca_error: PCA reconstruction error
        - labels: 0=normal, 1=ectopic, 2=intrinsic
        - real_r: measured correlation from exp05 (or computed if unavailable)
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    import torch
    import warnings
    warnings.filterwarnings('ignore')

    # Load real correlation from exp05
    real_r = None
    exp05_file = RESULTS_DIR / 'exp05_independence_analysis.csv'
    if exp05_file.exists():
        exp05_df = pd.read_csv(exp05_file)
        real_r = _fisher_z_mean(exp05_df['corr_pos_pca_all'].dropna())

    try:
        from core.data_generation import generate_synthetic_data
        from core import prepare_data, InversePredictionModel, train_model, compute_scores
        from core.baselines import compute_pca_error

        # Generate synthetic data
        np.random.seed(42)
        torch.manual_seed(42)
        X, coords, labels, ectopic_idx, intrinsic_idx = generate_synthetic_data(
            n_spots=3000, n_genes=500,
            n_ectopic=100, n_intrinsic=300,
            random_state=42,
        )

        # Train inverse prediction model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data = prepare_data(X, coords, k=15, device=device)

        model = InversePredictionModel(
            in_dim=data["n_genes"], hid_dim=64, dropout=0.3,
        ).to(device)

        model = train_model(
            model, data["x_tensor"], data["coords_tensor"],
            data["edge_index"], n_epochs=100, lr=1e-3,
            lambda_pos=0.5, verbose=False,
        )

        # Compute Inv_PosError
        scores = compute_scores(
            model, data["x_tensor"], data["coords_tensor"],
            data["edge_index"], random_state=42,
        )
        s_pos = scores["s_pos"]

        # Compute PCA_Error
        pca_error = compute_pca_error(data["X_norm"], n_components=50)

        # Use exp05 r if available, otherwise compute from this data
        if real_r is None:
            real_r = np.corrcoef(s_pos, pca_error)[0, 1]

        return s_pos, pca_error, labels, real_r

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, real_r


def create_panel_b():
    """Panel (b): Inv_Pos vs PCA scatter — ectopic (red triangles) vs intrinsic (blue squares)."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.1, SINGLE_COL * 0.9))
    _draw_scatter_panel(ax)
    plt.tight_layout()
    return fig


def create_panel_c():
    """Panel (c): Combined detection performance from real data."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.1, SINGLE_COL * 0.9))

    summary = _load_combined_detection_data()

    inv_ect = summary.loc['Inv_PosError', 'auc_ectopic']
    inv_int = summary.loc['Inv_PosError', 'auc_intrinsic']
    pca_ect = summary.loc['PCA_Error', 'auc_ectopic']
    pca_int = summary.loc['PCA_Error', 'auc_intrinsic']

    # TwoAxis_Max combines both
    if 'TwoAxis_Max' in summary.index:
        comb_ect = summary.loc['TwoAxis_Max', 'auc_ectopic']
        comb_int = summary.loc['TwoAxis_Max', 'auc_intrinsic']
    else:
        import warnings
        warnings.warn(
            "TwoAxis_Max not found in exp02 data. Using element-wise max of "
            "individual method means as an approximation.",
            UserWarning,
        )
        comb_ect = max(inv_ect, pca_ect)
        comb_int = max(inv_int, pca_int)

    methods = ['Inv_Pos\nonly', 'PCA\nonly', 'Combined\n(max)']
    ectopic_auc = [inv_ect, pca_ect, comb_ect]
    intrinsic_auc = [inv_int, pca_int, comb_int]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ectopic_auc, width,
                   label='Ectopic', color=COLORS['ectopic'],
                   edgecolor='black', linewidth=0.3)
    bars2 = ax.bar(x + width / 2, intrinsic_auc, width,
                   label='Intrinsic', color=COLORS['intrinsic'],
                   edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylabel('Detection AUC', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper left', fontsize=6)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=6)

    plt.tight_layout()
    return fig


# =============================================================================
# Panel Drawing Functions (for combined figure)
# =============================================================================

def _draw_correlation_panel(ax):
    """Draw panel (a) content on given axes — measured pairs only."""
    corr_pairs, _ = _load_correlation_data()
    corr, methods, measured_mask = _build_correlation_matrix(corr_pairs)

    corr_df = pd.DataFrame(corr, columns=methods, index=methods)
    # Mask upper triangle only
    display_mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)

    # Create annotation array
    annot_array = corr_df.copy().astype(str)
    for i in range(len(methods)):
        for j in range(len(methods)):
            if np.isnan(corr[i, j]) and not display_mask[i, j]:
                annot_array.iloc[i, j] = '-'
            elif not display_mask[i, j]:
                annot_array.iloc[i, j] = f'{corr[i, j]:.2f}'
            else:
                annot_array.iloc[i, j] = ''

    sns.heatmap(corr_df, annot=annot_array, fmt='', cmap='RdBu_r',
                vmin=-1, vmax=1, ax=ax, mask=display_mask,
                cbar_kws={'shrink': 0.6}, annot_kws={'size': 5})

    # Draw gray background for NaN (unmeasured) cells in the lower triangle
    for i in range(len(methods)):
        for j in range(len(methods)):
            if np.isnan(corr[i, j]) and not display_mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                             facecolor='#e0e0e0', edgecolor='white', linewidth=0.5))
                ax.text(j + 0.5, i + 0.5, '-', ha='center', va='center', fontsize=5, color='gray')

    ax.tick_params(axis='both', labelsize=5)


def _draw_scatter_panel(ax):
    """Draw panel (b): Inv_PosError vs PCA_Error scatter.

    Ectopic anomalies (red triangles) cluster bottom-right (high pos error, low PCA error).
    Intrinsic anomalies (blue squares) cluster top-left (low pos error, high PCA error).
    """
    s_pos, pca_error, labels, real_r = _compute_real_scores()
    if s_pos is not None:
        normal_mask = labels == 0
        ectopic_mask = labels == 1
        intrinsic_mask = labels == 2

        ax.scatter(s_pos[normal_mask], pca_error[normal_mask],
                   c=COLORS['normal'], marker='o', alpha=0.3, s=6,
                   label='Normal', zorder=1)
        ax.scatter(s_pos[ectopic_mask], pca_error[ectopic_mask],
                   c=COLORS['ectopic'], marker='^', alpha=0.7, s=25,
                   edgecolors='black', linewidths=0.3,
                   label='Ectopic', zorder=3)
        ax.scatter(s_pos[intrinsic_mask], pca_error[intrinsic_mask],
                   c=COLORS['intrinsic'], marker='s', alpha=0.7, s=20,
                   edgecolors='black', linewidths=0.3,
                   label='Intrinsic', zorder=2)

        ax.text(0.05, 0.95, f'r = {real_r:.3f}', transform=ax.transAxes,
                fontsize=6, va='top', fontweight='bold')
    else:
        if real_r is not None:
            ax.text(0.05, 0.95, f'r = {real_r:.3f}', transform=ax.transAxes,
                    fontsize=6, va='top', fontweight='bold')
        ax.text(0.5, 0.5, 'Score computation failed', transform=ax.transAxes,
                ha='center', va='center', fontsize=7, color='gray')

    ax.set_xlabel('Inv_PosError', fontsize=7)
    ax.set_ylabel('PCA_Error', fontsize=7)
    ax.legend(fontsize=5, loc='upper right')


def _draw_combined_panel(ax):
    """Draw panel (c) content on given axes."""
    summary = _load_combined_detection_data()

    inv_ect = summary.loc['Inv_PosError', 'auc_ectopic']
    inv_int = summary.loc['Inv_PosError', 'auc_intrinsic']
    pca_ect = summary.loc['PCA_Error', 'auc_ectopic']
    pca_int = summary.loc['PCA_Error', 'auc_intrinsic']

    if 'TwoAxis_Max' in summary.index:
        comb_ect = summary.loc['TwoAxis_Max', 'auc_ectopic']
        comb_int = summary.loc['TwoAxis_Max', 'auc_intrinsic']
    else:
        import warnings
        warnings.warn(
            "TwoAxis_Max not found in exp02 data (combined panel).",
            UserWarning,
        )
        comb_ect = max(inv_ect, pca_ect)
        comb_int = max(inv_int, pca_int)

    methods = ['Inv_Pos\nonly', 'PCA\nonly', 'Combined\n(max)']
    ectopic_auc = [inv_ect, pca_ect, comb_ect]
    intrinsic_auc = [inv_int, pca_int, comb_int]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ectopic_auc, width,
                   label='Ectopic', color=COLORS['ectopic'],
                   edgecolor='black', linewidth=0.3)
    bars2 = ax.bar(x + width / 2, intrinsic_auc, width,
                   label='Intrinsic', color=COLORS['intrinsic'],
                   edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=6)
    ax.set_ylabel('Detection AUC', fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.legend(loc='upper left', fontsize=5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2), textcoords="offset points",
                        ha='center', va='bottom', fontsize=5)


# =============================================================================
# Combined Figure
# =============================================================================

def create_combined():
    """Create combined Figure 4 with all panels."""
    set_nature_style()

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1],
                          wspace=0.5, left=0.08, right=0.95, top=0.88, bottom=0.12)

    ax_a = fig.add_subplot(gs[0])
    _draw_correlation_panel(ax_a)
    add_panel_label(ax_a, 'a', x=-0.18, y=1.05)

    ax_b = fig.add_subplot(gs[1])
    _draw_scatter_panel(ax_b)
    add_panel_label(ax_b, 'b', x=-0.12, y=1.05)

    ax_c = fig.add_subplot(gs[2])
    _draw_combined_panel(ax_c)
    add_panel_label(ax_c, 'c', x=-0.12, y=1.05)

    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all Figure 4 outputs."""
    print("Generating Figure 4 panels...")

    fig_a = create_panel_a()
    save_figure(fig_a, 'fig4/panel_a')
    plt.close(fig_a)

    fig_b = create_panel_b()
    save_figure(fig_b, 'fig4/panel_b')
    plt.close(fig_b)

    fig_c = create_panel_c()
    save_figure(fig_c, 'fig4/panel_c')
    plt.close(fig_c)

    fig_combined = create_combined()
    save_figure(fig_combined, 'fig4/combined')
    plt.close(fig_combined)

    print("Figure 4 complete.")


if __name__ == '__main__':
    main()
