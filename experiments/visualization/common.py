"""
Common utilities for figure generation.

Style guidelines:
- Nature Methods figure widths: 88mm (single), 180mm (double)
- Font: Arial or Helvetica, 6-8pt
- Line width: 0.5-1pt
- Color palette: colorblind-friendly
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Figure output directory
FIGURE_DIR = Path(__file__).parent.parent.parent / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Nature Methods style settings
def set_nature_style():
    """Set matplotlib style for Nature Methods figures."""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,

        # Lines
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.3,
        'ytick.minor.width': 0.3,

        # Figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Other
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# Colorblind-friendly palette
COLORS = {
    'inv_pos': '#0072B2',      # Blue
    'pca_error': '#D55E00',    # Orange
    'lisa': '#009E73',         # Green
    'neighbor_diff': '#CC79A7', # Pink
    'spotsweeper': '#F0E442',  # Yellow
    'lof': '#56B4E9',          # Light blue
    'isolation_forest': '#E69F00', # Light orange
    'stagate': '#999999',      # Gray
    'graphst': '#666666',      # Dark gray
    'squidpy': '#882255',      # Wine
    'stlearn': '#44AA99',      # Teal

    # Anomaly types
    'ectopic': '#E74C3C',      # Red
    'intrinsic': '#3498DB',    # Blue
    'normal': '#95A5A6',       # Gray
}

# Method display names
METHOD_NAMES = {
    'inv_pos': 'Inv_PosError',
    'pca_error': 'PCA_Error',
    'lisa': 'LISA',
    'neighbor_diff': 'Neighbor_Diff',
    'spotsweeper': 'SpotSweeper',
    'lof': 'LOF',
    'isolation_forest': 'Isolation_Forest',
    'stagate': 'STAGATE+IF',
    'graphst': 'GraphST+IF',
    'squidpy': 'Squidpy',
    'stlearn': 'STLearn',
}

# Figure sizes (in inches, Nature Methods uses mm)
MM_TO_INCH = 0.0393701
SINGLE_COL = 88 * MM_TO_INCH  # ~3.46 inches
DOUBLE_COL = 180 * MM_TO_INCH  # ~7.09 inches

def save_figure(fig, name, formats=['pdf', 'png']):
    """Save figure in multiple formats.

    If name contains '/', creates subdirectories.
    Example: save_figure(fig, 'fig1/panel_a') -> figures/fig1/panel_a.pdf
    """
    # Handle subdirectory structure
    if '/' in name:
        subdir = FIGURE_DIR / name.rsplit('/', 1)[0]
        subdir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = FIGURE_DIR / f"{name}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, format=fmt, dpi=300 if fmt == 'png' else None)
        print(f"Saved: {path}")

def add_panel_label(ax, label, x=-0.15, y=1.05):
    """Add panel label (a, b, c, etc.) to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

def load_results(filename):
    """Load results from CSV file."""
    path = RESULTS_DIR / filename
    if not path.exists():
        # Try benchmark_quick subdirectory
        path = RESULTS_DIR / "benchmark_quick" / filename
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {filename}")
    return pd.read_csv(path)

def compute_summary_stats(df, groupby_col, value_cols):
    """Compute mean and std for grouped data."""
    summary = df.groupby(groupby_col)[value_cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()
