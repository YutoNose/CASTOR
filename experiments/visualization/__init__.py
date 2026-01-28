"""
Visualization scripts for Inverse Position Prediction paper.

Main Figures:
- fig1_concept: Concept and method overview
- fig2_selectivity: Selective detection of ectopic anomalies
- fig3_robustness: Robustness to hard ectopic scenarios
- fig4_independence: Independence from expression-based methods
- fig5_her2st: Real data validation (HER2ST)

Supplementary Figures:
- figS1_noise: Noise robustness analysis
- figS2_ablation: Ablation study
- figS4_statistics: Statistical analysis
- figS5_scalability: Scalability analysis

Usage:
    python generate_all_figures.py          # All figures
    python generate_all_figures.py --main   # Main figures only
    python generate_all_figures.py --supp   # Supplementary only
"""

from .common import (
    set_nature_style,
    save_figure,
    add_panel_label,
    load_results,
    COLORS,
    METHOD_NAMES,
    SINGLE_COL,
    DOUBLE_COL,
    FIGURE_DIR,
    RESULTS_DIR,
)
