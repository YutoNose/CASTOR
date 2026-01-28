"""
Experiment modules for inverse prediction validation.

Each experiment module follows the same interface:
    run(config: ExperimentConfig, verbose: bool = True) -> pd.DataFrame

Experiments:
- exp01_cross_detection: Cross-detection AUC validation
- exp02_competitor: Comparison with competitor methods
- exp03_position_accuracy: Position prediction accuracy for ectopic
- exp04_noise_robustness: Robustness to noise and dropout
- exp05_independence: Independence analysis between scores
- exp07_ablation: Ablation studies
- exp08_clean_training: Clean training (BUGGY - do not use)
- exp09_clean_training_fixed: Clean training with proper experimental design
- exp10_multi_scenario: Multi-scenario validation (9 scenarios)
- exp11_real_data: Real data validation (Visium)
- exp12_embedding_comparison: Extended embedding comparison (optional)
- exp13_scalability: Scalability analysis (optional)
"""

from . import (
    exp01_cross_detection,
    exp02_competitor,
    exp03_position_accuracy,
    exp04_noise_robustness,
    exp05_independence,
    exp07_ablation,
    exp08_clean_training,
    exp09_clean_training_fixed,
    exp10_multi_scenario,
    exp11_real_data,
)

__all__ = [
    "exp01_cross_detection",
    "exp02_competitor",
    "exp03_position_accuracy",
    "exp04_noise_robustness",
    "exp05_independence",
    "exp07_ablation",
    "exp08_clean_training",
    "exp09_clean_training_fixed",
    "exp10_multi_scenario",
    "exp11_real_data",
]

# Optional experiments (may not exist yet)
try:
    from . import exp12_embedding_comparison
    __all__.append("exp12_embedding_comparison")
except ImportError:
    pass

try:
    from . import exp13_scalability
    __all__.append("exp13_scalability")
except ImportError:
    pass

try:
    from . import exp14_her2st_validation
    __all__.append("exp14_her2st_validation")
except ImportError:
    pass

try:
    from . import exp15_full_benchmark
    __all__.append("exp15_full_benchmark")
except ImportError:
    pass

try:
    from . import exp16_real_data_gene_analysis
    __all__.append("exp16_real_data_gene_analysis")
except ImportError:
    pass

try:
    from . import exp17_her2st_transplantation
    __all__.append("exp17_her2st_transplantation")
except ImportError:
    pass

try:
    from . import exp18_interpretability
    __all__.append("exp18_interpretability")
except ImportError:
    pass
