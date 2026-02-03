"""
CASTOR Inverse Prediction Core Module

This module provides the core functionality for inverse spatial prediction
anomaly detection in spatial transcriptomics data.

Key Components:
- models: InversePredictionModel for ectopic detection
- baselines: Competing methods (LISA, NeighborDiff, LOF, IF, PCA)
- evaluation: AUC metrics, statistical tests
- preprocessing: Unified data preprocessing
- data_generation: Synthetic data with controlled anomalies
- utils: Shared utilities (graph building, normalization)
"""

from .utils import (
    build_spatial_graph,
    build_delaunay_graph,
    build_radius_graph,
    aggregate_neighbors,
    robust_zscore,
    normalize_expression,
    normalize_coordinates,
    set_seed,
)
from .preprocessing import prepare_data
from .models import InversePredictionModel, train_model, compute_scores
from .baselines import (
    compute_neighbor_diff,
    compute_lisa,
    compute_local_spatial_deviation,
    compute_spotsweeper,
    compute_pca_error,
    compute_lof,
    compute_isolation_forest,
    compute_all_baselines,
)
from .evaluation import (
    compute_auc_metrics,
    compute_separation_auc,
    compute_position_accuracy,
    compute_correlation_matrix,
    statistical_tests,
    apply_fdr_correction,
)
from .data_generation import (
    generate_synthetic_data,
    generate_controlled_ectopic,
    inject_clustered_ectopic,
)
from .scenarios import (
    SCENARIOS,
    generate_scenario_data,
    ScenarioConfig,
    EctopicType,
    IntrinsicType,
    SpatialStructure,
)
from .real_data import (
    download_visium_dataset,
    inject_ectopic_by_region,
    inject_ectopic_by_distance,
    cluster_spots,
    filter_genes,
    filter_spots,
    subsample_genes,
    VISIUM_DATASETS,
)

__all__ = [
    # Utils
    "build_spatial_graph",
    "build_delaunay_graph",
    "build_radius_graph",
    "aggregate_neighbors",
    "robust_zscore",
    "normalize_expression",
    "normalize_coordinates",
    "set_seed",
    # Preprocessing
    "prepare_data",
    # Models
    "InversePredictionModel",
    "train_model",
    "compute_scores",
    # Baselines
    "compute_neighbor_diff",
    "compute_lisa",
    "compute_local_spatial_deviation",
    "compute_spotsweeper",
    "compute_pca_error",
    "compute_lof",
    "compute_isolation_forest",
    "compute_all_baselines",
    # Evaluation
    "compute_auc_metrics",
    "compute_separation_auc",
    "compute_position_accuracy",
    "compute_correlation_matrix",
    "statistical_tests",
    "apply_fdr_correction",
    # Data generation
    "generate_synthetic_data",
    "generate_controlled_ectopic",
    "inject_clustered_ectopic",
    # Scenarios
    "SCENARIOS",
    "generate_scenario_data",
    "ScenarioConfig",
    "EctopicType",
    "IntrinsicType",
    "SpatialStructure",
    # Real data
    "download_visium_dataset",
    "inject_ectopic_by_region",
    "inject_ectopic_by_distance",
    "cluster_spots",
    "filter_genes",
    "filter_spots",
    "subsample_genes",
    "VISIUM_DATASETS",
]
