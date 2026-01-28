"""Model layer: InversePredictionModel and training utilities."""

from castor.model.graph import aggregate_neighbors, build_spatial_graph
from castor.model.inverse_prediction import InversePredictionModel
from castor.model.training import compute_scores, train_model

__all__ = [
    "InversePredictionModel",
    "train_model",
    "compute_scores",
    "build_spatial_graph",
    "aggregate_neighbors",
]
