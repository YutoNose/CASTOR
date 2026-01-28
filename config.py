"""
Global configuration for inverse prediction experiments.

This module centralizes all hyperparameters and settings to ensure
consistency across experiments.
"""

from dataclasses import dataclass, field
from typing import List
import torch


def _get_default_device() -> str:
    """Get default device, checking if CUDA is actually usable."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return "cpu"
    try:
        # Test if CUDA actually works
        torch.cuda.synchronize()
        _ = torch.tensor([1.0], device='cuda')
        torch.cuda.synchronize()
        return "cuda"
    except (AssertionError, RuntimeError):
        return "cpu"


@dataclass
class ExperimentConfig:
    """Configuration for all experiments."""

    # Random seeds for reproducibility (n=30 for statistical rigor)
    seeds: List[int] = field(default_factory=lambda: list(range(42, 72)))

    # Data generation
    n_spots: int = 3000
    n_genes: int = 500
    n_ectopic: int = 100
    n_intrinsic: int = 300
    n_modules: int = 20
    min_distance_factor: float = 0.5

    # Model architecture
    hidden_dim: int = 64
    dropout: float = 0.3

    # Training
    n_epochs: int = 100
    learning_rate: float = 1e-3
    lambda_pos: float = 0.5
    lambda_self: float = 1.0

    # Spatial graph
    k_neighbors: int = 15

    # Evaluation
    test_size: float = 0.3
    confidence_level: float = 0.95

    # Device - uses robust check to verify CUDA actually works
    device: str = field(default_factory=_get_default_device)

    # Output
    results_dir: str = "results"
    figures_dir: str = "figures"


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""

    # Lambda position values to test
    lambda_pos_values: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.5, 1.0, 2.0]
    )

    # Hidden dimension values to test
    hidden_dim_values: List[int] = field(
        default_factory=lambda: [32, 64, 128]
    )

    # Number of neighbors to test
    k_neighbor_values: List[int] = field(
        default_factory=lambda: [5, 10, 15, 20, 30]
    )


@dataclass
class RealisticDataConfig:
    """Configuration for realistic data generation."""

    n_spots: int = 5000
    n_genes: int = 2000
    n_types: int = 7
    n_ectopic: int = 120
    n_intrinsic_patches: int = 8


# Default configurations
DEFAULT_CONFIG = ExperimentConfig()
ABLATION_CONFIG = AblationConfig()
REALISTIC_CONFIG = RealisticDataConfig()


# Quick test configuration (for debugging)
QUICK_CONFIG = ExperimentConfig(
    seeds=list(range(42, 45)),  # Only 3 seeds
    n_spots=1000,
    n_genes=200,
    n_epochs=20,
)
