"""Detection layer: registry, built-in detectors, scoring."""

# Importing intrinsic registers all built-in detectors.
from castor.detection import intrinsic as _intrinsic  # noqa: F401
from castor.detection.registry import (
    IntrinsicDetectorRegistry,
    register_intrinsic_detector,
)
from castor.detection.scoring import compute_diagnosis, compute_mad_zscore

__all__ = [
    "IntrinsicDetectorRegistry",
    "register_intrinsic_detector",
    "compute_diagnosis",
    "compute_mad_zscore",
]
