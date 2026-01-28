"""CASTOR: Dual-Axis Anomaly Detection for Spatial Transcriptomics."""

from castor.api import CASTOR
from castor.detection.registry import register_intrinsic_detector

__version__ = "0.1.0"
__all__ = ["CASTOR", "register_intrinsic_detector", "__version__"]
