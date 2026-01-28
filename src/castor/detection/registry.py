"""Plugin registry for intrinsic anomaly detection methods.

Users can register custom detectors via the decorator::

    from castor.detection.registry import register_intrinsic_detector

    @register_intrinsic_detector("my_method")
    def my_detector(X: np.ndarray, **kwargs) -> np.ndarray:
        # Higher score = more anomalous
        ...
        return scores
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class IntrinsicDetector(Protocol):
    """Protocol for intrinsic anomaly detector functions.

    A valid detector accepts a normalised expression matrix and returns
    per-spot anomaly scores where higher values indicate greater anomaly.
    """

    def __call__(self, X: np.ndarray, **kwargs: object) -> np.ndarray: ...


class IntrinsicDetectorRegistry:
    """Registry of intrinsic anomaly detection methods.

    Built-in methods are registered automatically when the ``castor.detection``
    package is imported.  Users can add custom methods via the
    :func:`register_intrinsic_detector` decorator.
    """

    _registry: dict[str, IntrinsicDetector] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[IntrinsicDetector], IntrinsicDetector]:
        """Decorator that registers a detector function under *name*."""

        def decorator(func: IntrinsicDetector) -> IntrinsicDetector:
            if name in cls._registry:
                logger.warning("Overwriting existing detector: %s", name)
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> IntrinsicDetector:
        """Retrieve a detector by name (raises ``KeyError`` if missing)."""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise KeyError(f"Unknown intrinsic detector: {name!r}. Available: {available}")
        return cls._registry[name]

    @classmethod
    def list_methods(cls) -> list[str]:
        """Return sorted list of registered method names."""
        return sorted(cls._registry.keys())

    @classmethod
    def compute(cls, name: str, X: np.ndarray, **kwargs: object) -> np.ndarray:
        """Compute intrinsic anomaly scores using the named method."""
        detector = cls.get(name)
        return detector(X, **kwargs)


# Convenience alias for end-user registration.
register_intrinsic_detector = IntrinsicDetectorRegistry.register
