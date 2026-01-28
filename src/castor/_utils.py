"""Shared utility functions."""

import numpy as np
import torch


def robust_zscore(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median and MAD.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Robust z-scores.
    """
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))

    if mad < 1e-10:
        mad = np.std(x)
    if mad < 1e-10:
        return np.zeros_like(x)

    return (x - med) / (mad * 1.4826)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
