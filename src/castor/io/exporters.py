"""Result export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_results(results: pd.DataFrame, path: str | Path) -> None:
    """Save detection results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=True)


def save_parameters(params: dict[str, Any], path: str | Path) -> None:
    """Save parameters to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serialisable types
    clean = {}
    for k, v in params.items():
        if hasattr(v, "item"):  # numpy scalar
            clean[k] = v.item()
        else:
            clean[k] = v

    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
