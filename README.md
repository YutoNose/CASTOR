# CASTOR

Dual-axis anomaly detection for spatial transcriptomics.

## Install

```bash
uv sync
```

## Usage

```bash
# Detect anomalies
castor detect data.h5ad
castor detect expression.csv --coords coordinates.csv

# Visualize results
castor visualize results.csv --input data.h5ad

# Gene & pathway enrichment
castor enrich results.csv --expression data.h5ad

# Show info & available methods
castor info --methods
```

## Python API

```python
from castor import CASTOR

c = CASTOR(intrinsic_method="pca_error")
results = c.fit_predict("data.h5ad")
c.plot_results(save_path="figure.png")
genes = c.get_contributing_genes()
```

## Custom intrinsic detectors

```python
from castor import register_intrinsic_detector
import numpy as np

@register_intrinsic_detector("my_method")
def my_detector(X: np.ndarray, **kwargs) -> np.ndarray:
    # Return per-spot anomaly scores (higher = more anomalous)
    ...
```
