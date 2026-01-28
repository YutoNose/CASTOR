# Competitor Methods for Spatial Transcriptomics Anomaly Detection

## Currently Implemented (in core/baselines.py)

| Method | Type | Implementation | AUC (Ectopic) | AUC (Intrinsic) |
|--------|------|----------------|---------------|-----------------|
| **Inv_PosError** | Inverse Prediction | Custom | 0.997 | 0.549 |
| PCA_Error | Global | sklearn | 0.429 | 1.000 |
| Neighbor_Diff | Spatial | Custom | 0.896 | 1.000 |
| LISA | Spatial Autocorr | esda/Custom | 0.917 | 0.992 |
| LOF | Global | sklearn | 0.438 | 1.000 |
| Isolation Forest | Global | sklearn | 0.443 | 0.999 |

## Methods to Consider Adding

### 1. SpotSweeper (Already mentioned but may need verification)
- **Purpose**: QC for spatial transcriptomics
- **Paper**: https://doi.org/10.1038/s41467-023-XXXXX
- **Status**: Implemented in baselines.py

### 2. Squidpy (Spatial Analysis Toolkit)
- **Install**: `pip install squidpy`
- **GitHub**: https://github.com/scverse/squidpy
- **Relevant Functions**:
  - `sq.gr.spatial_neighbors()`: Spatial neighborhood analysis
  - `sq.gr.nhood_enrichment()`: Could detect unusual neighborhoods
- **Paper**: https://doi.org/10.1038/s41592-021-01358-2
- **Status**: Not implemented

### 3. STLearn (Spatial Transcriptomics Toolkit)
- **Install**: `pip install stlearn`
- **GitHub**: https://github.com/BiomedicalMachineLearning/stLearn
- **Relevant Functions**:
  - Spatial trajectory analysis
  - Morphology-based clustering
- **Paper**: https://doi.org/10.1101/2020.05.31.125658
- **Status**: Not implemented
- **Note**: Focuses more on trajectory than anomaly detection

### 4. STAGATE (Graph Attention Network)
- **Install**: `pip install STAGATE`
- **GitHub**: https://github.com/zhanglabtools/STAGATE
- **Purpose**: Spatial clustering with graph attention
- **Potential Use**: Embedding-based anomaly detection
- **Paper**: https://doi.org/10.1038/s41467-022-34879-1
- **Status**: Not implemented

### 5. GraphST
- **Install**: `pip install GraphST`
- **GitHub**: https://github.com/JinmiaoChenLab/GraphST
- **Purpose**: Spatial transcriptomics analysis with GNN
- **Paper**: https://doi.org/10.1038/s41467-023-36796-3
- **Status**: Not implemented

### 6. SpaGene
- **GitHub**: https://github.com/jianhuupenn/SpaGene
- **Purpose**: Spatial gene expression pattern detection
- **Status**: Not implemented

## Why Not All Methods Are Suitable Competitors

Most ST analysis tools focus on:
1. **Clustering/Domain identification** - Not anomaly detection
2. **Cell-type deconvolution** - Different task
3. **Trajectory analysis** - Different task
4. **Differential expression** - Requires groups

**Our method is novel because**:
- First to use inverse spatial prediction for anomaly detection
- Specifically targets ectopic anomalies
- Provides interpretable position predictions

## Recommended Comparison Strategy

### Primary Comparisons (Already done)
1. PCA reconstruction error (global baseline)
2. Neighbor difference (spatial baseline)
3. LISA (spatial autocorrelation)
4. LOF (density-based)
5. Isolation Forest (ensemble)

### Optional Additional Comparisons
1. Squidpy neighborhood enrichment as anomaly score
2. STAGATE embedding + IF/LOF

## Code for Additional Comparisons

### Squidpy-based Score
```python
import squidpy as sq

def compute_squidpy_nhood_score(adata):
    """Neighborhood enrichment as anomaly score."""
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="cluster")
    # Convert to per-spot anomaly score
    # (Implementation depends on specific use case)
```

### STAGATE-based Score
```python
from STAGATE import STAGATE

def compute_stagate_score(X, coords):
    """STAGATE embedding + IF for anomaly detection."""
    # 1. Get STAGATE embedding
    # 2. Apply Isolation Forest on embedding
    # 3. Return anomaly scores
```

## Conclusion

Our current baseline comparisons are sufficient for Nature Methods because:
1. We compare against established statistical methods (LISA)
2. We compare against standard ML anomaly detectors (LOF, IF)
3. We compare against spatial methods (Neighbor_Diff)
4. We show that our method is orthogonal (r ≈ 0 correlation)

Adding GNN-based methods (STAGATE, GraphST) would strengthen the comparison but is not strictly necessary for the core claims.
