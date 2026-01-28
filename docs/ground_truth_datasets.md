# Ground Truth Datasets for Validation

This document lists publicly available spatial transcriptomics datasets with biological ground truth that can be used to validate ectopic anomaly detection.

## Priority 1: Datasets with Known Spatial Anomalies

### 1. Tumor Microenvironment Datasets

#### 10x Genomics Human Breast Cancer
- **Source**: 10x Genomics public datasets
- **Access**: `scanpy.datasets.visium_sge("V1_Breast_Cancer_Block_A_Section_1")`
- **Ground Truth**: Pathologist annotations of tumor vs stromal regions
- **Validation Strategy**: Detect cells at tumor-stroma boundary as potential ectopic
- **Paper**: 10x Genomics Application Note

#### Human Prostate Cancer (Berglund et al.)
- **Source**: Spatial Research / GEO
- **Access**: GSE158463
- **Ground Truth**: Gleason grade annotations, tumor regions
- **Size**: Multiple sections, ~4000 spots each
- **Paper**: https://doi.org/10.1038/s41467-020-19649-z

#### TNBC (Triple Negative Breast Cancer) - Wu et al.
- **Source**: Zenodo
- **Access**: https://doi.org/10.5281/zenodo.4739739
- **Ground Truth**: Cell type annotations, immune infiltration markers
- **Paper**: https://doi.org/10.1038/s41588-021-00911-1

### 2. Brain Datasets with Layer Annotations

#### spatialLIBD DLPFC (Dorsolateral Prefrontal Cortex)
- **Source**: Bioconductor/spatialLIBD
- **Access**: R package `spatialLIBD::fetch_data(type = "spe")`
- **Ground Truth**: Manual layer annotations (L1-L6, WM)
- **Size**: 12 samples, ~3600 spots each
- **Validation Strategy**: Cells with wrong layer expression profile
- **Paper**: https://doi.org/10.1038/s41593-020-00787-0

#### Allen Brain Atlas ST Data
- **Source**: Allen Institute
- **Access**: https://portal.brain-map.org/
- **Ground Truth**: ISH-validated gene expression patterns
- **Validation Strategy**: Validate spatial gene patterns

### 3. Developmental Datasets

#### Mouse Embryo (Lohoff et al.)
- **Source**: ArrayExpress
- **Access**: E-MTAB-11115
- **Ground Truth**: Cell type and developmental stage annotations
- **Technology**: seqFISH
- **Paper**: https://doi.org/10.1038/s41587-021-01006-2

#### Zebrafish Embryo Development
- **Source**: ZFIN
- **Ground Truth**: Known developmental patterns

## Priority 2: Semi-Synthetic Validation

### Current Implementation (exp11)

Use real ST data and inject artificial ectopic anomalies:
1. Load public Visium data
2. Cluster spots into regions
3. Copy expression from one region to another
4. Detect injected ectopic cells

Available datasets in `core/real_data.py`:
- `human_lymph_node`: V1_Human_Lymph_Node
- `mouse_brain_sagittal_posterior`: V1_Mouse_Brain_Sagittal_Posterior
- `mouse_brain_coronal`: V1_Adult_Mouse_Brain
- `human_breast_cancer`: V1_Breast_Cancer_Block_A_Section_1

## Priority 3: Immune Infiltration Datasets

### COVID-19 Lung (Rendeiro et al.)
- **Source**: GEO
- **Access**: GSE175644
- **Ground Truth**: Immune cell infiltration annotations
- **Paper**: https://doi.org/10.1038/s41586-021-03570-8

### Inflammatory Bowel Disease
- **Source**: Multiple studies
- **Ground Truth**: Immune cell spatial distribution

## Data Access Code Examples

### scanpy (Python)
```python
import scanpy as sc

# Human Lymph Node
adata = sc.datasets.visium_sge("V1_Human_Lymph_Node")

# Mouse Brain
adata = sc.datasets.visium_sge("V1_Adult_Mouse_Brain")

# Breast Cancer
adata = sc.datasets.visium_sge("V1_Breast_Cancer_Block_A_Section_1")
```

### spatialLIBD (R)
```r
library(spatialLIBD)
spe <- fetch_data(type = "spe")
```

### GEO (Python)
```python
import GEOparse
gse = GEOparse.get_GEO("GSE158463")
```

## Recommended Validation Pipeline

1. **Primary**: Run exp11 on all available Visium datasets
2. **Secondary**: Download DLPFC data, use layer annotations as ground truth
3. **Tertiary**: Access tumor microenvironment datasets for biological validation

## Notes for Nature Methods

- Emphasize that semi-synthetic validation is a standard approach in anomaly detection
- Cite similar validation strategies from other ST papers
- Include at least 2-3 real datasets with different tissue types
- Show that results are consistent across technologies (Visium, Slide-seq)
