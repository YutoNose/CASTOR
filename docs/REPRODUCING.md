# Reproducing the Results

This document provides step-by-step instructions to reproduce all results in the paper.

## System Requirements

### Hardware
- **Minimum**: 16 GB RAM, NVIDIA GPU with 8 GB VRAM
- **Recommended**: 32 GB RAM, NVIDIA GPU with 16+ GB VRAM
- **Tested on**: Ubuntu 22.04, NVIDIA RTX 3090 (24 GB)

### Software
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- ~10 GB disk space for data cache

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone repository
git clone https://github.com/YutoNose/CASTOR.git
cd CASTOR/experiments/14_inverse_prediction

# Create environment
conda env create -f environment.yml

# Activate
conda activate castor-inverse

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option 2: pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Validation

Before running full experiments, verify the setup:

```bash
# Quick test (3 seeds, ~2 minutes)
python run_all.py --quick

# Expected output:
# - exp01: Ectopic AUC ~0.99
# - exp03: Donor fraction ~0.98
# - exp05: Correlation ~-0.03
```

## Reproducing Main Results

### Table 1: Cross-Detection AUC (Figure 2)

```bash
# Full run: 30 seeds, ~30 minutes
python experiments/exp01_cross_detection.py

# Output: results/exp01_cross_detection.csv
```

Expected results:
| Method | Ectopic AUC | Intrinsic AUC |
|--------|-------------|---------------|
| Inv_PosError | 0.997 ± 0.003 | 0.549 ± 0.028 |
| PCA_Error | 0.429 ± 0.029 | 1.000 ± 0.000 |

### Figure 3: Position Prediction Interpretability

```bash
python experiments/exp03_position_accuracy.py

# Output: results/exp03_position_accuracy.csv
```

Expected: 98.7% of ectopic predictions closer to donor than true position.

### Figure 4: Independence Analysis

```bash
python experiments/exp05_independence.py

# Output: results/exp05_independence.csv
```

Expected: Correlation between Inv_PosError and PCA_Error ≈ -0.028.

### Figure 5: Ablation Study

```bash
python experiments/exp07_ablation.py

# Output: results/exp07_ablation.csv
```

Expected: λ_pos = 0 gives random AUC (~0.5), λ_pos > 0 gives high AUC.

### Figure 6: Multi-Scenario Validation

```bash
python experiments/exp10_multi_scenario.py

# Output: results/exp10_multi_scenario.csv
```

Expected: AUC 0.78-0.99 across 9 scenarios.

### Figure 7: Real Data Validation

```bash
python experiments/exp11_real_data.py \
    --datasets mouse_brain_sagittal_posterior human_lymph_node

# Output: results/exp11_real_data.csv
```

Expected: AUC 0.74-0.88 on real Visium data.

### Supplementary: Scalability

```bash
python experiments/exp13_scalability.py

# Output: results/exp13_scalability.csv
```

## Run All Experiments

```bash
# Full pipeline (all 30 seeds, ~3-4 hours)
python run_all.py

# Results saved to results/ directory
# Figures generated in figures/ directory
```

## Data Availability

### Synthetic Data
Generated on-the-fly with fixed random seeds for reproducibility.

### Real Data (Downloaded automatically)
- 10x Genomics Visium samples via `scanpy.datasets.visium_sge()`
- Cached in `data_cache/` directory

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size or use CPU
python run_all.py --device cpu
```

### scanpy Download Fails
```bash
# Manual download from 10x Genomics
# https://www.10xgenomics.com/datasets
```

### Numerical Differences
Small numerical differences (±0.01) may occur due to:
- Different CUDA versions
- Different random number generator implementations
- Floating-point precision

## Expected Runtime

| Experiment | Quick (3 seeds) | Full (30 seeds) |
|------------|-----------------|-----------------|
| exp01 | 1 min | 10 min |
| exp03 | 1 min | 10 min |
| exp05 | 1 min | 10 min |
| exp07 | 5 min | 60 min |
| exp10 | 5 min | 90 min |
| exp11 | 5 min | 60 min |
| **Total** | **~15 min** | **~4 hours** |

## Contact

For issues with reproduction, please open a GitHub issue or contact:
- Yuto Nose (yuto.nose@example.com)
