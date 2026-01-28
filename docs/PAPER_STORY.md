# Inverse Spatial Prediction for Anomaly Detection in Spatial Transcriptomics

## Paper Story and Figure Plan

---

## Title (案)
**"Inverse Spatial Prediction Reveals Ectopic Cell States in Spatial Transcriptomics"**

または

**"Position Prediction Error Enables Selective Detection of Spatially Misplaced Anomalies"**

---

## Abstract Summary

Spatial transcriptomics reveals spatially resolved gene expression, but detecting anomalous cells remains challenging. Existing methods detect cells with unusual expression profiles (intrinsic anomalies) but fail to identify cells that are spatially misplaced despite normal expression (ectopic anomalies). We propose **inverse spatial prediction**—predicting a cell's spatial position from its expression profile—to selectively detect ectopic anomalies. When a cell's predicted position differs from its actual location, this indicates spatial displacement. We demonstrate that position prediction error achieves near-perfect detection (AUC=0.999) of ectopic anomalies while remaining independent of expression-based methods, enabling complementary detection of both anomaly types.

---

## Key Claims (主張)

1. **Conceptual Novelty**: 発現→位置の逆予測を異常検出に初適用
2. **Selective Detection**: Position errorはEctopic特化、既存手法はIntrinsic特化
3. **Robustness**: Hard ectopic（近傍からの移植）で唯一高性能を維持
4. **Independence**: Expression-based手法と相関ゼロ → 相補的検出が可能
5. **Interpretability**: 予測位置がドナー位置を指す → 生物学的解釈可能

---

## Figure Plan

### Main Figures (4-5 figures)

#### **Figure 1: Concept and Method Overview**
**Caption**: Inverse spatial prediction for ectopic anomaly detection.

| Panel | Content | Purpose |
|-------|---------|---------|
| (a) | Schematic: Forward (position→expression) vs Inverse (expression→position) prediction | 概念の導入 |
| (b) | Two types of anomalies: Ectopic (spatially misplaced) vs Intrinsic (expression outlier) | 問題設定の明確化 |
| (c) | Method workflow: Expression → GNN → Position prediction → Error as anomaly score | 手法の概要 |
| (d) | Intuition: Ectopic cell's predicted position points to donor region | なぜこの手法が効くのか |

**File**: `fig1_concept.py`

---

#### **Figure 2: Selective Detection of Ectopic Anomalies**
**Caption**: Position prediction error selectively detects ectopic anomalies while expression-based methods detect intrinsic anomalies.

| Panel | Content | Purpose |
|-------|---------|---------|
| (a) | Cross-detection AUC heatmap (methods × anomaly types) | 選択性の定量化 |
| (b) | Score distributions: Inv_Pos separates ectopic, PCA separates intrinsic | 視覚的な選択性 |
| (c) | Scatter: Inv_Pos score vs PCA score with anomaly labels | 直交性の可視化 |

**Key message**: Inv_PosError achieves AUC=0.999 for ectopic, ~0.5 for intrinsic. PCA achieves AUC=1.0 for intrinsic, ~0.5 for ectopic.

**File**: `fig2_selectivity.py`

---

#### **Figure 3: Robustness to Hard Ectopic Scenarios**
**Caption**: Position prediction error is the only method robust to spatially proximal ectopic anomalies.

| Panel | Content | Purpose |
|-------|---------|---------|
| (a) | Scenario illustration: Easy ectopic (distant) vs Hard ectopic (nearby) | シナリオの説明 |
| (b) | Bar chart: Method performance across scenarios (baseline, noisy, hard, realistic) | 全シナリオ比較 |
| (c) | Hard ectopic detail: Only Inv_Pos maintains AUC>0.9, others collapse | 主張の核心 |

**Key message**: In hard_ectopic, Inv_PosError=0.948 while PCA=0.000, LOF=0.069, IF=0.013

**File**: `fig3_robustness.py`

---

#### **Figure 4: Independence from Expression-based Methods**
**Caption**: Position prediction error is orthogonal to global expression-based anomaly scores.

| Panel | Content | Purpose |
|-------|---------|---------|
| (a) | Correlation matrix of all methods | 全手法間の関係性 |
| (b) | Scatter: Inv_Pos vs PCA/LOF/IF scores | 直交性の詳細 |
| (c) | Combined detection: Using both Inv_Pos and PCA captures both anomaly types | 相補的検出の実証 |

**Key message**: Correlation with global methods ≈ 0 enables complementary detection

**File**: `fig4_independence.py`

---

#### **Figure 5: Real Data Validation (HER2ST)**
**Caption**: Position prediction error detects tumor regions in HER2-positive breast cancer spatial transcriptomics.

| Panel | Content | Purpose |
|-------|---------|---------|
| (a) | Spatial visualization: Pathologist annotations vs Inv_Pos scores | 定性的な一致 |
| (b) | ROC curves for cancer detection across samples | 定量的な評価 |
| (c) | Method comparison: AUC across 8 HER2ST samples | 手法比較 |

**Note**: HER2STの"anomaly"はclustered tumor regionsなので、spatial methodsが有利。Inv_Posの位置づけを慎重に議論。

**File**: `fig5_her2st.py`

---

### Supplementary Figures

#### **Figure S1: Noise Robustness Analysis**
**Caption**: Position prediction error maintains high performance under increasing expression noise.

| Panel | Content |
|-------|---------|
| (a) | AUC vs noise level for all methods |
| (b) | Score stability across noise levels |

**File**: `figS1_noise.py`

---

#### **Figure S2: Ablation Study**
**Caption**: Hyperparameter sensitivity analysis.

| Panel | Content |
|-------|---------|
| (a) | Effect of λ_pos (position loss weight) |
| (b) | Effect of hidden dimension |
| (c) | Effect of k neighbors |

**File**: `figS2_ablation.py`

---

#### **Figure S3: Position Prediction Interpretability**
**Caption**: Predicted positions point toward donor regions for ectopic cells.

| Panel | Content |
|-------|---------|
| (a) | Prediction arrows for normal vs ectopic cells |
| (b) | Distance from predicted position to true donor |

**File**: `figS3_interpretability.py`

---

#### **Figure S4: Statistical Tests**
**Caption**: Statistical significance of method comparisons.

| Panel | Content |
|-------|---------|
| (a) | Paired t-test p-values (Bonferroni corrected) |
| (b) | Effect sizes (Cohen's d) |

**File**: `figS4_statistics.py`

---

#### **Figure S5: Scalability Analysis**
**Caption**: Computational efficiency across dataset sizes.

| Panel | Content |
|-------|---------|
| (a) | Runtime vs number of spots |
| (b) | Memory usage vs number of spots |

**File**: `figS5_scalability.py`

---

## Experiment Summary Table

| Experiment | File | Description | Main Finding | Figure |
|------------|------|-------------|--------------|--------|
| exp01 | exp01_cross_detection.py | Cross-detection AUC | Selectivity | Fig 2a |
| exp02 | exp02_competitor.py | Competitor comparison | Method ranking | Fig 3b |
| exp03 | exp03_position_accuracy.py | Position prediction accuracy | Interpretability | Fig S3 |
| exp04 | exp04_noise_robustness.py | Noise robustness | Stability | Fig S1 |
| exp05 | exp05_independence.py | Score independence | Orthogonality | Fig 4 |
| exp07 | exp07_ablation.py | Hyperparameter sensitivity | Robustness | Fig S2 |
| exp10 | exp10_multi_scenario.py | Multiple scenarios | Hard ectopic | Fig 3 |
| exp12 | exp12_embedding_comparison.py | Embedding methods | STAGATE/GraphST | Fig 3b |
| exp13 | exp13_scalability.py | Scalability | Efficiency | Fig S5 |
| exp14 | exp14_her2st_validation.py | HER2ST validation | Real data | Fig 5 |
| exp15 | exp15_full_benchmark.py | Full benchmark | All methods | Fig 2,3,4 |

---

## Story Flow

```
1. Introduction
   - Spatial transcriptomics enables spatially resolved analysis
   - Anomaly detection is important for identifying rare/aberrant cells
   - Existing methods focus on expression-based outliers (intrinsic)
   - Gap: Spatially misplaced cells (ectopic) are not detected

2. Results
   2.1 Inverse prediction concept (Fig 1)
       - Forward: learn spatial patterns → predict expression
       - Inverse: learn expression patterns → predict position
       - Position error indicates spatial displacement

   2.2 Selective detection (Fig 2)
       - Inv_Pos detects ectopic (AUC=0.999), not intrinsic
       - PCA/LOF/IF detect intrinsic, not ectopic
       - Cross-detection matrix shows complementarity

   2.3 Robustness (Fig 3)
       - Hard ectopic scenario: nearby transplant
       - Only Inv_Pos maintains performance
       - Existing methods collapse (AUC→0)

   2.4 Independence (Fig 4)
       - Correlation analysis: r≈0 with global methods
       - Combined detection captures both anomaly types
       - Practical implication: use both methods

   2.5 Real data (Fig 5)
       - HER2ST breast cancer validation
       - Tumor detection performance
       - Comparison with state-of-the-art

3. Discussion
   - Why inverse prediction works for ectopic
   - Limitations: clustered anomalies, computational cost
   - Future: integration with cell type annotation
```

---

## Code/Logic Check Points

### Critical Items to Verify

1. **Ectopic injection logic** (`data_generation.py`)
   - [ ] Ensure donor is from distant position
   - [ ] Verify label assignment happens AFTER successful injection

2. **Score computation** (`core/__init__.py`)
   - [ ] Position error = ||predicted_pos - true_pos||
   - [ ] Higher score = more anomalous

3. **AUC computation** (`evaluation.py`)
   - [ ] Correct label encoding (1=anomaly, 0=normal)
   - [ ] No label conditioning that biases results

4. **Neighbor aggregation** (`baselines.py`)
   - [ ] axis=1 for k neighbors mean (not axis=0)

5. **Statistical tests** (`exp15_full_benchmark.py`)
   - [ ] Paired t-test with same samples
   - [ ] Bonferroni correction for multiple comparisons

---

## Key Results Summary

### Ablation Study Results (exp07)

| Parameter | Values Tested | Finding |
|-----------|---------------|---------|
| lambda_pos | 0, 0.1, 0.5, 1.0, 2.0 | λ=0: AUC=0.509 (random), λ≥0.1: AUC≥0.995 |
| hidden_dim | 32, 64, 128 | All work well (0.994-0.997) |
| k_neighbors | 5, 10, 15, 20, 30 | All work well (0.996-0.997) |

**Key insight**: Position loss is essential (λ=0 fails), but the method is robust to other hyperparameters.

### Scalability Results (exp13)

| n_spots | Total Time | Train Time | GPU Memory | AUC (Ectopic) |
|---------|------------|------------|------------|---------------|
| 1,000   | 0.65s      | 0.52s      | 40 MB      | 0.997         |
| 3,000   | 0.33s      | 0.15s      | 88 MB      | 0.995         |
| 5,000   | 0.44s      | 0.17s      | 140 MB     | 0.996         |
| 10,000  | 0.59s      | 0.17s      | 258 MB     | 0.996         |
| 20,000  | 1.21s      | 0.23s      | 483 MB     | 0.996         |
| 30,000  | 1.79s      | 0.29s      | 717 MB     | 0.997         |

**Key insight**: Scales linearly, processes 30K spots in <2 seconds, AUC remains consistent.

### Benchmark Results (exp15)

| Scenario | Inv_PosError | PCA_Error | LOF | IF | LISA |
|----------|--------------|-----------|-----|-----|------|
| baseline | 0.999 | 0.59 | 0.69 | 0.54 | 0.59 |
| hard_ectopic | **0.948** | 0.00 | 0.07 | 0.01 | - |
| noisy_ectopic | 0.99+ | 0.55 | 0.65 | 0.52 | - |

**Key insight**: Only Inv_PosError maintains performance in hard_ectopic scenario.

---

## Visualization Files

All visualization scripts are in `experiments/visualization/`:

### Main Figures
- `fig1_concept.py` - Concept and method overview
- `fig2_selectivity.py` - Selective detection of ectopic anomalies
- `fig3_robustness.py` - Robustness to hard ectopic scenarios
- `fig4_independence.py` - Independence from expression-based methods
- `fig5_her2st.py` - Real data validation (HER2ST)

### Supplementary Figures
- `figS1_noise.py` - Noise robustness analysis
- `figS2_ablation.py` - Ablation study
- `figS3_interpretability.py` - Position prediction interpretability
- `figS4_statistics.py` - Statistical analysis
- `figS5_scalability.py` - Scalability analysis

### Utilities
- `common.py` - Shared styles, colors, utilities
- `generate_all_figures.py` - Main script to generate all figures

---

## Reproducibility

All experiments should be reproducible with:

```bash
# Full benchmark (30 seeds)
CUDA_VISIBLE_DEVICES='' python experiments/exp15_full_benchmark.py --seeds 30

# HER2ST validation
CUDA_VISIBLE_DEVICES='' python experiments/exp14_her2st_validation.py --seeds 30

# Generate all figures
python experiments/visualization/generate_all_figures.py
python experiments/visualization/generate_all_figures.py --main   # Main only
python experiments/visualization/generate_all_figures.py --supp   # Supplementary only
python experiments/visualization/generate_all_figures.py --fig 2  # Specific figure
```

## Output Directories

- Results: `experiments/14_inverse_prediction/results/`
- Figures: `experiments/14_inverse_prediction/figures/`
