# Issues Requiring Experiment Re-run

これらの問題は結果に影響するため、修正後に実験の再実行が必要です。

**Status: すべての修正が完了しました。実験を再実行してください。**

---

## 修正完了項目 ✅

### 1. Ectopic Anomaly Cascading Contamination ✅
**File**: `core/data_generation.py`

**修正内容**: ループ前に `mu_original = mu_matrix.copy()` を追加し、全ての donor copy がオリジナルの mu_matrix から取得されるようになりました。

**影響**: 3つの関数すべて修正完了
- `generate_synthetic_data()` ✅
- `generate_controlled_ectopic()` ✅
- `generate_raw_counts()` ✅

---

### 2. PyTorch Seed Not Set ✅
**File**: 全実験ファイル (exp01-exp18)

**修正内容**: 全実験ファイルに `set_seed(seed)` の呼び出しを追加。
この関数は `core/utils.py` で定義されており、numpy と torch の両方の seed を設定します。

**修正済みファイル**:
- exp01_cross_detection.py ✅
- exp02_competitor.py ✅
- exp03_position_accuracy.py ✅
- exp04_noise_robustness.py ✅
- exp05_independence.py ✅
- exp07_ablation.py ✅
- exp08_clean_training.py ✅
- exp09_clean_training_fixed.py ✅
- exp10_multi_scenario.py ✅
- exp11_real_data.py ✅
- exp12_embedding_comparison.py ✅
- exp13_scalability.py ✅
- exp14_her2st_validation.py ✅
- exp15_full_benchmark.py ✅
- exp16_real_data_gene_analysis.py ✅
- exp17_her2st_transplantation.py ✅
- exp18_interpretability.py ✅

---

### 3. LISA Implementation ✅
**File**: `core/baselines.py`

**修正内容**:
- 真の Local Moran's I を `compute_lisa()` として実装
- 旧実装（mean absolute deviation）を `compute_local_spatial_deviation()` にリネーム
- `compute_all_baselines()` に両方を追加

**注**: 真の Local Moran's I の公式:
```
I_i = z_i * Σ_j(w_ij * z_j)
```
空間的外れ値は負の I 値を持つため、`-mean(I)` をスコアとして返す。

---

### 4. NumPy 2.0 Deprecation ✅
**File**: `core/utils.py`

**修正内容**: `np.ptp()` を `max() - min()` に置換。

---

### 5. Double Normalization Risk ✅
**File**: `core/preprocessing.py`

**修正内容**: `prepare_from_anndata()` にデータ状態検出ロジックを追加:
- Raw counts (max > 100): 完全な正規化
- Log-transformed (max < 15, min >= 0): scale のみ
- Z-scored (min < 0): 正規化スキップ
- `skip_normalization=True` オプション追加

---

### 6. Scenarios.py Cascading Contamination ✅
**File**: `core/scenarios.py`

**修正内容**: `generate_scenario_data()` 内の全ての ectopic タイプ (ectopic, noisy_ectopic, partial_ectopic, hard_ectopic, hardest) で `mu_original = mu_matrix.copy()` を追加。

---

### 7. Effect Size Formula ✅
**File**: `core/evaluation.py`

**修正内容**: Wilcoxon signed-rank の rank-biserial correlation 公式を修正:
- 旧 (誤り): `effect_size = 4 * stat / (n * (n + 1)) - 1`
- 新 (正解): `effect_size = 1 - 4 * stat / (n * (n + 1))`

---

### 8. Statistical Tests Multiple Comparison ✅
**File**: `experiments/exp15_full_benchmark.py`

**修正内容**: `statistical_tests()` 関数に Bonferroni 補正を追加:
- 総比較数をカウントし `alpha_corrected = alpha / n_comparisons` を計算
- Wilcoxon p値を使用（bounded AUC に適した non-parametric テスト）
- `significant` フィールドは補正後の alpha で判定

---

### 9. Exp18 Interpretability Statistical Test ✅
**File**: `experiments/exp18_interpretability.py`

**修正内容**: "closer to donor" の主張に対する統計検定を追加:
- Binomial test (H0: p=0.5)
- Wilcoxon signed-rank test on paired distances
- 結果を DataFrame.attrs に保存

---

### 10. SpotSweeper Z-scored Data Fix ✅
**File**: `core/baselines.py`

**修正内容**: `compute_spotsweeper()` で z-scored データを検出し、library size の代わりに L2 norm を使用するように修正。

---

## 追加で修正した項目 (non-result-affecting)

これらは以前の修正で完了済み:

- evaluation.py: Rank-biserial correlation effect size ✅
- evaluation.py: ddof=1 for sample std ✅
- fig5/combined.py: データ検索順序 ✅
- fig7/combined.py: 有意性ラインのラベル ✅
- figS4/panel_a.py: Heatmap vmin ✅
- figS6/panel_a.py: np.ptp() 置換 ✅
- figS3/panel_b.py: Wilcoxon one-sided test ✅
- figS4/panel_c.py: ddof=1 for Cohen's d ✅
- competitor_graphst.py: n_epochs パラメータの制限を文書化 ✅
- exp16/17/18: 冗長な torch.manual_seed() 呼び出しを削除 ✅

---

## 再実行コマンド

```bash
cd /home/yutonose/Projects/experiments

# 全実験を順次実行
for exp in exp*.py; do
    echo "=============================="
    echo "Running $exp..."
    echo "=============================="
    python "$exp"
done

# 可視化を再生成
cd visualization
python generate_all.py
```

### 優先度の高い実験

以下の実験は結果に最も影響があるため、最初に再実行してください:

```bash
# 1. 主要な cross-detection 実験
python exp01_cross_detection.py

# 2. 競合手法比較
python exp02_competitor.py

# 3. HER2ST 検証
python exp14_her2st_validation.py

# 4. Full benchmark
python exp15_full_benchmark.py
```

---

## 変更の影響予測

1. **Cascading contamination 修正** (data_generation.py + scenarios.py):
   - Ectopic detection AUC がわずかに向上する可能性
   - 二次汚染（~0.1%）の除去により、より正確な評価

2. **PyTorch seed 設定**:
   - 個別 seed の結果は変化
   - 30 seeds の平均は大きく変わらない見込み
   - 再現性が保証される

3. **LISA 実装変更**:
   - LISA の結果が変化（真の Local Moran's I）
   - `local_spatial_deviation` は旧実装と同一の結果

4. **Effect size 修正** (evaluation.py):
   - 統計検定の効果量が正しく報告される
   - 以前は符号が反転していた

5. **SpotSweeper 修正**:
   - z-scored データでも正しく動作
   - 結果が変化する可能性あり

6. **Statistical tests 修正** (exp15):
   - Bonferroni 補正により、以前 "significant" だった結果が non-significant になる可能性
   - より厳格で正確な有意性判定

7. **Exp18 statistical tests**:
   - 新しい binomial/Wilcoxon 検定により interpretability の主張が統計的に検証可能に

---

## 残存する Known Issues (修正不要または将来の改善)

これらは audit で特定されましたが、今回の再実行では修正しません：

1. **Competitor 実装の制限**: STLearn、Squidpy、GraphST は「embedding + Isolation Forest」パターンを使用。これらのツールの元来の設計目的とは異なりますが、anomaly detection への適用として妥当。

2. **Train/Test Split**: 全手法が同じデータで train/score。Unsupervised anomaly detection では標準的。

3. **Preprocessing 差異**: 提案手法と competitors で preprocessing が異なる。各手法の推奨パイプラインを使用。

---

*Last updated: 2026-01-28*
*All fixes applied by: Claude Code*
