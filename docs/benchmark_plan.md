# ベンチマーク・比較実験計画

## 概要

Nature Methods投稿に向けた競合手法との比較実験計画。
**原則**:
- 疑似実装ではなく実際のツールを使用
- フォールバックなし、失敗時は例外を返す
- 目的別に比較実験を分離

---

## 利用可能なデータセット

### 1. 合成データ (Semi-synthetic)
- **現状**: Human_Lymph_Node, Mouse_Brain_Sagittal_Posterior
- **追加予定**: Breast_Cancer, Mouse_Brain_Coronal

### 2. 実データ (Ground Truth付き)
- **HER2ST**: 8サンプル (A1, B1, C1, D1, E1, F1, G2, H1)
  - 病理医アノテーション: invasive cancer, dcis (異常) vs connective tissue, fat (正常)
  - `/home/yutonose/CASTOR/her2st/`
  - `HER2STDataLoader` で読み込み

---

## 競合手法の分類

### Category A: 空間異常検出（直接比較可能）
| 手法 | パッケージ | 異常スコア取得方法 | 実装優先度 |
|------|-----------|-------------------|-----------|
| LISA | esda/libpysal | Local Moran's I | ✅ 実装済 |
| Neighbor_Diff | Custom | 近傍との発現差 | ✅ 実装済 |
| SpotSweeper | Custom/stlearn | QC metrics | ✅ 実装済 |

### Category B: 空間Embedding + 異常検出
| 手法 | パッケージ | 手順 | 実装優先度 |
|------|-----------|------|-----------|
| STAGATE + IF | STAGATE 1.0.1 | 1. STAGATE embedding → 2. IF on embedding | 高 |
| GraphST + IF | GraphST 1.1.1 | 1. GraphST embedding → 2. IF on embedding | 高 |
| Squidpy Nhood | squidpy 1.6.5 | Neighborhood enrichment異常度 | 中 |

### Category C: グローバル異常検出（ベースライン）
| 手法 | パッケージ | 状態 |
|------|-----------|------|
| PCA_Error | sklearn | ✅ 実装済 |
| LOF | sklearn | ✅ 実装済 |
| Isolation Forest | sklearn | ✅ 実装済 |

### Category D: 目的が異なる手法（参考比較）
| 手法 | 本来の目的 | 比較方法 |
|------|-----------|---------|
| STLearn SME | 空間マーカー検出 | マーカー発現 vs 異常スコア相関 |
| STLearn Trajectory | 軌跡推定 | Pseudo-time extremes as anomaly |

---

## 実験設計

### Exp12: 主要比較実験 (Embedding-based)

```
データ:
  - 合成データ: exp01と同一設定 (3000 spots, 100 ectopic, 300 intrinsic)
  - 実データ: HER2ST全8サンプル

手法 (全て実ツール使用):
  1. Inv_PosError (ours)
  2. STAGATE + IF
  3. GraphST + IF
  4. Squidpy neighborhood score
  5. [既存] PCA_Error, LISA, LOF, IF, Neighbor_Diff

評価指標:
  - Ectopic Detection AUC
  - Intrinsic Detection AUC (合成のみ)
  - Cancer Detection AUC (HER2STのみ)
  - Computation Time
  - Memory Usage
```

### Exp14: HER2ST実データ検証

```
目的: Ground truthのある実データでの性能検証

データ: HER2ST 8サンプル
  - y_true: 病理医アノテーション (invasive cancer/dcis vs others)

評価:
  - Cancer vs Normal separation AUC
  - サンプルごとの性能
  - ERBB2発現との相関分析 (参考)
```

### Exp15: 追加Visiumデータ検証

```
目的: データセット多様性の確保

データ (ダウンロード予定):
  - V1_Breast_Cancer_Block_A_Section_1
  - V1_Adult_Mouse_Brain (Coronal)
  - V1_Mouse_Brain_Sagittal_Anterior

評価: Semi-synthetic ectopic injection
```

---

## 各手法の実装詳細

### STAGATE + IF
```python
import STAGATE
from sklearn.ensemble import IsolationForest

def compute_stagate_score(adata):
    """STAGATE embedding + Isolation Forest."""
    # 1. Run STAGATE
    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    STAGATE.train_STAGATE(adata, alpha=0, random_seed=42)

    # 2. Get embedding
    embedding = adata.obsm['STAGATE']

    # 3. Apply IF
    clf = IsolationForest(contamination=0.1, random_state=42)
    scores = -clf.decision_function(embedding)  # Higher = more anomalous

    return scores
```

### GraphST + IF
```python
from GraphST import GraphST
from sklearn.ensemble import IsolationForest

def compute_graphst_score(adata):
    """GraphST embedding + Isolation Forest."""
    # 1. Run GraphST
    model = GraphST.GraphST(adata, device='cuda')
    adata = model.train()

    # 2. Get embedding
    embedding = adata.obsm['emb']

    # 3. Apply IF
    clf = IsolationForest(contamination=0.1, random_state=42)
    scores = -clf.decision_function(embedding)

    return scores
```

### Squidpy Neighborhood
```python
import squidpy as sq

def compute_squidpy_nhood_score(adata):
    """Squidpy neighborhood enrichment as anomaly score."""
    # 1. Compute spatial neighbors
    sq.gr.spatial_neighbors(adata, coord_type='generic', n_neighs=15)

    # 2. Cluster first
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.leiden(adata, resolution=0.5)

    # 3. Neighborhood enrichment
    sq.gr.nhood_enrichment(adata, cluster_key='leiden')

    # 4. Per-spot score: fraction of neighbors in different cluster
    # (実装: 近傍の多数派クラスタと自身のクラスタの不一致度)
    scores = compute_cluster_mismatch(adata)

    return scores
```

---

## 実験パラメータ

### 共通設定
```python
COMMON_CONFIG = {
    'random_state': 42,
    'n_seeds': 30,  # 統計的有意性
    'contamination': 0.1,  # IF用 (実際の異常率に近似)
}
```

### 手法別設定 (デフォルト使用、チューニングなし)
```python
STAGATE_CONFIG = {
    'alpha': 0,  # Reconstruction weight
    'hidden_dims': [512, 30],  # Default
    'rad_cutoff': 150,  # Spatial radius
}

GRAPHST_CONFIG = {
    'n_top_genes': 3000,
    'epochs': 1000,  # Default
}
```

---

## ファイル構成

```
experiments/14_inverse_prediction/
├── experiments/
│   ├── exp12_embedding_comparison.py   # STAGATE, GraphST比較
│   ├── exp14_her2st_validation.py      # HER2ST実データ
│   └── exp15_additional_visium.py      # 追加Visiumデータ
├── core/
│   ├── competitor_stagate.py           # STAGATE wrapper
│   ├── competitor_graphst.py           # GraphST wrapper
│   └── competitor_squidpy.py           # Squidpy wrapper
└── data/
    ├── V1_Human_Lymph_Node/
    ├── V1_Mouse_Brain_Sagittal_Posterior/
    ├── V1_Breast_Cancer_Block_A_Section_1/  # 追加
    └── V1_Adult_Mouse_Brain/                # 追加
```

---

## 実装順序

### Phase 1: コア実装 ✅
1. [x] `core/competitor_stagate.py` - STAGATE wrapper
2. [x] `core/competitor_graphst.py` - GraphST wrapper
3. [x] `core/competitor_squidpy.py` - Squidpy wrapper
4. [x] 各wrapperのテスト (単体)

**注意**: STAGATE/GraphSTはRTX 5090 (compute 12.0)との互換性問題あり
- STAGATE: TensorFlowがGPUで動作せず、CPU専用
- GraphST: CUDAメモリアクセスエラー

### Phase 2: 統合実験 ✅
5. [x] `exp12_embedding_comparison.py` - 合成データ比較
6. [x] `exp14_her2st_validation.py` - HER2ST実データ
7. [ ] 追加Visiumデータのダウンロード

### Phase 3: 検証・図表 (進行中)
8. [ ] 全実験の実行 (30 seeds)
9. [ ] 結果の可視化
10. [ ] 統計検定

---

## 予備結果

### exp12 Synthetic (5 seeds, quick test)

| Scenario | Inv_PosError | LISA | Neighbor_Diff | Squidpy | PCA_Error | IF |
|----------|-------------|------|---------------|---------|-----------|-----|
| baseline | 0.953±0.034 | 0.999±0.001 | 0.999±0.001 | 0.963±0.018 | 0.349±0.013 | 0.500±0.043 |
| noisy_ectopic | 0.968±0.012 | 1.000±0.000 | 1.000±0.000 | 0.968±0.031 | 0.613±0.012 | 0.590±0.026 |
| **hard_ectopic** | **0.691±0.124** | 0.514±0.041 | 0.527±0.042 | **0.716±0.031** | 0.001±0.000 | 0.018±0.004 |
| realistic | 0.961±0.020 | 0.995±0.002 | 0.998±0.001 | 0.952±0.014 | 0.440±0.034 | 0.595±0.025 |

**Key insight**: hard_ectopicシナリオ(50% mixing + noise)では:
- LISA/Neighbor_Diff → ランダム (0.51-0.53)
- Inv_PosError → 維持 (0.69)
- Squidpy → 最高 (0.72)

### exp14 HER2ST (1 sample, 1 seed)

| Method | Cancer Detection AUC |
|--------|---------------------|
| LISA | 0.776 |
| IF | 0.720 |
| Neighbor_Diff | 0.697 |
| LOF | 0.690 |
| PCA_Error | 0.670 |
| SpotSweeper | 0.670 |
| **Inv_PosError** | 0.524 |
| Squidpy | 0.224 |

**Note**: HER2STはクラスタ状のGround Truth (腫瘍領域) であり、
散在するEctopic異常とは異なるタスク。Inv_PosErrorの低AUCは予想通り。

---

## 注意事項

1. **例外処理**: フォールバックなし。ツールが失敗したら即座に例外
2. **バージョン固定**:
   - STAGATE==1.0.1
   - GraphST==1.1.1
   - squidpy==1.6.5
3. **再現性**: 全てrandom_state=42で固定
4. **計算リソース**: STAGATE/GraphSTはGPU推奨
