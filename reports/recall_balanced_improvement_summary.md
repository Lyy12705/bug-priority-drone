# Recall-Balanced 改善實驗總結

本次依照以下順序實作：

1. 分析 `P1 -> P2` 與 `P4 -> P2` 錯誤細節
2. 加入 `P1/P2 boundary classifier`
3. 加入 `P4 false-high suppression`
4. 使用 `macro recall + minimum recall` 做 final objective grid search

## 1. P1 -> P2 與 P4 -> P2 錯誤分析

使用目前最佳模型 `improved_priority_p2_keywords_model.joblib` 分析後，兩個目標錯誤方向如下：

| 錯誤方向 | 筆數 |
|---|---:|
| P1 Blocker -> P2 Critical | 50 |
| P4 Minor -> P2 Critical | 23 |

解讀：

- `P1 -> P2` 代表高優先級 bug 被判太輕，會壓低 P1 recall。
- `P4 -> P2` 代表低優先級 bug 被模型拉太高，表示模型可能過度相信某些 high-priority 訊號。

輸出：

- `reports/targeted_error_directions.csv`
- `reports/targeted_error_directions.md`

## 2. Recall-Balanced 模型設計

新增模型流程：

```text
base DRONE/REP- classifier
        ↓
P1/P2 boundary classifier
        ↓
P4 false-high suppression
        ↓
final prediction P1-P5
```

### P1/P2 boundary classifier

用途：

- 只針對 base model 預測為 P1 或 P2 的資料再次判斷。
- 目標是補回 P1 recall，避免 P1 被降成 P2。

### P4 false-high suppression

用途：

- 只針對目前被預測為 P2 的資料再次判斷是否其實比較像 P4。
- 目標是降低 `P4 -> P2` 這種低優先級被拉太高的錯誤。

## 3. Final Objective Grid Search

這次選模不只看 P2 recall，而是改成平衡每個 priority：

```text
macro recall
+ minimum recall
+ macro F1
+ accuracy
+ off-by-one accuracy
- MAE penalty
- low-recall penalty
```

目的：

- 提升整體 per-class recall 平衡度。
- 避免只提升某一類，卻讓另一類 recall 崩掉。
- 特別關注 P1、P4 這兩個原本需要改善的類別。

## 4. 結果比較

| 模型 | Accuracy | Macro F1 | Macro Recall | Min Recall | Off-by-one | MAE | P1 Recall | P2 Recall | P3 Recall | P4 Recall | P5 Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 目前最佳模型 | 0.7246 | 0.7240 | 0.7244 | 0.6482 | 0.8995 | 0.4090 | 0.6482 | 0.6818 | 0.8750 | 0.6834 | 0.7337 |
| Recall-balanced 模型 | 0.7317 | 0.7309 | 0.7315 | 0.6583 | 0.9025 | 0.4000 | 0.6583 | 0.6818 | 0.8750 | 0.7085 | 0.7337 |

## 5. 主要改善

| 指標 | 目前最佳 | Recall-balanced | 變化 |
|---|---:|---:|---:|
| Accuracy | 0.7246 | 0.7317 | +0.0070 |
| Macro F1 | 0.7240 | 0.7309 | +0.0069 |
| Macro Recall | 0.7244 | 0.7315 | +0.0070 |
| Min Recall | 0.6482 | 0.6583 | +0.0101 |
| MAE | 0.4090 | 0.4000 | -0.0090 |
| P1 Recall | 0.6482 | 0.6583 | +0.0101 |
| P4 Recall | 0.6834 | 0.7085 | +0.0251 |

## 6. 錯誤方向改善

| 錯誤方向 | 原本 | Recall-balanced | 變化 |
|---|---:|---:|---:|
| P1 -> P2 | 50 | 50 | 0 |
| P4 -> P2 | 23 | 18 | -5 |

解讀：

- P1 recall 有提升，但主要不是因為 `P1 -> P2` 降低，而是減少了 P1 被錯判到其他類別的情況。
- P4 false-high suppression 有作用，`P4 -> P2` 從 23 筆降到 18 筆。
- P4 recall 從 0.6834 提升到 0.7085，是本次最明顯的 per-class recall 改善。

## 7. 結論

Recall-balanced 模型比目前最佳模型更好：

- 整體 accuracy 更高
- macro F1 更高
- macro recall 更高
- minimum recall 更高
- MAE 更低
- P1、P4 recall 都改善，且 P2/P3/P5 recall 沒有下降

因此目前可以把 `recall_balanced_priority_model.joblib` 視為新的最佳實驗模型。報告中可以說明：這次不是單純追求 P2 recall，而是根據 per-class error analysis 針對 P1/P2 與 P4 false-high 問題做局部修正，最後用 macro recall 和 minimum recall 作為更平衡的選模目標。

## 8. 輸出檔案

- `scripts/analyze_targeted_error_directions.py`
- `scripts/train_recall_balanced_priority_model.py`
- `reports/targeted_error_directions.md`
- `reports/targeted_error_directions_recall_balanced.md`
- `reports/recall_balanced_priority_model.md`
- `reports/recall_balanced_best_eval.csv`
- `reports/recall_balanced_best_class_report.csv`
- `reports/recall_balanced_grid_search.csv`
- `reports/recall_balanced_comparison.csv`
- `reports/per_class_error_analysis_recall_balanced.md`

