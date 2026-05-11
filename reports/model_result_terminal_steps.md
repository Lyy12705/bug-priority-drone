# Model Result Terminal Steps

本文件記錄如何在終端機查看目前最佳模型的優先級判定結果。目前最佳模型已更新為 `improved_priority_p2_keywords_model`，它在 BM25/BM25F boundary model 上加入 P2 error-driven keyword features 與 boundary objective 調整。

## 1. 進入專案資料夾

```bash
cd /Users/linyaying/Documents/bug-priority-drone-v2
```

## 2. 查看目前最佳模型結果

```bash
cat reports/improved_priority_p2_keywords_eval.csv
cat reports/improved_priority_p2_keywords_class_report.csv
```

## 3. 用表格方式查看重點指標

```bash
python3 - <<'PY'
import pandas as pd

df = pd.read_csv("reports/improved_priority_p2_keywords_eval.csv")
cols = [
    "accuracy",
    "macro_f1",
    "off_by_one_accuracy",
    "mae",
    "p1_recall",
    "p2_recall",
    "p3_recall",
    "p4_recall",
    "p5_recall",
]
print(df[cols].to_string(index=False))
PY
```

目前 natural holdout 結果：

| Metric | Value |
|---|---:|
| Accuracy | 0.7246 |
| Macro F1 | 0.7240 |
| Off-by-one accuracy | 0.8995 |
| MAE | 0.4090 |
| P1 recall | 0.6482 |
| P2 recall | 0.6818 |
| P3 recall | 0.8750 |
| P4 recall | 0.6834 |
| P5 recall | 0.7337 |

## 4. 查看改良實驗比較結果

```bash
cat reports/improved_priority_experiment_summary.md
cat reports/improved_priority_comparison.csv
cat reports/improved_priority_p2_keywords_grid_search.md
```

## 5. 查看先前 P2 錯誤分析

查看真實 P2 被錯判成 P1 或 P3 的案例摘要：

```bash
cat reports/p2_error_analysis_bm25_boundary_best_summary.md
```

查看人工分類後的錯誤類型：

```bash
cat reports/p2_error_manual_categories_bm25_boundary_best.md
```

## 6. 重新產生目前最佳模型

如果特徵資料已存在，只要重新訓練與評估目前最佳設定：

```bash
python3 scripts/grid_search_bm25_boundary.py \
  --config-names p2_error_keywords \
  --skip-feature-build \
  --model-types sgd_log \
  --alpha-values 0.001,0.01,0.1 \
  --boundary-apply-values base_1_2,base_1_2_3 \
  --p2-weight-values 0.10,0.16 \
  --collapse-floor-values 0.55 \
  --boundary-p2-sample-weights 1.0,1.2 \
  --p1-weight 0.02 \
  --p3-weight 0.01 \
  --off-by-one-weight 0.02 \
  --accuracy-drop-weight 0.25 \
  --collapse-penalty-weight 0.35 \
  --max-accuracy-drop 0.015 \
  --output-csv reports/improved_priority_p2_keywords_grid_search.csv \
  --summary-md reports/improved_priority_p2_keywords_grid_search.md \
  --best-model-path models/improved_priority_p2_keywords_model.joblib \
  --best-eval-csv reports/improved_priority_p2_keywords_eval.csv \
  --best-class-report-csv reports/improved_priority_p2_keywords_class_report.csv
```

## 7. 指標判讀

`accuracy` 是整體完全判對的比例，越高越好。

`macro F1` 是 P1-P5 五個類別各自 F1 的平均，越高代表模型不是只偏向某一類。

`off-by-one accuracy` 是預測差一級也算可接受的比例，例如 P2 判成 P1 或 P3。Priority 本身是序位等級，所以這個指標很適合補充說明。

`MAE` 是預測 priority 與真實 priority 的平均距離，越低越好。

`P1-P5 recall` 代表每個真實 priority 被模型成功抓出的比例。這次最佳模型的 P2 recall 是 `0.6818`，比前一版 `0.6515` 更適合說明 P2 邊界改善。

## 8. 報告建議

報告時可以把目前模型說成：

本研究先依據 DRONE/GRAY 建立六類因素，包含 textual、temporal、author、related-report、severity、product/component。接著利用 Bugzilla 的 `dupe_of` 欄位訓練 REP- related-report 權重，並進一步調整 BM25/BM25F 中 summary 與 description 的欄位權重。最後根據 P2 錯誤分析加入 keyword features，並使用 P1/P2/P3 boundary refiner 改善 P2 容易被判成 P1 或 P3 的問題。

目前最佳模型是 `improved_priority_p2_keywords_model`，在 natural holdout 上 accuracy 為 `0.7246`，macro F1 為 `0.7240`，P2 recall 為 `0.6818`。
