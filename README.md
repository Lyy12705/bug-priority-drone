# Eclipse Bug Priority DRONE/GRAY 優先級判定專題

本專案實作 Eclipse Bugzilla bug report 的 priority prediction。研究方法以 DRONE/GRAY 文獻為基礎，建立 textual、temporal、author、related-report、severity、product/component 六類特徵，並進一步加入 duplicate-trained REP-、BM25/BM25F 欄位權重調整、P1/P2/P3 boundary classifier，以及 P2 錯誤分析導向的 keyword features。

目前 GitHub 版本保留可重現實驗所需的程式、報告與輕量權重檔。大型資料集、feature matrix 與 `.joblib` 模型檔已透過 `.gitignore` 排除，避免 repository 過大；需要時可依照本文件指令重新產生。

## 目前最佳模型

| 項目 | 路徑 |
|---|---|
| 最佳模型 | `models/improved_priority_p2_keywords_model.joblib` |
| REP- 權重 | `models/rep_minus_weights_bm25_p2_error_keywords.json` |
| 訓練特徵 | `data/processed/features_bm25_p2_error_keywords_train` |
| 驗證特徵 | `data/processed/features_bm25_p2_error_keywords_validation` |
| 最終測試特徵 | `data/processed/features_bm25_p2_error_keywords_natural_test` |
| 主要評估結果 | `reports/improved_priority_p2_keywords_eval.csv` |
| 各類別分類報告 | `reports/improved_priority_p2_keywords_class_report.csv` |
| 改良實驗摘要 | `reports/improved_priority_experiment_summary.md` |

## 目前最佳結果

評估資料集為 `natural_holdout`，也就是不參與訓練與調參的最終測試資料。

| 指標 | 數值 |
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

和前一版最佳模型相比：

| 模型 | Accuracy | Macro F1 | P2 recall | MAE |
|---|---:|---:|---:|---:|
| 舊最佳 BM25 boundary | 0.7116 | 0.7108 | 0.6515 | 0.4302 |
| 新最佳 p2 keywords SGD | 0.7246 | 0.7240 | 0.6818 | 0.4090 |

## 專案流程

1. 從 Eclipse Bugzilla 抓取 P1-P5 bug report，並保留 `dupe_of` 欄位。
2. 清洗 `summary`、第一則 comment 作為 `description`、類別欄位與建立時間。
3. 建立 `train_balanced`、`validation_balanced`、`natural_holdout` 三種資料切分。
4. 建立 DRONE 六類特徵：文字、時間、作者、相關報告、嚴重度、產品/元件。
5. 使用 duplicate bug links 訓練 REP- related-report 權重。
6. 調整 BM25/BM25F summary、description、unigram、bigram 欄位權重。
7. 使用 P1/P2/P3 boundary classifier 改善 P2 容易被判成 P1 或 P3 的問題。
8. 針對 P2 剩餘錯誤加入 keyword features，重新比較模型表現。
9. 在 `natural_holdout` 上報告 accuracy、macro F1、off-by-one accuracy、MAE 與 P1-P5 recall。

## 查看目前結果

```bash
cd /Users/linyaying/Documents/bug-priority-drone-v2

python3 scripts/show_model_metrics.py
```

若要查看原始 CSV 與完整分類報告：

```bash
cat reports/improved_priority_p2_keywords_eval.csv
cat reports/improved_priority_p2_keywords_class_report.csv
cat reports/improved_priority_experiment_summary.md
```

用較好讀的表格查看重點指標：

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

## 重新產生資料與模型

### 1. 抓取 Eclipse Bugzilla 資料

```bash
python3 scripts/fetch_eclipse_bugzilla.py \
  --target-per-priority 2000 \
  --output data/raw/eclipse_bugzilla_raw_with_dupe.csv
```

### 2. 清洗資料

```bash
python3 scripts/clean_bug_reports.py \
  --input data/raw/eclipse_bugzilla_raw_with_dupe.csv \
  --output data/processed/eclipse_bug_reports_clean_with_dupe.csv
```

詳細清洗規則請見 `reports/cleaning_rules.md`。

### 3. 建立資料切分

```bash
python3 scripts/build_experiment_splits.py \
  --input data/processed/eclipse_bug_reports_clean_with_dupe.csv \
  --output-dir data/processed/experiment_splits_with_dupe \
  --natural-test-size 0.1 \
  --balanced-train-per-class 1200 \
  --balanced-validation-per-class 250 \
  --balanced-test-per-class 250
```

### 4. 重新訓練目前最佳改良模型

若特徵資料已存在，可直接重跑目前最佳設定：

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

若要從 REP- 權重與 feature matrix 全部重建，移除 `--skip-feature-build` 即可。

## P2 錯誤分析

改良後，真實 P2 被判成 P1 或 P3 的錯誤由 50 筆降為 45 筆。

```bash
cat reports/p2_error_analysis_improved_priority_p2_keywords_summary.md
cat reports/p2_error_manual_categories_improved_priority_p2_keywords.md
```

目前主要剩餘錯誤類型：

- P2 被判成 P1：stack trace / exception 訊號讓模型判太嚴重。
- P2 被判成 P3：severity 是 normal / enhancement / minor 時容易被判太輕。
- P2 被判成 P1：UI / debug / core 模組歷史訊號偏高。

## 主要文件

- `reports/cleaning_rules.md`：資料清洗與前處理規則。
- `reports/literature_implementation.md`：本專題如何依據與延伸 DRONE/GRAY 文獻。
- `reports/model_result_terminal_steps.md`：查看模型結果的終端機指令。
- `reports/improved_priority_experiment_summary.md`：最新改良實驗與結果比較。
- `reports/p2_error_analysis_improved_priority_p2_keywords_summary.md`：改良後 P2 錯誤摘要。
- `reports/p2_error_manual_categories_improved_priority_p2_keywords.md`：改良後 P2 錯誤人工分類。

## GitHub 版本控制說明

本 repository 不直接追蹤大型資料與模型檔：

- `data/raw/`
- `data/processed/`
- `models/*.joblib`

這些檔案可由上述 pipeline 重新產生。GitHub 上保留的是程式碼、報告、實驗結果 CSV、Markdown 說明與 REP- JSON 權重。
