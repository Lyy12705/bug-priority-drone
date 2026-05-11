# Literature-Style GRAY And REP- Implementation

本文件說明目前專案中「貼近文獻版本」的 DRONE/GRAY 實作範圍、執行方式、結果與限制。主要依據為 Tian et al. 的 DRONE/GRAY priority prediction，以及 Sun et al. 的 REP duplicate-report retrieval。

## Implemented Components

| Literature part | Current implementation |
|---|---|
| DRONE six factors | textual、temporal、author、related-report、severity、product/component |
| Textual factor | `paper_tf` 使用 description/summary 清理後的 TF count，不加入 TF-IDF |
| REP- related-report factor | `rep_minus` 使用 unigram BM25Fext、bigram BM25Fext、product match、component match |
| REP vs REP- | 已移除 priority、type、version，只保留 DRONE 論文中的 REP- 欄位 |
| GRAY classification engine | `LinearRegression` 先輸出 ordinal score，再用 4 個 thresholds 對應 P1-P5 |
| Threshold calibration | 依 validation priority 比例初始化 thresholds，再 greedy hill climbing 最大化 average F-measure / macro F1 |
| Chronological setup | `build_literature_splits.py` 依 `creation_time` 建立 model-building、validation、test |
| Duplicate-based REP training hook | `train_rep_minus_weights.py` 可由 `dupe_of` 建立 ranking pairs，輸出 REP- weights JSON |

## REP- Details

REP- 在 `scripts/build_features.py` 中對每個 bug report 只比較「更早建立」的歷史 reports，避免使用未來資訊。相似度為：

```text
REP-(d, q) =
  w1 * BM25Fext_unigram(d, q)
  + w2 * BM25Fext_bigram(d, q)
  + w3 * same_product(d, q)
  + w4 * same_component(d, q)
```

可調參數包含：

- REP feature weights：`--rep-unigram-feature-weight`, `--rep-bigram-feature-weight`, `--rep-product-weight`, `--rep-component-weight`
- summary/description 欄位權重：`--rep-summary-weight`, `--rep-description-weight`, `--rep-bigram-summary-weight`, `--rep-bigram-description-weight`
- BM25Fext 參數：`--rep-k1-*`, `--rep-k3-*`, `--rep-summary-b-*`, `--rep-description-b-*`
- 已訓練權重：`--rep-weights-json models/rep_minus_weights.json`

若重新抓資料並保留 `dupe_of`，可執行：

```bash
python3 scripts/train_rep_minus_weights.py \
  --input data/processed/eclipse_bug_reports_clean_with_dupe.csv \
  --output models/rep_minus_weights.json
```

目前 with-dupe 資料已訓練出 `models/rep_minus_weights.json`，使用 1780 個 duplicate ranking pairs。權重如下：

| REP- feature | Weight |
|---|---:|
| unigram BM25Fext | 0.2370 |
| bigram BM25Fext | 0.0138 |
| same product | 2.3982 |
| same component | 1.0230 |

這代表在目前 Eclipse sample 中，duplicate retrieval 對 product/component 欄位非常敏感，文字 similarity 仍有作用但權重較小。

## Balanced Literature Run

這組結果使用既有 balanced split，適合和目前專題模型比較：

```bash
python3 scripts/build_features.py \
  --input data/processed/experiment_splits/train_balanced_clean.csv \
  --feature-dir data/processed/features_literature_balanced_train \
  --mode fit \
  --text-feature-mode paper_tf \
  --related-mode rep_minus

python3 scripts/train_literature_gray.py \
  --feature-dir data/processed/features_literature_balanced_train \
  --calibration-feature-dir data/processed/features_literature_balanced_validation \
  --model-path models/literature_gray_rep_minus_model.joblib \
  --report-path reports/literature_gray_rep_minus_report.csv
```

| Eval set | Accuracy | Macro F1 | Off-by-one | P1 recall | P2 recall | P3 recall | P4 recall | P5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.4632 | 0.4633 | 0.8400 | 0.4440 | 0.5280 | 0.4400 | 0.3280 | 0.5760 |
| balanced test | 0.4448 | 0.4452 | 0.8320 | 0.3840 | 0.4800 | 0.4240 | 0.3720 | 0.5640 |
| natural holdout | 0.4728 | 0.4734 | 0.8421 | 0.5126 | 0.5076 | 0.3650 | 0.3769 | 0.6030 |

P2 error analysis:

```bash
reports/p2_error_analysis_literature_gray_rep_minus_balanced.csv
```

此檔包含 true P2 但被預測為 P1 或 P3 的 92 筆資料，可用於報告分析 P2 邊界模糊問題。

## Duplicate-Trained REP- Run

這組結果使用重新抓取的 with-dupe raw data、重新清洗與重新切分資料，並套用 `models/rep_minus_weights.json`：

```bash
python3 scripts/build_experiment_splits.py \
  --input data/processed/eclipse_bug_reports_clean_with_dupe.csv \
  --output-dir data/processed/experiment_splits_with_dupe \
  --natural-test-size 0.1 \
  --balanced-train-per-class 1200 \
  --balanced-validation-per-class 250 \
  --balanced-test-per-class 250
```

| Eval set | Accuracy | Macro F1 | Off-by-one | P1 recall | P2 recall | P3 recall | P4 recall | P5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.4544 | 0.4566 | 0.8560 | 0.4840 | 0.4520 | 0.4040 | 0.3640 | 0.5680 |
| balanced test | 0.4568 | 0.4565 | 0.8504 | 0.4640 | 0.4360 | 0.4200 | 0.3520 | 0.6120 |
| natural holdout | 0.4513 | 0.4548 | 0.8392 | 0.4874 | 0.4242 | 0.4050 | 0.3869 | 0.5528 |

Outputs:

- Model：`models/literature_gray_dupe_rep_model.joblib`
- Balanced eval：`reports/balanced_distribution_eval_literature_gray_dupe_rep.csv`
- Natural eval：`reports/natural_distribution_eval_literature_gray_dupe_rep.csv`
- P2 error analysis：`reports/p2_error_analysis_literature_gray_dupe_rep_balanced.csv`

與預設 REP- 權重相比，duplicate-trained REP- 的 balanced macro F1 從 0.4452 提升到 0.4565，off-by-one accuracy 從 0.8320 提升到 0.8504；但 P2 recall 從 0.4800 降到 0.4360。這代表 duplicate-trained REP- 讓整體序位預測更穩，但 P2 邊界仍需要靠 threshold profile 或 hierarchical splitter 進一步改善。

## Hierarchical DRONE/GRAY On Duplicate-Trained REP-

這組結果使用同一批 `features_literature_dupe_*`，也就是 TF-only textual factor 加 duplicate-trained REP- related-report factor。差別是 classification engine 從單層 GRAY 改為 hierarchical DRONE/GRAY：

1. 第一層：`P1/P2`、`P3`、`P4/P5`
2. 第二層：`P1` vs `P2`
3. 第二層：`P4` vs `P5`

```bash
python3 scripts/train_hierarchical_drone_gray.py \
  --feature-dir data/processed/features_literature_dupe_train \
  --calibration-feature-dir data/processed/features_literature_dupe_validation \
  --model-type ridge \
  --ridge-alpha 1.0 \
  --model-path models/hierarchical_literature_dupe_rep_model.joblib \
  --report-path reports/hierarchical_literature_dupe_rep_report.csv
```

| Eval set | Accuracy | Macro F1 | Off-by-one | P1 recall | P2 recall | P3 recall | P4 recall | P5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.5040 | 0.5020 | 0.8456 | 0.3040 | 0.6040 | 0.6280 | 0.5160 | 0.4680 |
| balanced test | 0.5032 | 0.4983 | 0.8720 | 0.2760 | 0.5800 | 0.6240 | 0.5680 | 0.4680 |
| natural holdout | 0.4864 | 0.4867 | 0.8482 | 0.3367 | 0.5606 | 0.5900 | 0.4824 | 0.4623 |

Outputs:

- Model：`models/hierarchical_literature_dupe_rep_model.joblib`
- Balanced eval：`reports/balanced_distribution_eval_hierarchical_literature_dupe_rep.csv`
- Natural eval：`reports/natural_distribution_eval_hierarchical_literature_dupe_rep.csv`
- P2 error analysis：`reports/p2_error_analysis_hierarchical_literature_dupe_rep_balanced.csv`

與單層 duplicate-trained GRAY 相比，hierarchical model 的 balanced macro F1 從 0.4565 提升到 0.4983，accuracy 從 0.4568 提升到 0.5032，P2 recall 從 0.4360 提升到 0.5800，P2 邊界錯誤數從 105 筆降到 87 筆。代價是 P1 recall 從 0.4640 降到 0.2760，表示高優先級 splitter 更偏向把邊界案例判為 P2。

## Accuracy Improvement Experiments

後續實驗保留 DRONE/REP 特徵，但調整 classification engine 或特徵表示。

### Tuned Hierarchical Objective

調整 high-priority splitter 的 objective，降低 P2 recall 權重並補回 P1 recall。Balanced test 結果：

| Model | Accuracy | Macro F1 | Off-by-one | P1 recall | P2 recall |
|---|---:|---:|---:|---:|---:|
| Original hierarchical | 0.5032 | 0.4983 | 0.8720 | 0.2760 | 0.5800 |
| Tuned hierarchical | 0.5112 | 0.5113 | 0.8640 | 0.5160 | 0.3800 |

此設定讓整體 accuracy 與 macro F1 小幅提升，也解決 P1 collapse；但 P2 recall 明顯下降。

### Direct Classifier Engine

文獻 GRAY 是合理 baseline，但把同一批 DRONE/REP features 餵給直接分類器後，效果大幅提升。最佳 literature-feature classifier 是 `SGDClassifier(loss="log_loss", alpha=0.01)`。

| Model | Eval set | Accuracy | Macro F1 | Off-by-one | P1 recall | P2 recall | P3 recall | P4 recall | P5 recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct classifier + duplicate REP- | balanced test | 0.7192 | 0.7185 | 0.9064 | 0.6680 | 0.6440 | 0.8840 | 0.6640 | 0.7360 |
| Direct classifier + duplicate REP- | natural holdout | 0.7015 | 0.7008 | 0.8985 | 0.6734 | 0.5909 | 0.8250 | 0.6633 | 0.7538 |

Outputs:

- Model：`models/direct_classifier_literature_dupe_model.joblib`
- Balanced eval：`reports/balanced_distribution_eval_direct_classifier_literature_dupe.csv`
- Natural eval：`reports/natural_distribution_eval_direct_classifier_literature_dupe.csv`
- P2 error analysis：`reports/p2_error_analysis_direct_classifier_literature_dupe_balanced.csv`

### Enhanced Text Features

Enhanced features 加入 summary/description 分開的 TF-IDF、char n-gram，以及 error-driven binary features。使用同樣的 `SGDClassifier(loss="log_loss", alpha=0.01)`：

| Model | Eval set | Accuracy | Macro F1 | Off-by-one | P1 recall | P2 recall | P3 recall | P4 recall | P5 recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Enhanced direct classifier | balanced test | 0.7104 | 0.7105 | 0.9112 | 0.5800 | 0.6840 | 0.8560 | 0.7080 | 0.7240 |
| Enhanced direct classifier | natural holdout | 0.7015 | 0.7016 | 0.9095 | 0.5930 | 0.6616 | 0.8050 | 0.6935 | 0.7538 |

Enhanced 版本的 balanced accuracy 略低於 literature-feature direct classifier，但 P2 recall 和 off-by-one accuracy 較高。若專題主軸是「提高整體準確率」，建議以 direct classifier + duplicate REP- 作為 final model；若主軸是「改善 P2」，可把 enhanced direct classifier 作為補充結果。

## Chronological Literature Run

這組結果使用文獻式 chronological split：

```bash
python3 scripts/build_literature_splits.py \
  --input data/processed/eclipse_bug_reports_clean.csv \
  --output-dir data/processed/literature_splits
```

目前切分分布：

| split | rows | P1 | P2 | P3 | P4 | P5 |
|---|---:|---:|---:|---:|---:|---:|
| model_building | 2483 | 127 | 179 | 1778 | 224 | 175 |
| validation | 2483 | 637 | 701 | 219 | 542 | 384 |
| test | 4966 | 1224 | 1095 | 0 | 1221 | 1426 |

因 test set 沒有 P3，這組結果只能驗證文獻流程可執行，不適合當作五分類最終成績：

| Eval set | Accuracy | Macro F1 | Off-by-one | P1 recall | P2 recall | P3 recall | P4 recall | P5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chronological validation | 0.3306 | 0.2994 | 0.6895 | 0.4411 | 0.3138 | 0.1142 | 0.3266 | 0.3073 |
| chronological test | 0.2753 | 0.2074 | 0.5868 | 0.5948 | 0.2457 | 0.0000 | 0.1835 | 0.1024 |

## Important Limitations

- 目前資料是 P1-P5 各抓 2000 筆的 quota-balanced dataset，不是文獻使用的完整 chronological Eclipse defect stream。
- 因 quota fetch 造成各 priority 的時間分布不同，P3 幾乎集中在 2001-10-11，導致 chronological test 沒有 P3。
- `train_rep_minus_weights.py` 目前訓練 REP- 四個 linear feature weights；BM25Fext 內部 12 個 free parameters 已開放在 JSON/CLI 中，但尚未完整做文獻中的全參數 gradient descent。
- 文獻的 First/Assigned scenarios 需要 priority modification history，目前 Eclipse REST raw fetch 尚未重建這兩種情境。

## References

- DRONE/GRAY：Tian et al., `Automated Prediction of Bug Report Priority Using Multi-Factor Analysis`，本機 PDF 位於 `/Users/linyaying/Downloads/大專生計畫參考/文獻/Automated Prediction of Bug Report Priority Using Multi-Factor An.pdf`，出版頁：[SMU ScholarBank](https://ink.library.smu.edu.sg/sis_research/2437/)。
- REP：Sun et al., `Towards More Accurate Retrieval of Duplicate Bug Reports`，方法摘要與出版資訊：[SWAG Lab](https://swag.uwaterloo.ca/publications/towards-more-accurate-retrieval-of-duplicate-bug-reports.html)，PDF mirror：[CiteSeerX](https://citeseerx.ist.psu.edu/document?doi=b8570c6d9f9670f4a864b275f021af9c404e3070&repid=rep1&type=pdf)。
