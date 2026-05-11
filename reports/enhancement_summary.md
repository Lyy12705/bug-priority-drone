# DRONE/GRAY Enhancement Summary

本次改善目標是提升 P2 recall，並避免單純提高全域 P2 權重時造成 P1 recall 崩落。

## Implemented Changes

- 調整資料切分：`natural-test-size=0.1`，balanced train 每類 1200 筆，validation/test 每類 250 筆。
- 加強文字特徵：保留文獻式 TF textual factor，新增 TF-IDF unigram/bigram 片語特徵。
- 建立 hierarchical DRONE/GRAY：
  - Stage 1：判斷 `P1/P2`, `P3`, `P4/P5`。
  - Stage 2-high：專門判斷 `P1` vs `P2`。
  - Stage 2-low：專門判斷 `P4` vs `P5`。
- High splitter objective 同時重視 P1 recall 與 P2 recall，並懲罰過度預測 P2。
- 輸出 P2 error analysis：真實 P2 但預測為 P1/P3 的案例。

## Key Results

| Model | Split | Accuracy | Macro F1 | P1 Recall | P2 Recall | Off-by-1 |
|---|---|---:|---:|---:|---:|---:|
| Single GRAY | Balanced | 0.455 | 0.445 | 0.624 | 0.244 | 0.838 |
| Single GRAY + P2 weight | Balanced | 0.454 | 0.446 | 0.600 | 0.268 | 0.840 |
| Hierarchical GRAY | Balanced | 0.458 | 0.461 | 0.408 | 0.396 | 0.826 |
| Single GRAY | Natural holdout | 0.476 | 0.465 | 0.729 | 0.259 | 0.857 |
| Single GRAY + P2 weight | Natural holdout | 0.482 | 0.475 | 0.719 | 0.310 | 0.855 |
| Hierarchical GRAY | Natural holdout | 0.502 | 0.509 | 0.508 | 0.523 | 0.847 |

## Interpretation

Hierarchical DRONE/GRAY gives the clearest P2 recall improvement. On balanced test, P2 recall improves from 0.244 to 0.396. On natural holdout, P2 recall improves from 0.259 to 0.523, while macro F1 also improves from 0.465 to 0.509.

The tradeoff is lower P1 recall because the high-priority splitter now separates some tickets previously predicted as P1 into P2. This is expected and is more balanced than the earlier aggressive threshold-only P2 profile, which recovered P2 by almost collapsing P1.

## Generated Files

- `models/hierarchical_drone_gray_model.joblib`
- `reports/hierarchical_drone_gray_report.csv`
- `reports/balanced_distribution_eval_hierarchical_drone_gray.csv`
- `reports/natural_distribution_eval_hierarchical_drone_gray.csv`
- `reports/enhanced_model_comparison.csv`
- `reports/p2_error_analysis_hierarchical_balanced.csv`
- `reports/p2_error_analysis_hierarchical_natural.csv`
