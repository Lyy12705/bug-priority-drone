# Eclipse Bug Priority DRONE/GRAY v2

This project implements Eclipse Bugzilla bug-priority prediction based on the DRONE/GRAY paper idea, then extends it with duplicate-trained REP- related-report features, BM25/BM25F tuning, and a P1/P2/P3 boundary refiner.

The compact workspace now keeps the current best reproducible model only. Markdown reports are preserved for reporting and explanation.

## Current Best Model

| Item | Path |
|---|---|
| Best model | `models/improved_priority_p2_keywords_model.joblib` |
| REP- weights | `models/rep_minus_weights_bm25_p2_error_keywords.json` |
| Train features | `data/processed/features_bm25_p2_error_keywords_train` |
| Validation features | `data/processed/features_bm25_p2_error_keywords_validation` |
| Natural holdout features | `data/processed/features_bm25_p2_error_keywords_natural_test` |
| Main evaluation | `reports/improved_priority_p2_keywords_eval.csv` |
| P1-P5 classification report | `reports/improved_priority_p2_keywords_class_report.csv` |
| Improvement summary | `reports/improved_priority_experiment_summary.md` |

## Current Result

Natural holdout evaluation:

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

## View Results

```bash
cd /Users/linyaying/Documents/bug-priority-drone-v2

cat reports/improved_priority_p2_keywords_eval.csv
cat reports/improved_priority_p2_keywords_class_report.csv
cat reports/improved_priority_p2_keywords_grid_search.md
cat reports/improved_priority_experiment_summary.md
```

Readable metric table:

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

## Reproduce Current Best Result

### 1. Fetch Eclipse Bugzilla data

The fetcher keeps `dupe_of`, which is needed for REP- training.

```bash
python3 scripts/fetch_eclipse_bugzilla.py \
  --target-per-priority 2000 \
  --output data/raw/eclipse_bugzilla_raw_with_dupe.csv
```

### 2. Clean data

```bash
python3 scripts/clean_bug_reports.py \
  --input data/raw/eclipse_bugzilla_raw_with_dupe.csv \
  --output data/processed/eclipse_bug_reports_clean_with_dupe.csv
```

Cleaning rules are documented in `reports/cleaning_rules.md`.

### 3. Build train / validation / natural holdout split

```bash
python3 scripts/build_experiment_splits.py \
  --input data/processed/eclipse_bug_reports_clean_with_dupe.csv \
  --output-dir data/processed/experiment_splits_with_dupe \
  --natural-test-size 0.1 \
  --balanced-train-per-class 1200 \
  --balanced-validation-per-class 250 \
  --balanced-test-per-class 250
```

### 4. Train duplicate-based REP- weights

This version is the best BM25/BM25F setting found in the experiments. Summary text is weighted higher than description text.

```bash
python3 scripts/train_rep_minus_weights.py \
  --input data/processed/eclipse_bug_reports_clean_with_dupe.csv \
  --output models/rep_minus_weights_bm25_summary_high.json \
  --summary-weight 3.0 \
  --description-weight 0.8 \
  --bigram-summary-weight 3.0 \
  --bigram-description-weight 0.8 \
  --k1-unigram 1.2 \
  --k1-bigram 1.2 \
  --summary-b-unigram 0.65 \
  --summary-b-bigram 0.65 \
  --description-b-unigram 0.85 \
  --description-b-bigram 0.85
```

### 5. Build BM25 summary-high features

Train:

```bash
python3 scripts/build_features.py \
  --input data/processed/experiment_splits_with_dupe/train_balanced_clean.csv \
  --feature-dir data/processed/features_bm25_summary_high_train \
  --mode fit \
  --text-feature-mode enhanced \
  --related-mode rep_minus \
  --rep-weights-json models/rep_minus_weights_bm25_summary_high.json
```

Validation:

```bash
python3 scripts/build_features.py \
  --input data/processed/experiment_splits_with_dupe/validation_balanced_clean.csv \
  --feature-dir data/processed/features_bm25_summary_high_validation \
  --mode transform \
  --reference-feature-dir data/processed/features_bm25_summary_high_train \
  --history-input data/processed/experiment_splits_with_dupe/train_balanced_clean.csv \
  --text-feature-mode enhanced \
  --related-mode rep_minus \
  --rep-weights-json models/rep_minus_weights_bm25_summary_high.json
```

Natural holdout:

```bash
python3 scripts/build_features.py \
  --input data/processed/experiment_splits_with_dupe/natural_test_clean.csv \
  --feature-dir data/processed/features_bm25_summary_high_natural_test \
  --mode transform \
  --reference-feature-dir data/processed/features_bm25_summary_high_train \
  --history-input data/processed/experiment_splits_with_dupe/train_balanced_clean.csv \
  --text-feature-mode enhanced \
  --related-mode rep_minus \
  --rep-weights-json models/rep_minus_weights_bm25_summary_high.json
```

### 6. Train and evaluate best boundary model

Use existing best feature directories:

```bash
python3 scripts/grid_search_bm25_boundary.py \
  --config-names summary_high \
  --skip-feature-build
```

This writes:

- `models/bm25_boundary_best_model.joblib`
- `reports/bm25_boundary_best_eval.csv`
- `reports/bm25_boundary_best_class_report.csv`
- `reports/bm25_boundary_grid_search.csv`
- `reports/bm25_boundary_grid_search.md`

### 7. Generate P2 error analysis

```bash
python3 scripts/analyze_priority_errors.py \
  --model models/bm25_boundary_best_model.joblib \
  --feature-dir data/processed/features_bm25_summary_high_natural_test \
  --train-feature-dir data/processed/features_bm25_summary_high_train \
  --true-priorities 2 \
  --predicted-priorities 1,3 \
  --output reports/p2_error_analysis_bm25_boundary_best_natural.csv \
  --summary-output reports/p2_error_analysis_bm25_boundary_best_summary.md
```

```bash
python3 scripts/categorize_p2_boundary_errors.py \
  --error-csv reports/p2_error_analysis_bm25_boundary_best_natural.csv \
  --feature-meta data/processed/features_bm25_summary_high_natural_test/feature_meta.csv \
  --output-csv reports/p2_error_manual_categories_bm25_boundary_best.csv \
  --summary-md reports/p2_error_manual_categories_bm25_boundary_best.md \
  --model-name bm25_boundary_best
```

## Project Flow

1. Fetch Eclipse Bugzilla data with priority labels and `dupe_of`.
2. Clean summary, first-comment description, category fields, and creation time.
3. Split data into balanced training, balanced validation, and natural holdout.
4. Build DRONE-style six-factor features: textual, temporal, author, related-report, severity, and product/component.
5. Train REP- weights from duplicate links to improve related-report similarity.
6. Tune BM25/BM25F field weights and train a boundary-refined classifier.
7. Evaluate on natural holdout and report accuracy, macro F1, off-by-one accuracy, MAE, and P1-P5 recall.

## Main Report Files

- `reports/cleaning_rules.md`: data-cleaning and preprocessing rules.
- `reports/literature_implementation.md`: how the implementation follows and extends DRONE/GRAY.
- `reports/model_result_terminal_steps.md`: terminal commands for viewing results.
- `reports/improved_priority_experiment_summary.md`: latest improvement result and comparison.
- `reports/bm25_boundary_grid_search.md`: previous BM25/BM25F tuning result.
- `reports/p2_error_analysis_bm25_boundary_best_summary.md`: P2 boundary error summary.
- `reports/p2_error_manual_categories_bm25_boundary_best.md`: manual-style categories for remaining P2 errors.
