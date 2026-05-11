# Recall-Balanced Priority Model

This experiment adds two local correction layers on top of the base DRONE/REP- classifier:

1. P1/P2 boundary classifier to recover P1 recall.
2. P4 false-high suppression to reduce P4 cases incorrectly predicted as P2.

Selection uses validation-set `macro recall + minimum recall + macro F1`, with MAE and low-recall penalties.

## Best Natural-Holdout Result

| model_stage          | eval_set        |   accuracy |   macro_f1 |   macro_recall |   min_recall |   off_by_one_accuracy |   mae |   p1_recall |   p2_recall |   p3_recall |   p4_recall |   p5_recall |
|:---------------------|:----------------|-----------:|-----------:|---------------:|-------------:|----------------------:|------:|------------:|------------:|------------:|------------:|------------:|
| recall_balanced_best | natural_holdout |     0.7317 |     0.7309 |         0.7315 |       0.6583 |                0.9025 |   0.4 |      0.6583 |      0.6818 |       0.875 |      0.7085 |      0.7337 |

## Top Validation Candidates

| stage               | base_model_type   | base_params   | p1p2_model_type   | p1p2_params   |   p1_sample_weight | p2p4_model_type   | p2p4_params   |   p4_sample_weight |   validation_objective |   validation_macro_recall |   validation_min_recall |   validation_p1_recall |   validation_p2_recall |   validation_p4_recall |   natural_accuracy |   natural_macro_recall |   natural_min_recall |
|:--------------------|:------------------|:--------------|:------------------|:--------------|-------------------:|:------------------|:--------------|-------------------:|-----------------------:|--------------------------:|------------------------:|-----------------------:|-----------------------:|-----------------------:|-------------------:|-----------------------:|---------------------:|
| p1p2_p4_suppression | sgd_log           | alpha=0.05    | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.05    |                1   |                 1.615  |                    0.6952 |                   0.624 |                  0.624 |                  0.624 |                  0.628 |             0.7317 |                 0.7315 |               0.6583 |
| p1p2_p4_suppression | sgd_log           | alpha=0.05    | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.05    |                1.3 |                 1.613  |                    0.6952 |                   0.62  |                  0.624 |                  0.62  |                  0.632 |             0.7317 |                 0.7315 |               0.6583 |
| p1p2_p4_suppression | sgd_log           | alpha=0.05    | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.1     |                1.3 |                 1.613  |                    0.6952 |                   0.62  |                  0.624 |                  0.62  |                  0.632 |             0.7307 |                 0.7305 |               0.6583 |
| p1p2_p4_suppression | sgd_log           | alpha=0.05    | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.1     |                1   |                 1.6129 |                    0.6944 |                   0.624 |                  0.624 |                  0.624 |                  0.624 |             0.7317 |                 0.7315 |               0.6583 |
| p1p2_p4_suppression | sgd_log           | alpha=0.05    | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.03    |                1   |                 1.6111 |                    0.6944 |                   0.62  |                  0.624 |                  0.62  |                  0.628 |             0.7307 |                 0.7304 |               0.6583 |
| p1p2_p4_suppression | sgd_log           | alpha=0.05    | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.1     |                1.6 |                 1.6059 |                    0.6944 |                   0.616 |                  0.624 |                  0.616 |                  0.632 |             0.7317 |                 0.7315 |               0.6583 |
| p1p2_p4_suppression | sgd_log           | alpha=0.1     | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.05    |                1   |                 1.6056 |                    0.6944 |                   0.616 |                  0.624 |                  0.616 |                  0.624 |             0.7276 |                 0.7274 |               0.6533 |
| p1p2_boundary       | sgd_log           | alpha=0.05    | sgd_log           | alpha=0.1     |                  1 |                   |               |                1   |                 1.6053 |                    0.6944 |                   0.616 |                  0.624 |                  0.632 |                  0.616 |             0.7307 |                 0.7305 |               0.6583 |
| p1p2_p4_suppression | sgd_log           | alpha=0.1     | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.1     |                1   |                 1.6034 |                    0.6936 |                   0.616 |                  0.624 |                  0.616 |                  0.62  |             0.7276 |                 0.7274 |               0.6533 |
| p1p2_boundary       | sgd_log           | alpha=0.05    | sgd_log           | alpha=0.05    |                  1 |                   |               |                1   |                 1.6023 |                    0.6928 |                   0.616 |                  0.628 |                  0.62  |                  0.616 |             0.7256 |                 0.7254 |               0.6533 |
| p1p2_p4_suppression | sgd_log           | alpha=0.1     | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.1     |                1.3 |                 1.6007 |                    0.6944 |                   0.612 |                  0.624 |                  0.612 |                  0.628 |             0.7266 |                 0.7264 |               0.6533 |
| p1p2_p4_suppression | sgd_log           | alpha=0.1     | sgd_log           | alpha=0.1     |                  1 | sgd_log           | alpha=0.05    |                1.3 |                 1.6007 |                    0.6944 |                   0.612 |                  0.624 |                  0.612 |                  0.628 |             0.7276 |                 0.7274 |               0.6533 |

## Confusion Matrix

Rows are true P1-P5; columns are predicted P1-P5.

```text
[[131  50  10   5   3]
 [ 28 135  15  14   6]
 [  4   0 175  15   6]
 [  7  18  19 141  14]
 [  2   6  16  29 146]]
```

## Interpretation

- This model should be compared with the current best model before replacing it.
- If macro recall or minimum recall improves but accuracy drops, keep it as an improvement experiment rather than the main model.
