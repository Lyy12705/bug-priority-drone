# Boundary Objective Grid Search

This search changes the objective used to select the P1/P2/P3 boundary classifier.

The base direct classifier is fixed; only boundary objective settings are compared.

## Base Natural Holdout

- Accuracy: `0.6995`
- Macro F1: `0.6990`
- P1 recall: `0.6935`
- P2 recall: `0.5859`
- P3 recall: `0.8150`

## Best Natural Holdout Candidate

- Boundary apply: `base_1_2`
- P2 weight: `0.0`
- Collapse floor: `0.5`
- Boundary classifier: `sgd_log alpha=0.1`
- Accuracy: `0.7116`
- Macro F1: `0.7106`
- P2 recall: `0.6364`

## Top Candidates

|   rank | boundary_apply   |   p2_weight |   collapse_floor | boundary_model_type   | boundary_params   |   natural_accuracy |   natural_macro_f1 |   natural_p1_recall |   natural_p2_recall |   natural_p3_recall |   natural_mae |   validation_score |
|-------:|:-----------------|------------:|-----------------:|:----------------------|:------------------|-------------------:|-------------------:|--------------------:|--------------------:|--------------------:|--------------:|-------------------:|
|      1 | base_1_2         |        0    |             0.5  | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.6746 |
|      2 | base_1_2         |        0    |             0.55 | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.6746 |
|      3 | base_1_2         |        0    |             0.6  | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.6746 |
|      4 | base_1_2         |        0    |             0.65 | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.6746 |
|      5 | base_1_2         |        0.05 |             0.5  | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.7024 |
|      6 | base_1_2         |        0.05 |             0.55 | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.7024 |
|      7 | base_1_2         |        0.05 |             0.6  | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.7024 |
|      8 | base_1_2         |        0.05 |             0.65 | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.7024 |
|      9 | base_1_2         |        0.1  |             0.5  | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.7302 |
|     10 | base_1_2         |        0.1  |             0.55 | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.7302 |
|     11 | base_1_2         |        0.1  |             0.6  | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.7302 |
|     12 | base_1_2         |        0.1  |             0.65 | sgd_log               | alpha=0.1         |             0.7116 |             0.7106 |              0.6683 |              0.6364 |                0.85 |        0.4291 |             0.7302 |
