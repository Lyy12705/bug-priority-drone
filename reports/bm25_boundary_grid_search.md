# BM25 / BM25F Boundary Grid Search

This experiment tests a small set of BM25F field-weight and normalization settings.

Each config retrains REP- weights from duplicate links, rebuilds improved DRONE/REP- features, then evaluates the boundary-refined model.

## Best Config

- Config: `summary_high`
- Accuracy: `0.7116`
- Macro F1: `0.7108`
- P2 recall: `0.6515`
- MAE: `0.4302`

## All Configs

| config_name          |   refined_accuracy |   refined_macro_f1 |   refined_p1_recall |   refined_p2_recall |   refined_p3_recall |   refined_mae |   rep_product_weight |   rep_component_weight |   rep_severity_weight |
|:---------------------|-------------------:|-------------------:|--------------------:|--------------------:|--------------------:|--------------:|---------------------:|-----------------------:|----------------------:|
| summary_high         |             0.7116 |             0.7108 |              0.6734 |              0.6515 |               0.84  |        0.4302 |               2.3259 |                 0.9431 |                0.1775 |
| default              |             0.7116 |             0.7106 |              0.6683 |              0.6364 |               0.85  |        0.4291 |               2.3854 |                 1.0108 |                0.2904 |
| description_balanced |             0.7116 |             0.7106 |              0.6683 |              0.6313 |               0.85  |        0.4281 |               2.5432 |                 1.1662 |                0.4445 |
| soft_norm            |             0.7065 |             0.7058 |              0.6935 |              0.596  |               0.825 |        0.4302 |               2.4054 |                 1.0726 |                0.3512 |
| strong_norm          |             0.7065 |             0.7052 |              0.6533 |              0.6313 |               0.85  |        0.4281 |               2.3455 |                 0.9452 |                0.2388 |
