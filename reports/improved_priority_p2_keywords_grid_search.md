# BM25 / BM25F Boundary Grid Search

This experiment tests a small set of BM25F field-weight and normalization settings.

Each config retrains REP- weights from duplicate links, rebuilds improved DRONE/REP- features, then evaluates the boundary-refined model.

## Best Config

- Config: `p2_error_keywords`
- Accuracy: `0.7246`
- Macro F1: `0.7240`
- P2 recall: `0.6818`
- MAE: `0.4090`

## All Configs

| config_name       | base_model_type   | base_params   | boundary_model_type   | boundary_params   | boundary_apply   |   boundary_p2_sample_weight |   objective_p1_weight |   objective_p2_weight |   objective_p3_weight |   objective_off_by_one_weight |   objective_collapse_floor |   refined_accuracy |   refined_macro_f1 |   refined_p1_recall |   refined_p2_recall |   refined_p3_recall |   refined_mae |   rep_product_weight |   rep_component_weight |   rep_severity_weight |
|:------------------|:------------------|:--------------|:----------------------|:------------------|:-----------------|----------------------------:|----------------------:|----------------------:|----------------------:|------------------------------:|---------------------------:|-------------------:|-------------------:|--------------------:|--------------------:|--------------------:|--------------:|---------------------:|-----------------------:|----------------------:|
| p2_error_keywords | sgd_log           | alpha=0.1     | sgd_log               | alpha=0.1         | base_1_2         |                         1.2 |                  0.02 |                  0.16 |                  0.01 |                          0.02 |                       0.55 |             0.7246 |              0.724 |              0.6482 |              0.6818 |               0.875 |         0.409 |               2.2575 |                 0.8855 |                0.1334 |
