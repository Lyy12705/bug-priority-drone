# Boundary Error Analysis Summary

- True labels inspected: `[2]`
- Predicted labels selected: `[1, 3]`
- True-label rows: `198`
- Selected boundary errors: `50`

## Predicted Label Counts

- `P1 Blocker`: 34
- `P3 Major`: 16

## Top Products In Selected Errors

- `platform`: 25
- `jdt`: 23
- `pde`: 2

## Top Components In Selected Errors

- `ui`: 18
- `debug`: 13
- `core`: 9
- `update  (deprecated - use eclipse>equinox>p2)`: 4
- `ant`: 2
- `compare`: 1
- `resources`: 1
- `build`: 1

## Top Severities In Selected Errors

- `normal`: 33
- `major`: 13
- `enhancement`: 3
- `minor`: 1

## Keyword Signals With Higher Error Rates

| keyword_feature                  |   selected_error_rate |   true_label_rate |   difference |
|:---------------------------------|----------------------:|------------------:|-------------:|
| has_stack_trace                  |                0.3600 |            0.1212 |       0.2388 |
| has_exception_terms              |                0.3000 |            0.1010 |       0.1990 |
| has_install_update_terms         |                0.2400 |            0.1162 |       0.1238 |
| summary_has_exception_terms      |                0.1400 |            0.0404 |       0.0996 |
| summary_has_stack_trace          |                0.1000 |            0.0303 |       0.0697 |
| has_test_failure_terms           |                0.0800 |            0.0455 |       0.0345 |
| has_api_break_terms              |                0.0400 |            0.0101 |       0.0299 |
| has_blocking_terms               |                0.0400 |            0.0152 |       0.0248 |
| summary_has_crash_terms          |                0.0200 |            0.0051 |       0.0149 |
| has_crash_terms                  |                0.0200 |            0.0051 |       0.0149 |
| summary_has_install_update_terms |                0.0600 |            0.0455 |       0.0145 |
| has_data_loss_terms              |                0.0200 |            0.0101 |       0.0099 |
