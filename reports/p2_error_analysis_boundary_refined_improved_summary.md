# Boundary Error Analysis Summary

- True labels inspected: `[2]`
- Predicted labels selected: `[1, 3]`
- True-label rows: `198`
- Selected boundary errors: `53`

## Predicted Label Counts

- `P1 Blocker`: 37
- `P3 Major`: 16

## Top Products In Selected Errors

- `platform`: 27
- `jdt`: 24
- `pde`: 2

## Top Components In Selected Errors

- `ui`: 19
- `debug`: 14
- `core`: 9
- `update  (deprecated - use eclipse>equinox>p2)`: 4
- `ant`: 2
- `compare`: 1
- `resources`: 1
- `build`: 1

## Top Severities In Selected Errors

- `normal`: 35
- `major`: 14
- `enhancement`: 3
- `minor`: 1

## Keyword Signals With Higher Error Rates

| keyword_feature                  |   selected_error_rate |   true_label_rate |   difference |
|:---------------------------------|----------------------:|------------------:|-------------:|
| has_stack_trace                  |                0.3396 |            0.1212 |       0.2184 |
| has_exception_terms              |                0.2830 |            0.1010 |       0.1820 |
| has_install_update_terms         |                0.2264 |            0.1162 |       0.1103 |
| summary_has_exception_terms      |                0.1321 |            0.0404 |       0.0917 |
| summary_has_stack_trace          |                0.0943 |            0.0303 |       0.0640 |
| has_test_failure_terms           |                0.0755 |            0.0455 |       0.0300 |
| has_api_break_terms              |                0.0377 |            0.0101 |       0.0276 |
| has_blocking_terms               |                0.0377 |            0.0152 |       0.0226 |
| summary_has_crash_terms          |                0.0189 |            0.0051 |       0.0138 |
| has_crash_terms                  |                0.0189 |            0.0051 |       0.0138 |
| summary_has_install_update_terms |                0.0566 |            0.0455 |       0.0111 |
| has_data_loss_terms              |                0.0189 |            0.0101 |       0.0088 |
