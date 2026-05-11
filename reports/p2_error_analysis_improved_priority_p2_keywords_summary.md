# Boundary Error Analysis Summary

- True labels inspected: `[2]`
- Predicted labels selected: `[1, 3]`
- True-label rows: `198`
- Selected boundary errors: `45`

## Predicted Label Counts

- `P1 Blocker`: 28
- `P3 Major`: 17

## Top Products In Selected Errors

- `jdt`: 22
- `platform`: 21
- `pde`: 2

## Top Components In Selected Errors

- `ui`: 16
- `debug`: 9
- `core`: 9
- `update  (deprecated - use eclipse>equinox>p2)`: 4
- `ant`: 2
- `compare`: 1
- `user assistance`: 1
- `resources`: 1

## Top Severities In Selected Errors

- `normal`: 29
- `major`: 12
- `enhancement`: 3
- `minor`: 1

## Keyword Signals With Higher Error Rates

| keyword_feature                   |   selected_error_rate |   true_label_rate |   difference |
|:----------------------------------|----------------------:|------------------:|-------------:|
| has_file_line_stack_terms_count   |               11.6667 |            2.8535 |       8.8131 |
| has_stack_trace_count             |                3.7111 |            1.0404 |       2.6707 |
| has_compiler_terms_count          |                1.2444 |            0.3232 |       0.9212 |
| has_exception_terms_count         |                0.6444 |            0.1869 |       0.4576 |
| has_workspace_project_terms_count |                0.8000 |            0.3737 |       0.4263 |
| has_install_update_terms_count    |                0.6222 |            0.2980 |       0.3242 |
| has_stack_trace                   |                0.4222 |            0.1212 |       0.3010 |
| has_debug_terms_count             |                0.6222 |            0.3535 |       0.2687 |
| has_exception_terms               |                0.3556 |            0.1010 |       0.2545 |
| has_file_line_stack_terms         |                0.3111 |            0.0859 |       0.2253 |
| has_internal_error_terms_count    |                0.2444 |            0.0556 |       0.1889 |
| has_widget_disposed_terms_count   |                0.2000 |            0.0455 |       0.1545 |
