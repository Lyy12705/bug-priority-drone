# Targeted Error Direction Analysis

- Rows analyzed after overlap removal: `995`
- Removed overlapping training rows: `0`
- Directions: `P1->P2, P4->P2`

## Direction Counts

| error_direction           |   count |   percent |
|:--------------------------|--------:|----------:|
| P1 Blocker -> P2 Critical |      50 |    0.7353 |
| P4 Minor -> P2 Critical   |      18 |    0.2647 |

## P1 Blocker -> P2 Critical

- Error rows: `50`
- True-class rows: `199`

### Severity

| severity    |   count |   percent |
|:------------|--------:|----------:|
| normal      |      40 |      0.8  |
| major       |       6 |      0.12 |
| minor       |       1 |      0.02 |
| critical    |       1 |      0.02 |
| enhancement |       1 |      0.02 |
| blocker     |       1 |      0.02 |

### Product

| product   |   count |   percent |
|:----------|--------:|----------:|
| jdt       |      31 |      0.62 |
| platform  |      16 |      0.32 |
| pde       |       3 |      0.06 |

### Component

| component                                     |   count |   percent |
|:----------------------------------------------|--------:|----------:|
| ui                                            |      29 |      0.58 |
| debug                                         |      11 |      0.22 |
| core                                          |       7 |      0.14 |
| swt                                           |       2 |      0.04 |
| update  (deprecated - use eclipse>equinox>p2) |       1 |      0.02 |

### Signals Compared With True Class

| signal                   |   direction_error_rate |   true_class_reference_rate |   difference |
|:-------------------------|-----------------------:|----------------------------:|-------------:|
| severity_normal_or_lower |                   0.84 |                      0.5678 |       0.2722 |
| component_ui_debug_core  |                   0.94 |                      0.8342 |       0.1058 |
| related_pull_to_low      |                   0.22 |                      0.1206 |       0.0994 |
| low_impact_signal        |                   0.18 |                      0.1156 |       0.0644 |
| high_impact_signal       |                   0.08 |                      0.0955 |      -0.0155 |
| update_install_signal    |                   0.14 |                      0.1608 |      -0.0208 |
| product_platform_or_jdt  |                   0.94 |                      0.9648 |      -0.0248 |
| debug_signal             |                   0.18 |                      0.2312 |      -0.0512 |
| related_pull_to_high     |                   0.62 |                      0.7236 |      -0.1036 |
| stack_exception_signal   |                   0.16 |                      0.4221 |      -0.2621 |
| severity_major_or_higher |                   0.16 |                      0.4322 |      -0.2722 |

### Top Phrases

| phrase        |   count |
|:--------------|--------:|
| java          |     427 |
| code          |     129 |
| compiled      |      88 |
| eclipse       |      88 |
| java compiled |      87 |
| compiled code |      87 |
| org           |      86 |
| org eclipse   |      81 |
| main          |      77 |
| core          |      59 |
| public        |      44 |
| method        |      41 |

## P4 Minor -> P2 Critical

- Error rows: `18`
- True-class rows: `199`

### Severity

| severity    |   count |   percent |
|:------------|--------:|----------:|
| normal      |      16 |    0.8889 |
| major       |       1 |    0.0556 |
| enhancement |       1 |    0.0556 |

### Product

| product   |   count |   percent |
|:----------|--------:|----------:|
| jdt       |      11 |    0.6111 |
| platform  |       7 |    0.3889 |

### Component

| component                                     |   count |   percent |
|:----------------------------------------------|--------:|----------:|
| ui                                            |       7 |    0.3889 |
| core                                          |       5 |    0.2778 |
| swt                                           |       2 |    0.1111 |
| debug                                         |       1 |    0.0556 |
| team                                          |       1 |    0.0556 |
| update  (deprecated - use eclipse>equinox>p2) |       1 |    0.0556 |
| resources                                     |       1 |    0.0556 |

### Signals Compared With True Class

| signal                   |   direction_error_rate |   true_class_reference_rate |   difference |
|:-------------------------|-----------------------:|----------------------------:|-------------:|
| related_pull_to_high     |                 0.3889 |                      0.3216 |       0.0673 |
| product_platform_or_jdt  |                 1      |                      0.9648 |       0.0352 |
| severity_major_or_higher |                 0.0556 |                      0.0352 |       0.0204 |
| component_ui_debug_core  |                 0.7222 |                      0.7186 |       0.0036 |
| update_install_signal    |                 0.1111 |                      0.1256 |      -0.0145 |
| severity_normal_or_lower |                 0.9444 |                      0.9648 |      -0.0204 |
| stack_exception_signal   |                 0.0556 |                      0.0804 |      -0.0248 |
| debug_signal             |                 0      |                      0.0352 |      -0.0352 |
| low_impact_signal        |                 0.1667 |                      0.206  |      -0.0394 |
| high_impact_signal       |                 0      |                      0.0452 |      -0.0452 |
| related_pull_to_low      |                 0.3889 |                      0.4975 |      -0.1086 |

### Top Phrases

| phrase    |   count |
|:----------|--------:|
| view      |      12 |
| error     |      11 |
| class     |       9 |
| composite |       9 |
| new       |       9 |
| does      |       8 |
| swt       |       8 |
| file      |       7 |
| user      |       7 |
| package   |       7 |
| message   |       7 |
| label     |       6 |

## Interpretation

- P1 -> P2 表示高優先級 bug 被模型判太輕，需要 P1/P2 boundary classifier 補回 P1 recall。
- P4 -> P2 表示低優先級 bug 被模型拉太高，需要 false-high suppression 抑制過度預測 P2。
