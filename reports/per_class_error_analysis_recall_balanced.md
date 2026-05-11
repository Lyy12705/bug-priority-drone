# Per-Class Error Analysis

This report answers where each true priority class is predicted when the model is wrong.

## Overall Metrics

|   rows |   removed_training_overlap |   accuracy |   macro_f1 |   off_by_one_accuracy |   mae |
|-------:|---------------------------:|-----------:|-----------:|----------------------:|------:|
|    995 |                          0 |     0.7317 |     0.7309 |                0.9025 |   0.4 |

## P1-P5 被錯判到哪裡

| true_priority   |   support |   correct |   wrong |   recall | most_common_wrong_prediction   |   most_common_wrong_count |   predicted_P1 |   predicted_P2 |   predicted_P3 |   predicted_P4 |   predicted_P5 |
|:----------------|----------:|----------:|--------:|---------:|:-------------------------------|--------------------------:|---------------:|---------------:|---------------:|---------------:|---------------:|
| P1 Blocker      |       199 |       131 |      68 |   0.6583 | P2 Critical                    |                        50 |            131 |             50 |             10 |              5 |              3 |
| P2 Critical     |       198 |       135 |      63 |   0.6818 | P1 Blocker                     |                        28 |             28 |            135 |             15 |             14 |              6 |
| P3 Major        |       200 |       175 |      25 |   0.875  | P4 Minor                       |                        15 |              4 |              0 |            175 |             15 |              6 |
| P4 Minor        |       199 |       141 |      58 |   0.7085 | P3 Major                       |                        19 |              7 |             18 |             19 |            141 |             14 |
| P5 Trivial      |       199 |       146 |      53 |   0.7337 | P4 Minor                       |                        29 |              2 |              6 |             16 |             29 |            146 |

## Wrong-Prediction Destinations

| true_priority   | predicted_priority   |   count |   percent_of_true_class |   percent_of_wrong_errors |
|:----------------|:---------------------|--------:|------------------------:|--------------------------:|
| P1 Blocker      | P2 Critical          |      50 |                  0.2513 |                    0.7353 |
| P1 Blocker      | P3 Major             |      10 |                  0.0503 |                    0.1471 |
| P1 Blocker      | P4 Minor             |       5 |                  0.0251 |                    0.0735 |
| P1 Blocker      | P5 Trivial           |       3 |                  0.0151 |                    0.0441 |
| P2 Critical     | P1 Blocker           |      28 |                  0.1414 |                    0.4444 |
| P2 Critical     | P3 Major             |      15 |                  0.0758 |                    0.2381 |
| P2 Critical     | P4 Minor             |      14 |                  0.0707 |                    0.2222 |
| P2 Critical     | P5 Trivial           |       6 |                  0.0303 |                    0.0952 |
| P3 Major        | P4 Minor             |      15 |                  0.075  |                    0.6    |
| P3 Major        | P5 Trivial           |       6 |                  0.03   |                    0.24   |
| P3 Major        | P1 Blocker           |       4 |                  0.02   |                    0.16   |
| P4 Minor        | P3 Major             |      19 |                  0.0955 |                    0.3276 |
| P4 Minor        | P2 Critical          |      18 |                  0.0905 |                    0.3103 |
| P4 Minor        | P5 Trivial           |      14 |                  0.0704 |                    0.2414 |
| P4 Minor        | P1 Blocker           |       7 |                  0.0352 |                    0.1207 |
| P5 Trivial      | P4 Minor             |      29 |                  0.1457 |                    0.5472 |
| P5 Trivial      | P3 Major             |      16 |                  0.0804 |                    0.3019 |
| P5 Trivial      | P2 Critical          |       6 |                  0.0302 |                    0.1132 |
| P5 Trivial      | P1 Blocker           |       2 |                  0.0101 |                    0.0377 |

## Confusion Matrix

Rows are true priorities and columns are predicted priorities.

| true_priority   |   P1 Blocker |   P2 Critical |   P3 Major |   P4 Minor |   P5 Trivial |
|:----------------|-------------:|--------------:|-----------:|-----------:|-------------:|
| P1 Blocker      |          131 |            50 |         10 |          5 |            3 |
| P2 Critical     |           28 |           135 |         15 |         14 |            6 |
| P3 Major        |            4 |             0 |        175 |         15 |            6 |
| P4 Minor        |            7 |            18 |         19 |        141 |           14 |
| P5 Trivial      |            2 |             6 |         16 |         29 |          146 |

## Interpretation

- `recall` 表示該 priority 的真實資料中，有多少比例被模型正確抓到。
- `most_common_wrong_prediction` 表示該 priority 最常被錯判成哪一類。
- 如果錯誤集中在相鄰 priority，例如 P2 -> P1 / P3，代表主要問題是邊界模糊。
- 如果錯誤跨很多級，例如 P1 -> P5，才代表模型有較嚴重的排序判斷問題。
