# Eclipse Bug Report Cleaning Rules

本文件說明 `scripts/clean_bug_reports.py` 的清洗規則。目標是將 Eclipse Bugzilla 原始資料整理成可供 DRONE/GRAY priority prediction 使用的標準資料表。

## Input And Output

- Input：`data/raw/eclipse_bugzilla_raw_with_dupe.csv`
- Output：`data/processed/eclipse_bug_reports_clean_with_dupe.csv`
- 主要標籤：`priority`
- 模型標籤：`priority_num`
- Description 來源：Eclipse Bugzilla 的第一則 comment，也就是原始 bug report 描述。

## Cleaning Rules

| Rule | Purpose | Implementation |
|---|---|---|
| 1. 保留研究欄位 | 只保留 priority prediction 需要的欄位 | `id`, `summary`, `description`, `priority`, `severity`, `product`, `component`, `creator`, `creation_time`, `status`, `resolution`, `fetch_group`；若 raw data 有 `dupe_of` 也會保留 |
| 2. 移除缺少原始文字的報告 | DRONE textual factor 需要 summary 與第一則 description | 移除空白 `summary` 或空白 `description` |
| 3. 清理文字 | 降低 URL、email、hex、log-like token 對文字特徵的干擾 | `clean_text()` |
| 4. 保留有效 priority | 只保留標準 P1 到 P5 標籤 | `priority in {"P1", "P2", "P3", "P4", "P5"}` |
| 5. 建立 ordinal label | 將 P1 到 P5 轉成 GRAY regression 可用的序位數值 | `P1=1`, `P2=2`, `P3=3`, `P4=4`, `P5=5` |
| 6. 補齊缺失類別欄位 | 避免 feature extraction 或 one-hot encoding 出現缺值錯誤 | 對 `severity`, `product`, `component`, `creator`, `status`, `resolution`, `fetch_group` 補 `unknown` |
| 7. 整併低頻類別 | 降低 creator/product/component 的稀疏程度 | 出現次數小於 `RARE_VALUE_THRESHOLD=5` 的值改為 `other` |
| 8. 解析時間欄位 | temporal factor 需要可靠時間順序 | `creation_time` 轉為 datetime，無法解析者移除 |
| 9. 建立 text 欄位 | DRONE textual factor 使用 summary + description | `text = summary + " " + description` |
| 10. 控制文字長度 | 移除太短資料，截斷極長描述 | `MIN_TEXT_LEN=10`, `MAX_TEXT_LEN=5000` |
| 11. 時間排序與去重 | 避免 temporal/related-report features 使用未來資訊或重複 ticket | 依 `creation_time` 排序，依 `id` 去重 |
| 12. 可選排除 id | 建立 holdout 或測試資料時避免資料重疊 | `--exclude-ids-from` |

## Text Normalization

`clean_text()` 會執行以下處理：

- 缺失文字轉為空字串。
- 文字轉為小寫。
- 換行與 tab 類空白壓成一般空白。
- URL 替換為 `<URL>`。
- Email 替換為 `<EMAIL>`。
- Hex code 替換為 `<HEX>`。
- 移除過長的 alphanumeric/log-like token。
- 移除大量重複符號。
- 合併多餘空白。

## Priority Mapping

| Original Priority | Meaning | `priority_num` |
|---|---|---:|
| P1 | Highest Priority | 1 |
| P2 | High Priority | 2 |
| P3 | Medium Priority | 3 |
| P4 | Low Priority | 4 |
| P5 | Lowest Priority | 5 |

`priority_num` 保留 priority 的序位關係，數值越小代表優先級越高。這符合 GRAY 將 priority 視為 ordinal label，再透過 threshold 判定類別的設定。

## Leakage Control

- Fetch 階段只取第一則 comment 作為 `description`，不取後續 comments，避免混入修復過程、討論與狀態變更資訊。
- 清洗後依 `creation_time` 排序；後續 temporal、author、related-report features 只使用歷史 ticket。
- Evaluation 預設會移除與 training metadata 重疊的 bug id。
- `dupe_of` 只用於 REP-/duplicate similarity 權重訓練，不會直接作為 priority prediction label。

## Fetch Notes

`scripts/fetch_eclipse_bugzilla.py` 先使用 Eclipse Bugzilla REST 搜尋 ticket metadata。若公開 REST comment endpoint 回空 body，fetcher 會 fallback 到 Bugzilla XML endpoint，讀取 `comment_count=0` 的 `thetext` 作為原始 description。

若單筆 description 請求遇到暫時性 500 或 XML 解析失敗，fetcher 會重試；重試後仍失敗時，該列會保留 metadata、將 `description` 留空，並記錄 `description_source=missing` 與 `description_error`。清洗階段會依 Rule 2 排除這類沒有原始 description 的列。

若某些 priority 可用資料不足 `--target-per-priority`，fetcher 會保留實際可抓筆數，並在輸出時列出 priority distribution 與缺少 description 的筆數。
