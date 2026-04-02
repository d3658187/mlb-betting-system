# MLB performance_tracker 修復報告（2026-04-02）

## 範圍
工作目錄：`/Users/duste/.openclaw/workspace-zongzhihui/mlb_betting_system`

## Step 1 — 清理現有 tracker
- 新增 `update_tracker.py`，提供 schema 正規化 + 去重邏輯。
- 去重 key：`date + home_team + away_team`
- 保留規則：優先保留 `ml_model_prob` 有值的列；同優先級時保留最新列。
- 已執行：
  - `python3 update_tracker.py --tracker data/performance_tracker.csv`
  - 清理結果：`before=56, after=56, removed=0`
- `data/performance_tracker.csv` 已統一為欄位：
  - `date, game_id, home_team, away_team, ml_model_prob, market_prob, actual_outcome, correct_ml`

## Step 2 — 每日預測自動寫入機制
- 保留 `daily_predictor.py` 原本「預測完成即寫入 tracker」行為。
- 新增 `update_tracker.py` 並接入 `scripts/mlb_daily_pipeline.sh`：
  - 在 daily predictor 後自動執行 tracker 正規化/去重。
- 統一保留欄位：
  - `date, game_id, home_team, away_team, ml_model_prob, market_prob, actual_outcome, correct_ml`
- 去重 key：`date + home_team + away_team`，並優先保留 `ml_model_prob` 有值的列。

## Step 3 — 賽果更新腳本
- 新增 `scripts/update_results.py`
- 功能：
  - 輸入日期（預設昨天）
  - 從 MLB Stats API (`/api/v1/schedule?sportId=1&date=YYYY-MM-DD`) 抓 Final 比賽
  - 更新 tracker 對應比賽 `actual_outcome` 與 `correct_ml`
  - 不新增重複場次（僅更新既有 row，最後再做去重）
- 實測：
  - `python3 scripts/update_results.py --date 2026-03-31 --tracker data/performance_tracker.csv`
  - 輸出：`finals=14 tracker_rows_updated=14`

## Step 4 — launchd 每日自動任務（09:00 台灣時間）
- 新增 plist：
  - `scripts/launchd/ai.openclaw.mlb.update-results.plist`
- 新增安裝腳本：
  - `scripts/install_update_results_launchd.sh`
- 已安裝並 load 到使用者 LaunchAgents：
  - `~/Library/LaunchAgents/ai.openclaw.mlb.update-results.plist`
- 排程：每日 09:00 執行 `scripts/update_results.py`
- 日誌：
  - `/Users/duste/.openclaw/workspace-zongzhihui/mlb_betting_system/logs/update_results.launchd.out.log`
  - `/Users/duste/.openclaw/workspace-zongzhihui/mlb_betting_system/logs/update_results.launchd.err.log`

## 驗證
- 語法檢查：
  - `python3 -m py_compile daily_predictor.py update_tracker.py scripts/update_results.py`
- 去重檢查：
  - `duplicates by (date,home_team,away_team) = 0`
- launchd 測試：
  - `launchctl start ai.openclaw.mlb.update-results` 可正常執行

## 新增/修改檔案
- `update_tracker.py`（新增，tracker 正規化與去重）
- `scripts/update_results.py`（新增，MLB Stats API 賽果回填）
- `scripts/mlb_daily_pipeline.sh`（新增 Step 3.1，自動執行 tracker 去重）
- `scripts/launchd/ai.openclaw.mlb.update-results.plist`（新增）
- `scripts/install_update_results_launchd.sh`（新增）
- `data/performance_tracker.csv`（已清理）
- `TRACKER_FIX_REPORT_2026-04-02.md`（本報告）
