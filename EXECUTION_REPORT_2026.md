# EXECUTION_REPORT_2026

## 任務
MLB 冷啟動改善（2026-03-31 晚間）

## 完成項目

### Step 1 — since_date 延長
- 檔案：`feature_builder.py`
- 變更：
  - `since_date = target_date - timedelta(days=200)`
  - → `since_date = target_date - timedelta(days=400)`
- 位置：`feature_builder.py:979`

### Step 2 — performance tracker 檔案
- 檔案：`data/performance_tracker.csv`
- 已確認存在，且表頭符合要求：
  - `date,game_id,home_team,away_team,pythagorean_prob,ml_model_prob,market_prob,actual_outcome,correct_pythagorean,correct_ml`

### Step 3 — daily_predictor 離線模式（避開 DB）
- 檔案：`daily_predictor.py`
- 核心改動：
  1. 新增離線 odds-json 預測流程 `run_offline_prediction_mode(...)`
  2. 當 DB 無法讀取（例如 `mlb.db` 無 table）時，自動 fallback 到離線模式
  3. 離線模式資料來源：
     - Odds：`data/odds/the-odds-api_YYYY-MM-DD.json`
     - Features template：優先 `--offline-features-csv`，預設 `data/features_2026-03-20.csv`，再 fallback 至最新 `data/features_*.csv`
  4. 若模型/特徵不可用，使用 base rate（預設 `0.410674`）
  5. 產出檔案：`data/predictions_YYYY-MM-DD.csv`
- 新增參數：
  - `--offline-odds-json`
  - `--offline-features-csv`
  - `--offline-base-rate`
  - `--offline-max-games`

### Step 3 驗證（3/31 當晚）
執行：
```bash
DATABASE_URL=sqlite:///data/mlb.db \
python3 daily_predictor.py \
  --date 2026-03-31 \
  --offline-odds-json data/odds/the-odds-api_2026-03-31.json \
  --offline-max-games 11 \
  --out data/predictions_2026-03-31.csv
```

結果：
- 成功偵測 DB 無 tables，切換離線模式
- 成功輸出：`data/predictions_2026-03-31.csv`
- 筆數：11 場（`unique game_id = 11`）

輸出欄位：
- `prediction_date,game_id,away_team,home_team,home_win_prob,away_win_prob,market_home_prob,home_price,away_price,predicted_winner,model_source`

## 備註
- 目前 `data/` 與 `*.csv` 在 `.gitignore` 中，預測檔與 tracker 檔屬執行產物，不會被 git 追蹤。
- 離線模式在冷啟動（空 DB）情境可直接出預測，避免原本全 0/無輸出問題。
