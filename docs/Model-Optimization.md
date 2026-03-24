# Model Optimization (XGBoost v6)

## 現況與瓶頸
- **目前表現**：XGBoost v6（2022-2024），準確率 57.7%。
- **主要瓶頸**
  1. **先發投手狀態缺口**：v6 資料集多為賽季平均指標，缺少「最近幾場先發表現」「休息天數」「對特定對手的歷史表現」。
  2. **僅單一模型**：XGB 單模型在不同市場（讓分/大小分）與不同賽季波動時泛化不足。
  3. **特徵變化不足**：滾動與 H2H 只針對球隊，投手層級的 temporal 訊號不足。

## 已實作優化
### 1) 新增投手近期狀態特徵（build_training_v6_dataset.py）
新增 `attach_pitcher_form_features()`，產生以下特徵並寫入 v6 訓練集：
- **先發休息天數**：`home_pitcher_rest_days` / `away_pitcher_rest_days`
- **先發累積出賽數**：`home_pitcher_start_count` / `away_pitcher_start_count`
- **先發近期勝率（last5/last10）**：`home_pitcher_roll5_win_pct`, `home_pitcher_roll10_win_pct`（客隊同理）
- **先發近期淨勝分（last5/last10）**：`home_pitcher_roll5_run_diff_mean`, `home_pitcher_roll10_run_diff_mean`（客隊同理）
- **投手 vs 對手勝率**：`home_pitcher_vs_opp_win_pct`（客隊同理）
- **對應差值特徵**：
  - `diff_pitcher_roll5_win_pct`
  - `diff_pitcher_roll10_win_pct`
  - `diff_pitcher_roll5_run_diff_mean`
  - `diff_pitcher_roll10_run_diff_mean`
  - `diff_pitcher_rest_days`
  - `diff_pitcher_vs_opp_win_pct`

> 以上特徵皆為 **shift(1)** 形式，避免資訊外洩。

### 2) 新增集成模型訓練腳本（train_ensemble_model.py）
- 使用 **XGBoost + LightGBM + HistGradientBoosting** 軟投票（soft voting）。
- 當環境缺少某模型時自動降級。
- 支援時間序列切分（game_date）避免資料洩漏。

### 3) 2022-2025 v6（投手狀態特徵）集成模型訓練結果
- **訓練資料**：`data/training_2022_2025_enhanced_v6.csv`
- **模型檔**：`models/mlb_v6_ensemble_pitcher_state.pkl`
- **評估指標（time_series split）**：
  - Train: 9,144 / Test: 2,287
  - Accuracy: **54.53%**
  - ROC-AUC: **0.5267**
- **特徵數**：75
- **使用特徵（完整清單）**：
  - game_date_x, home_team, away_team, home_pitcher_mlbam, away_pitcher_mlbam, season, game_key, game_date_y
  - home_roll5_runs_scored_mean, home_roll5_runs_allowed_mean, home_roll5_run_diff_mean, home_roll5_win_mean
  - home_ha_roll5_runs_scored_mean, home_ha_roll5_runs_allowed_mean, home_ha_roll5_win_mean
  - away_roll5_runs_scored_mean, away_roll5_runs_allowed_mean, away_roll5_run_diff_mean, away_roll5_win_mean
  - away_ha_roll5_runs_scored_mean, away_ha_roll5_runs_allowed_mean, away_ha_roll5_win_mean
  - home_h2h_games, home_h2h_win_pct, home_h2h_runs_scored_avg, home_h2h_runs_allowed_avg
  - away_h2h_games, away_h2h_win_pct, away_h2h_runs_scored_avg, away_h2h_runs_allowed_avg
  - home_p_ERA, home_p_WHIP, home_p_K%, home_p_BB%, home_p_K-BB%, home_p_FIP, home_p_xFIP, home_p_SIERA, home_p_WAR, home_p_IP
  - away_p_ERA, away_p_WHIP, away_p_K%, away_p_BB%, away_p_K-BB%, away_p_FIP, away_p_xFIP, away_p_SIERA, away_p_WAR, away_p_IP
  - home_bat_AVG, home_bat_OBP, home_bat_SLG, home_bat_OPS, home_bat_ISO, home_bat_wOBA, home_bat_wRC+, home_bat_BB%, home_bat_K%, home_bat_R, home_bat_HR, home_bat_SB
  - away_bat_AVG, away_bat_OBP, away_bat_SLG, away_bat_OPS, away_bat_ISO, away_bat_wOBA, away_bat_wRC+, away_bat_BB%, away_bat_K%, away_bat_R, away_bat_HR, away_bat_SB
  - diff_p_ERA, diff_p_WHIP, diff_p_K-BB%, diff_bat_wOBA, diff_bat_wRC+, diff_bat_OPS

### 4) 2022-2025 Platoon Splits（feature_builder historical）集成模型訓練結果
- **訓練資料**：`data/training_2022_2025_platoon.csv`
- **資料筆數**：總計 10,685；標註 (home_win) 801
- **Platoon Splits 覆蓋率**：**100%**（home/away 皆完整）
- **模型檔**：`models/mlb_v6_ensemble_platoon.pkl`
- **評估指標（time_series split）**：
  - Train: 640 / Test: 161
  - Accuracy: **45.96%**
  - ROC-AUC: **0.4274**
- **特徵數**：2,293
- **Platoon Splits 特徵**：
  - home_platoon_ba_diff, home_platoon_ops_diff, home_platoon_k_rate_lhb, home_platoon_k_rate_rhb
  - away_platoon_ba_diff, away_platoon_ops_diff, away_platoon_k_rate_lhb, away_platoon_k_rate_rhb
  - diff_platoon_ba_diff, diff_platoon_ops_diff, diff_platoon_k_rate_lhb, diff_platoon_k_rate_rhb
- **特徵重要性檢查（XGBoost 探查）**：
  - Platoon Splits 欄位重要性均為 0（未進入 Top 15）
  - 明細：`models/mlb_v6_platoon_feature_importance.json`

### 5) 2022-2024 Platoon Splits（完整標籤 v7）集成模型訓練結果
- **訓練資料**：`data/training_2022_2024_platoon.csv`
- **Platoon Splits 覆蓋率**：主場 97.3%、客場 96.9%
- **模型檔**：`models/mlb_v7_ensemble_platoon.pkl`
- **評估指標（time_series split）**：
  - Train: 6,808 / Test: 1,703
  - Accuracy: **54.73%**
  - ROC-AUC: **0.5829**
- **特徵數**：101

## 6) 2022-2025 v8 Over/Under + Run Line（回歸）
- **訓練資料**：`data/training_2022_2025_enhanced_v6.csv`
- **標籤來源**：
  - total_runs = home_runs + away_runs
  - run_margin = home_runs - away_runs
- **模型檔**：
  - Over/Under：`models/mlb_v8_overunder.booster`
  - Run Line：`models/mlb_v8_runline.booster`
- **評估指標（time_series split）**：
  - Over/Under MAE: **3.70** / RMSE: **4.73** / R2: **-0.0648**
  - Run Line MAE: **3.61** / RMSE: **4.72** / R2: **-0.0535**
- **特徵數**：76
- **備註**：現有訓練集無 **cover_spread / 大小分盤口** 欄位，僅能回歸預測總得分與勝分差；若需盤口命中率評估，需串接 SportsdataIO / Odds API 等賠率資料。

## 使用方式
### 產生新版 v6 訓練集
```bash
DATABASE_URL=postgresql://user:pass@host:5432/dbname \
python build_training_v6_dataset.py \
  --start-date 2022-01-01 --end-date 2024-12-31 \
  --out ./data/training_2022_2024_enhanced_v6.csv \
  --pybaseball-dir ./data/pybaseball
```

### 訓練集成模型（目標：home_win / cover_spread）
```bash
python train_ensemble_model.py \
  --csv ./data/training_2022_2024_enhanced_v6.csv \
  --target home_win \
  --out-dir ./models \
  --model-name mlb_v6_ensemble
```

## 建議下一步（可選）
1. **針對市場建立專門模型**
   - 讓分：以 `cover_spread` 訓練。
   - 大小分：增加 `total_points` 相關特徵（天氣 + 先發 / bullpen）。
2. **分賽季 / 分月份校正**
   - 每季重新訓練或使用時間衰減權重。
3. **校準預測機率**
   - 可加上 `CalibratedClassifierCV` 讓賠率決策更穩定。

## 變更檔案
- `build_training_v6_dataset.py`：新增投手近期狀態 + 對戰特徵
- `train_ensemble_model.py`：新增集成模型訓練腳本
