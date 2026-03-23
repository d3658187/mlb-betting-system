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
