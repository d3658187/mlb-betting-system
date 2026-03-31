# SINGLE VARIABLE ANALYSIS (v10 Sprint 1)
- 方法：單變量 Logistic Regression + train AUC
- Walk-Forward 切分：train=9160, test=2291（時間序）
- Gate：AUC >= 0.52
- 資料洩漏欄位處理：移除 home_runs, away_runs（不納入 feature set）

| Feature | Coverage(train) | AUC(train) | Keep |
|---|---:|---:|:---:|
| diff_p_ERA | 100.0% | 0.5975 | ✅ |
| diff_p_WHIP | 100.0% | 0.5927 | ✅ |
| diff_p_FIP | 100.0% | 0.5862 | ✅ |
| diff_p_xFIP | 100.0% | 0.5832 | ✅ |
| diff_p_K-BB% | 100.0% | 0.5771 | ✅ |
| diff_p_SIERA | 100.0% | 0.5746 | ✅ |
| diff_p_K% | 100.0% | 0.5672 | ✅ |
| diff_p_BB% | 100.0% | 0.5531 | ✅ |
| diff_h2h_runs_scored_avg | 95.0% | 0.5354 | ✅ |
| diff_h2h_runs_allowed_avg | 95.0% | 0.5354 | ✅ |
| diff_bat_wRC+ | 100.0% | 0.5312 | ✅ |
| home_h2h_win_pct | 95.0% | 0.5288 | ✅ |
| diff_h2h_win_pct | 95.0% | 0.5284 | ✅ |
| away_h2h_win_pct | 95.0% | 0.5279 | ✅ |
| diff_roll5_run_diff_mean | 100.0% | 0.5265 | ✅ |
| home_roll5_run_diff_mean | 100.0% | 0.5221 | ✅ |
| diff_roll5_win_mean | 100.0% | 0.5195 | ❌ |
| away_roll5_run_diff_mean | 100.0% | 0.5188 | ❌ |
| home_roll5_win_mean | 100.0% | 0.5167 | ❌ |
| away_roll5_win_mean | 100.0% | 0.5141 | ❌ |
| home_platoon_splits_score | 42.3% | 0.5119 | ❌ |
| diff_bat_wOBA | 100.0% | 0.5102 | ❌ |
| away_platoon_splits_score | 42.3% | 0.5044 | ❌ |
| home_platoon_ops_diff | 42.3% | 0.5041 | ❌ |
| diff_platoon_ops_diff | 42.3% | 0.5010 | ❌ |
| diff_platoon_splits_score | 42.3% | 0.5000 | ❌ |
| away_platoon_ops_diff | 42.3% | 0.4930 | ❌ |
