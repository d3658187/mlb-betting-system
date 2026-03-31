# V10_FEATURES (Core 16)
- 來源：v6 訓練集 + 補齊 platoon 欄位 + 派生 diff 欄位
- 篩選規則：單變量 Logistic AUC >= 0.52

## Final Feature List
1. `diff_p_xFIP` (AUC=0.5832)
2. `diff_p_K-BB%` (AUC=0.5771)
3. `diff_bat_wRC+` (AUC=0.5312)
4. `diff_p_ERA` (AUC=0.5975)
5. `diff_p_WHIP` (AUC=0.5927)
6. `diff_p_FIP` (AUC=0.5862)
7. `diff_p_SIERA` (AUC=0.5746)
8. `diff_p_K%` (AUC=0.5672)
9. `diff_p_BB%` (AUC=0.5531)
10. `home_roll5_run_diff_mean` (AUC=0.5221)
11. `diff_roll5_run_diff_mean` (AUC=0.5265)
12. `home_h2h_win_pct` (AUC=0.5288)
13. `away_h2h_win_pct` (AUC=0.5279)
14. `diff_h2h_win_pct` (AUC=0.5284)
15. `diff_h2h_runs_scored_avg` (AUC=0.5354)
16. `diff_h2h_runs_allowed_avg` (AUC=0.5354)

## Note
- 軍師建議的 platoon 四欄在目前資料的單變量 AUC 未達 0.52，已依 Gate 排除。
- 為維持 16-18 核心欄位，使用同類型且通過 Gate 的差值/動能/H2H 欄位補齊。
