# 數據字典

## 資料庫表格

### `teams`
| 欄位 | 類型 | 說明 |
|------|------|------|
| id | UUID | 主鍵 |
| mlb_team_id | INTEGER | MLB 官方球隊 ID |
| name | TEXT | 球隊名稱 |
| abbreviation | TEXT | 縮寫（如 NYY, BOS） |

### `games`
| 欄位 | 類型 | 說明 |
|------|------|------|
| id | UUID | 主鍵 |
| mlb_game_id | BIGINT | MLB 官方比賽 ID |
| game_date | DATE | 比賽日期 |
| game_datetime | TIMESTAMPTZ | 比賽時間（UTC） |
| home_team_id | UUID | 主場球隊 |
| away_team_id | UUID | 客場球隊 |
| venue | TEXT | 比賽場地 |
| status | TEXT | 比賽狀態 |

### `odds`
| 欄位 | 類型 | 說明 |
|------|------|------|
| id | UUID | 主鍵 |
| game_id | UUID | 關聯比賽 |
| sportsbook | TEXT | 博彩公司 |
| market | TEXT | 市場類型（moneyline/spread/total） |
| selection | TEXT | 選項（home/away/over/under） |
| price | INTEGER | 賠率（美式） |
| line | NUMERIC | 盤口 |
| retrieved_at | TIMESTAMPTZ | 抓取時間 |

### `game_results`
| 欄位 | 類型 | 說明 |
|------|------|------|
| game_id | UUID | 關聯比賽 |
| home_score | INTEGER | 主場得分 |
| away_score | INTEGER | 客場得分 |
| home_win | BOOLEAN | 主場是否贏 |
| total_points | INTEGER | 總分 |

### `game_weather`
| 欄位 | 類型 | 說明 |
|------|------|------|
| mlb_game_id | BIGINT | MLB 比賽 ID |
| temperature_c | NUMERIC | 溫度（攝氏） |
| wind_speed | NUMERIC | 風速 |
| wind_direction | NUMERIC | 風向 |

### `etl_runs`
| 欄位 | 類型 | 說明 |
|------|------|------|
| run_date | DATE | 執行日期 |
| status | TEXT | 狀態（success/failed） |
| started_at | TIMESTAMPTZ | 開始時間 |
| finished_at | TIMESTAMPTZ | 結束時間 |

---

## 模型特徵

### 投手特徵
| 特徵 | 說明 |
|------|------|
| era | 防禦率 |
| whip | 每局被上壘數 |
| era_calc | 計算防禦率 |
| starter_era_last3/5/10 | 近3/5/10場 ERA |
| starter_whip_last5/10 | 近5/10場 WHIP |
| platoon_ba_diff | vs LHB 與 vs RHB 打擊率差 |
| platoon_ops_diff | vs LHB 與 vs RHB OPS 差 |
| platoon_k_rate_lhb | 對左打三振率 (SO/PA) |
| platoon_k_rate_rhb | 對右打三振率 (SO/PA) |
| platoon_splits_score | 綜合 platoon 差距指標 |

### 牛棚特徵
| 特徵 | 說明 |
|------|------|
| bullpen_fatigue_index | 牛棚疲勞指數 |
| bullpen_pitch_count | 牛棚總投球數 |
| bullpen_appearance_days | 牛棚出賽天數 |
| bullpen_pitcher_count | 牛棚投手數量 |
| bullpen_avg_rest_days | 牛棚平均休息天數 |

### 滾動統計
| 特徵 | 說明 |
|------|------|
| roll5_runs_scored_mean | 近5場平均得分 |
| roll5_runs_allowed_mean | 近5場平均失分 |
| roll5_win_mean | 近5場勝率 |

### 對戰紀錄
| 特徵 | 說明 |
|------|------|
| h2h_games | 歷史對戰場次 |
| h2h_win_pct | 歷史對戰勝率 |

---

## 市場與盤口

### 市場類型
- `moneyline`：不讓分（主勝/客勝）
- `spread`：讓分盤（美洲盤）
- `total`：大小分

### 美式赔率轉換
```
美式赔率 → 機率：P = 100 / (|odds| + 100) （odds > 0）
                        P = |odds| / (|odds| + 100)  （odds < 0）
```

### 去水（Devig）
移除莊家抽水（10-15%），還原真實機率。
