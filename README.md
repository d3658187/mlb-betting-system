# MLB Betting System

MLB 投注預測系統，專為台灣運彩設計。

## 功能架構

```
mlb_betting_system/
├── daily_predictor.py      # 每日預測主程式
├── model_trainer.py        # 模型訓練
├── feature_builder.py      # 特徵工程
├── backtest.py             # 回測系統
├── etl_daily.py            # 每日資料 ETL
├── discord_notifier.py     # Discord 推播通知
├── pybaseball_daily_crawler.py  # pybaseball 資料抓取
├── fangraphs_crawler.py    # FanGraphs 進階數據
├── weather_crawler.py      # 天氣資料
├── taiwan_lottery_crawler.py  # 台灣運彩赔率
├── bullpen_fatigue.py      # 牛棚疲勞度分析
├── mlb_stats_api_crawler.py   # MLB Stats API
├── init_db.sql             # 資料庫結構
└── models/                 # 訓練好的模型
```

## 模型版本

- **v6**（當前）：XGBoost，含滾動統計、牛棚疲勞度、對戰紀錄等進階特徵
- **v5**：早期版本，含天氣、特徵工程

## 預測目標

| 類型 | 說明 |
|------|------|
| 讓分盤（Cover Spread） | 預測主場是否贏過盤口 |
| 勝分差（Run Margin） | 預測勝分範圍 |
| 大小分（Over/Under） | 進行中 |

## 資料來源

- **MLB Stats API**：即時比賽數據
- **pybaseball**：Play-by-Play、Statcast
- **FanGraphs**：進階投手/打者數據（FIP、wOBA、WAR）
- **台灣運彩**：盤口赔率

## 預測流程

```
每日 20:00（MLB 開打前）
  └─ 抓取先發投手名單
  └─ 抓取最新投手/打者統計
  └─ 計算滾動特徵（近5場）
  └─ 模型預測 + 機率輸出
  └─ 計算期望值（EV）
  └─ Discord 推播
```

## 安裝需求

```bash
pip install xgboost pandas scikit-learn psycopg2-binary
pip install pybaseball
```

## 資料庫

- PostgreSQL
- 主要資料表：`games`、`stats_batting`、`stats_pitching`

## 主要特徵

- 先發投手 ERA、FIP、WHIP、SO
- 主客場滾動表現（近5場）
- 牛棚疲勞度（牛棚投手最近60場平均）
- 對戰歷史（H2H）
- 天氣（溫度、風向）

## 文件

- [數據字典](docs/Data-Dictionary.md) — 資料庫欄位、模型特徵說明
- [模型文檔](docs/Model-Documentation.md) — 模型架構、訓練流程、預測邏輯

## Issues

追蹤系統問題：https://github.com/d3658187/mlb-betting-system/issues
