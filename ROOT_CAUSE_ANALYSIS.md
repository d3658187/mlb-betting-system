# MLB DB 空白根因分析與修復（2026-03-31）

## TL;DR
- `data/mlb.db` 之所以是 0 bytes，**不是資料壞掉**，而是原始專案流程本來就不是寫 SQLite。  
- 既有 ETL/feature pipeline 主要依賴 `DATABASE_URL` + PostgreSQL（`init_db.sql` 也是 Postgres DDL）。
- `data/odds/*.json` 只有赔率，沒有打者/投手統計，單靠它無法建完整特徵。
- `data/mlb_stats_api/` 內容可用（特別是 `games_2022~2026.csv`），但先前沒有自動匯入到 `mlb.db`。
- 已落地 **方案 C（DB cache + CSV fallback）**：
  1) 新增本地 SQLite 建庫腳本 `build_local_mlb_db.py`  
  2) 強化 `feature_builder.py`，當 pybaseball 缺少 2026 starters 時可自動回退讀 `data/mlb_stats_api/`，並可直接輸出指定日期特徵

---

## Step 1 — 程式掃描結果

掃描 `mlb_betting_system/` 下所有 `.py`（排除 `.venv/.git`）共 35 支。

### 1) 爬蟲/資料來源相關（主要）
- `mlb_batch_crawler.py`（MLB Stats API 批次賽程 + 比分）
- `mlb_stats_api_crawler.py`（單日 schedule + probable starters）
- `mlb_stats_crawler.py`（寫入 DB 的 crawler，依賴 PostgreSQL）
- `pybaseball_daily_crawler.py`（pybaseball 日更 CSV）
- `fangraphs_crawler.py` / `fangraphs_platoon_splits_crawler.py`
- `fetch_odds_api.py`（The Odds API）
- `fetch_results.py`, `weather_crawler.py`

### 2) DB schema / DB 依賴
- Schema 檔是 `init_db.sql`（PostgreSQL 語法，含 `uuid_generate_v4()` / `TIMESTAMPTZ` 等）
- 依賴 `DATABASE_URL`/sqlalchemy 的核心腳本：
  - `feature_builder.py`
  - `daily_predictor.py`
  - `mlb_stats_crawler.py`
  - `etl_daily.py`
  - `model_trainer.py`
  - `bullpen_fatigue.py`
  - `weather_crawler.py` 等

### 3) `mlb.db` 直接引用檢查
- 原始專案幾乎沒有任何 `mlb.db` 寫入流程。  
- 掃描結果：在本次修復前，專案 Python 程式內沒有真正使用 `data/mlb.db` 作主流程 DB。

**結論：** `mlb.db` 空白是流程設計落差（Postgres-first），不是單一腳本 bug。

---

## Step 2 — `data/mlb_stats_api/` 可用性

目錄內主要有：
- `batch/games_2022.csv`（2752 rows）
- `batch/games_2023.csv`（2963 rows）
- `batch/games_2024.csv`（2957 rows）
- `batch_2025/games_2025.csv`（2961 rows）
- `spring_2026/games_2026.csv`（2920 rows）
- `daily/games_2026-03-30.csv`（15 rows）
- 各資料夾 `teams_mlb.csv`
- `spring_2026/probable_pitchers_2026-03-21.csv`（只有單日）

**可用結論：**
- 可用來重建「比賽層」歷史資料（game-level）。
- 但投手層（probable starters）並不完整（只有少量日期）。
- 需搭配 pybaseball/FanGraphs 才能補齊 batting/pitching 特徵。

---

## Step 3 — pybaseball 實測

實測指令（在 `.venv`）：
```python
from pybaseball import team_pitching
print(len(team_pitching(2025)))  # 30
print(len(team_pitching(2026)))  # 30
```

結果：
- `team_pitching(2025)` 可取到 30 隊
- `team_pitching(2026)` 也可取到 30 隊（當季累積統計）

**結論：** pybaseball 可直接作為本地資料源（至少 team-level 可行）。

---

## Step 4 — 選擇方案

選擇 **方案 C：DB 作 cache + CSV fallback 並存**。

理由：
1. 不破壞原本 PostgreSQL 正式流程
2. 本地可離線工作（不再卡在空的 `mlb.db`）
3. 2026 年初期資料不完整時，能從 `mlb_stats_api` 回填 game-level 資料

---

## 已實作修復

### A) 新增本地 SQLite 建庫腳本
新增：`build_local_mlb_db.py`

功能：
- 讀取 `data/pybaseball` +（透過 `feature_builder`）整合 `data/mlb_stats_api`
- 建立/覆蓋 `data/mlb.db`
- 寫入 tables：
  - `games`
  - `team_batting`
  - `team_pitching`
  - `pitcher_stats`
  - `starting_pitchers`
  - `platoon_splits`
  - `metadata`

實測：
```bash
python build_local_mlb_db.py --db-path data/mlb.db --data-dir data/pybaseball --seasons 2022-2026
```
輸出：
- `data/mlb.db` 約 3.8 MB（非 0 bytes）
- `games` 14874 rows
- `team_batting` 124 rows
- `team_pitching` 124 rows
- `pitcher_stats` 3094 rows
- `starting_pitchers` 9719 rows
- `platoon_splits` 993 rows

### B) 強化 `feature_builder.py`（CSV fallback）
主要改動：
1. `load_pybaseball_games()` 新增對 `data/mlb_stats_api/**/*.csv` 的回填能力（特別補 2026）
2. 新增隊名/縮寫正規化（例如 `SD->SDP`, `SF->SFG`, `TB->TBR`, `AZ->ARI`, `WSH/WAS->WSN`）
3. 新增 `build_daily_features_from_csv()`
4. CLI 新增 `--historical-date YYYY-MM-DD`，可直接輸出單日特徵

---

## 驗證：2026-03-31 特徵非全 0

執行：
```bash
python feature_builder.py \
  --historical \
  --historical-date 2026-03-31 \
  --seasons 2022-2026 \
  --data-dir ./data/pybaseball \
  --out ./data/features_2026-03-31.csv
```

結果：
- 產出 `data/features_2026-03-31.csv`
- 14 場比賽
- 每場非零數值特徵數量約 74~83（**無全 0**）

---

## 對既有爬蟲是否可運作的結論

- `mlb_batch_crawler.py`：可正常拉 season schedule/game details（測到 2026 共 2920 game_pks）。
- `mlb_stats_api_crawler.py`：可正常拉單日 schedule + probable starters（2026-03-31 測到 games=14, probable_starters=27）。
- `mlb_stats_crawler.py`：設計上需 PostgreSQL + 既有 schema，不適合直接寫 `data/mlb.db`。

---

## 其他資料源可行性

- **FanGraphs**：可透過 pybaseball 路徑取得（目前專案已在用）
- **Baseball Reference**：無穩定官方 public API；通常仍透過 pybaseball/爬取間接取得

結論：實務上以 **pybaseball + MLB Stats API CSV** 最穩。

---

## 變更檔案

- `feature_builder.py`（已修改）
- `build_local_mlb_db.py`（新增）
- `ROOT_CAUSE_ANALYSIS.md`（新增）

---

## 後續建議

1. 若要讓 `daily_predictor.py` 在無 Postgres 下也完全走新 fallback，可再把離線模式接到 `build_daily_features_from_csv()`。  
2. `import_historical_data.py` / `import_pitcher_data.py` 的合併檔命名目前固定 `2023_2026`，建議改為依實際季別生成，避免檔名與內容不一致。  
3. `starting_pitchers_2023_2025.csv` 內容日期與檔名曾出現不一致跡象，建議加資料一致性檢查。