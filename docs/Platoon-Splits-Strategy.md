# Platoon Splits 數據策略決策報告

**日期：** 2026-03-24
**角色：** 軍師（策略分析）
**背景：** 需要投手 Platoon Splits 數據（投手對左/右打者表現），FanGraphs 直接爬蟲失敗。

---

## 1. 數據來源評估

### 方案 A：pybaseball `get_splits()` + Baseball Reference

**可行性：** ✅ 已驗證可用

**運作方式：**
- pybaseball 已有 `get_splits(playerid, year, pitching_splits=True)` 函式
- 數據來源：**Baseball Reference**（非 FanGraphs，避開 JS 渲染問題）
- 需使用 Baseball Reference player ID（格式：`kershcl01`）

**實測結果（Clayton Kershaw 2024）：**
```
Split Types 包含：Platoon Splits
Platoon Splits 值：['vs RHB', 'vs LHB', 'vs RHB as LHP', 'vs LHB as LHP']

關鍵欄位：G, PA, AB, H, 2B, 3B, HR, BB, SO, BA, OBP, SLG, OPS, BAbip, tOPS+, sOPS+
```

**優勢：**
- 數據完整，Platoon Splits 包含所有核心指標
- pybaseball 已整合在專案 `.venv` 中，無需額外安裝
- Baseball Reference 反爬機制比 FanGraphs 寬鬆
- `playerid_reverse_lookup(mlbam_ids, key_type='mlbam')` 可將 MLBAM ID 轉換為 BBRef ID

**劣勢：**
- 每位投手需一次 HTTP 請求（Baseball Reference 單一球員頁面）
- 全聯盟投手名單約 600-800 人：需批次 + 限速處理
- Career splits 的 HTML 結構略有不同（多一個 `I` 欄位），需注意解析邏輯

**估算成本：** 中等（需實作批次迴圈 + 限速，估計 10-15 分鐘完成全量抓取）

---

### 方案 B：Statcast 原始數據自行聚合

**可行性：** ⚠️ 可行但成本高

**運作方式：**
- 使用 `statcast_pitcher(start_dt, end_dt, pitcher_id)` 或 `statcast()`
- 對結果按 `stand`（L/R）分組聚合
- 計算 wOBA、K/PA 等指標

**劣勢：**
- Statcast 返回的是**逐球級別**數據（一次查詢數十萬筆）
- 無法直接查詢「本賽季 vs LHB」的預聚合 splits
- 自己計算 wOBA 需要完整的 PA 數據（計算複雜）
- 實作成本高，準確度未必優於 BBRef

**估算成本：** 高

---

### 方案 C：MLB Stats API + 計算層

**可行性：** ⚠️ API 無法直接取得 Splits

MLB Stats API (`statsapi.mlb.com`) 目前沒有公開的投手 Platoon Splits 端點。需自行從逐球數據聚合。

**結論：** 不推薦

---

### 方案 D：FanGraphs（cloudscraper 繞過）

**可行性：** ⚠️ 不穩定

現有 `fangraphs_crawler.py` 已使用 `cloudscraper`，但 FanGraphs 的防爬機制時常更新，容易失敗。

**結論：** 作為備用方案，不作為首選

---

## 2. 決策

**首選方案：A（pybaseball `get_splits()` + Baseball Reference）**

理由：
1. 數據完全吻合需求（Platoon Splits - vs LHB / vs RHB）
2. pybaseball 已是專案依賴，無額外整合成本
3. `playerid_reverse_lookup` 可解決 MLBAM → BBRef ID 映射問題
4. Baseball Reference 反爬較寬鬆，可批次處理

---

## 3. 實作路徑

### 3.1 數據管線設計

```
MLB Stats API (現有)
    → 取得投手名單 + MLBAM ID
    ↓
playerid_reverse_lookup (pybaseball)
    → 轉換為 BBRef ID
    ↓
get_splits(bbref_id, year=season, pitching_splits=True) (pybaseball)
    → 取得原始 splits DataFrame
    ↓
過濾：Split Type = 'Platoon Splits', Split in ['vs LHB', 'vs RHB']
    → 提取關鍵指標
    ↓
寫入 DB / CSV
```

### 3.2 關鍵 Derived Features（建議）

從 `vs LHB` / `vs RHB` 原始數據計算：

| Feature | 公式 | 說明 |
|---------|------|------|
| `platoon_ba_diff` | BA_vs_LHB - BA_vs_RHB | 左打打擊率 - 右打打擊率 |
| `platoon_ops_diff` | OPS_vs_LHB - OPS_vs_RHB | 左打 OPS - 右打 OPS |
| `platoon_k_rate_lhb` | SO / PA (vs LHB) | 對左打三振率 |
| `platoon_k_rate_rhb` | SO / PA (vs RHB) | 對右打三振率 |
| `platoon_hr_rate_lhb` | HR / PA (vs LHB) | 對左打全壘打率 |
| `platoon_iso_lhb` | SLG - BA (vs LHB) | 對左打 Isolated Power |
| `platoon_iso_rhb` | SLG - BA (vs RHB) | 對右打 Isolated Power |
| `platoon_splits_score` | `abs(platoon_ba_diff)` + `abs(platoon_ops_diff) * 0.5` | 綜合 platoon 差距 |

### 3.3 實作時的重點考量

1. **限速**：每分鐘不超過 20 次請求，避免被 block
2. **快取**：每個球員當天只請求一次，結果寫入 CSV/SQLite
3. **失敗重試**：HTTP 錯誤時 exponential backoff
4. **季末 vs 季中**：賽季中每週更新一次，季末一次全量更新
5. **缺失數據**：新秀投手樣本數不足時（PA < 30），標註為 `NULL` 或用 league average 替代

### 3.4 驗證方式

```
1. 隨機抽 10 位投手，對比 pybaseball 輸出與 Baseball Reference 網頁
2. 確認 G/PA/AB/H/HR/BB/SO 數值完全一致
3. 確認計算derived features邏輯正確
```

---

## 4. 風險提示

| 風險 | 等級 | 緩解措施 |
|------|------|----------|
| Baseball Reference 反爬升級 | 中 | 加入 `time.sleep()` 限速；準備 cloudscraper fallback |
| 投手樣本數不足（少數 PA） | 中 | 設定最低 PA 門檻（如 30 PA），否則標註 `NULL` |
| BBRef ID 映射失敗 | 低 | 少數新秀可能映射失敗，需人工核對 |
| 數據延遲（BBRef 更新落後） | 低 | 通常比賽後 1-2 小時更新，可接受 |

---

## 5. 總結

| 項目 | 結論 |
|------|------|
| **首選數據源** | Baseball Reference（via pybaseball） |
| **關鍵函式** | `get_splits(bbref_id, year, pitching_splits=True)` |
| **ID 映射** | `playerid_reverse_lookup(mlbam_ids, key_type='mlbam')` |
| **Platoon Splits 欄位** | `vs LHB`, `vs RHB`（來自 Split Type = 'Platoon Splits'） |
| **實作成本** | 中（需批次 + 限速迴圈，約 1-2 天） |
| **備用方案** | Statcast 自行聚合（成本高，不建議） |
