# MLB 投注赔率數據最佳可行方案

**報告日期：** 2026-03-24
**角色：** 軍師（策略分析）
**目標：** 評估並選擇 MLB 投注赔率數據來源，重點為台灣運彩（讓分盤、大小分）

---

## 1. 背景問題重述

| 數據需求 | 背景 |
|---|---|
| 目標市場 | 台灣運彩（讓分盤、大小分） |
| JBot API（sportsbot.tech）| MLB 免費版服務不穩定 |
| The Odds API | 背景記載「DNS 無法解析，網路被封鎖」|
| 台灣運彩官網 | 無法直接存取 |

---

## 2. 關鍵發現：網路狀況重新驗證

### ⚠️ 背景描述與事實不符

經實測驗證，以下數據**並未封鎖**：

| API | 域名 | DNS 解析 | HTTP 回應 | 結論 |
|---|---|---|---|---|
| **The Odds API** | `api.the-odds-api.com` | ✅ 解析至 52.86.109.39 | ✅ 401 (需 key) | **未封鎖，可直連** |
| **TheRundown** | `therundown.io` | ✅ | ✅ 正常 | **未封鎖，可直連** |
| **JBot** | `sportsbot.tech` | ✅ | ✅ 正常 | **未封鎖，需 Token** |
| **SportsDataIO** | `sportsdata.io` | ✅ | ✅ 正常 | **未封鎖，需付費** |

**「DNS 無法解析」的說法需要更正：** 本機網路可正常解析這些域名，問題僅是需要 API Key 驗證，而非網路封鎖。**

台灣運彩官網 (`sportslottery.com.tw`) 是否封鎖尚待驗證（需進一步測試）。

---

## 3. 數據源完整評估

### 3.1 The Odds API（最推薦）

**狀態：** ✅ 直連可用，**免費版不需信用卡**

**方案：** https://the-odds-api.com
**API 端點：** `https://api.the-odds-api.com/v4/sports/baseball_mlb/odds`

**規格：**
- **免費額度：** 500 次/月（MLB 約可查詢 150-200 場比賽）
- **收費方案：** 免費版 $0；付費 $9/月起（500,000 次調用）
- **市場覆蓋：** Moneyline、Spreads（讓分）、Totals（大小）、Run Line
- **數據來源：** DraftKings、Fanduel、PointsBet、Barstool 等美國博彩公司
- **格式：** 美式赔率（+105、-125 等）

**優點：**
- 直連無需代理
- 台灣團隊驗證：**DNS 正常，API 回應正常**
- 免費額度對原型開發充足
- 覆蓋美國盤口，可用於校準與參考

**缺點：**
- **非台灣運彩赔率**，而是美國博彩公司赔率
- 需註冊取得 API Key
- 需將美國赔率轉換為台灣運彩格式（可計算隱含概率）

**與台灣運彩的橋樑：** 台灣運彩的讓分/大小分與美國盤口高度相關，可透過美國盤口計算合理讓分值，再對比台灣運彩溢價後的赔率找出價值。

---

### 3.2 TheRundown

**狀態：** ✅ 直連可用，**免費版需申請**

**方案：** https://therundown.io
**API Key 申請：** therundown.io/api（免費）

**規格：**
- **免費額度：** 20,000 數據點/天，3 個 bookmakers，賽前赔率
- **收費方案：** Starter $49/月；Pro $149/月（含即時數據）
- **市場覆蓋：** Moneyline、Spread、Total（MLB 全場、半場）
- **數據來源：** 5Dimes、Pinnacle、Matchbook、Bovada、DraftKings、BetMGM 等
- **特殊功能：** WebSocket 即時推送（Pro 版）

**優點：**
- 直連無需代理
- 免費版就涵蓋主要市場
- 覆蓋 Pinnacle（最準確的赔率來源）

**缺點：**
- 需申請 API Key（非開源即可用）
- 同樣是美國博彩公司赔率，非台灣運彩
- 免費版每天 20,000 數據點，MLB 賽季（約 2,430 場）夠用但需控制查詢頻率

---

### 3.3 JBot API（台灣運彩原生）

**狀態：** ⚠️ 直連可用，**MLB 免費版不穩定**

**方案：** https://api.sportsbot.tech/v2/odds
**Token 申請：** sportsbot.tech（免費版每日 20 次呼叫）

**規格：**
- **免費額度：** 20 次/天
- **覆蓋：** 台灣運彩 NBA、MLB、中職、日棒、韓棒、足球、網球、冰球、排球
- **市場：** 角球、不讓分、讓分、總分、主場得分、客場得分
- **數據格式：** 台灣運彩原生格式（讓分、大小分）

**優點：**
- **台灣運彩原生數據**（主要目標）
- 數據格式直接可用
- 免費版有 20 次/天

**缺點：**
- **MLB 免費版目前服務不穩定**（背景記載）
- 每日 20 次不足以應付完整賽季（每天 MLB 約 15 場）
- 若升級付費版，需額外成本

**建議：** 作為主要數據源（若 JBot MLB 恢復穩定），同時用 The Odds API 作為備援與美國盘口校準參考。

---

### 3.4 SportsDataIO

**狀態：** ✅ 直連可用，**需付費**

**方案：** https://sportsdata.io
**免費試用：** 需註冊，僅提供上賽季數據

**規格：**
- **MLB 覆蓋：** 即時赔率、歷史赔率、先發投手數據、打者統計
- **收費：** 約 $99/月起（Startup Plan）
- **特點：** 同時提供比赛统计数据與赔率

**缺點：**
- 免費版僅提供上賽季（延遲一年），對預測系統無即時價值
- 需信用卡
- 主要針對美國市場，非台灣運彩

---

### 3.5 台灣運彩官網爬蟲（長期方案）

**狀態：** ❌ 目前無法直接存取

**技術方案：**
1. **Playwright + 代理伺服器：** 使用台灣 VPS 或住宅代理繞過封鎖
2. **Bright Data / NetNut 台灣代理：** 可取得台灣 IP，直接存取台灣運彩
3. **成本：** Bright Data 約 $15/月起（台灣代理）

**風險：** 需確認台灣運彩官網是否有反爬蟲機制。

---

## 4. 推薦方案：分層策略

### 核心策略：三源分層

```
第1層（主要）: JBot API（台灣運彩原生）
  └─ 若 MLB 免費版恢復穩定：作為首選
  └─ 若仍不穩定：升級付費版或使用台灣代理爬蟲

第2層（參考/校準）: The Odds API（免費版）
  └─ 美國博彩公司赔率，用於計算合理盤口值
  └─ 與台灣運彩盘口對比，識別價值投注

第3層（備援）: TheRundown（免費版）
  └─ Pinnacle 盘口作為市場共識參考
  └─ WebSocket 即時更新（Pro 版需求時）

長期方案：台灣代理爬蟲（台灣 VPS + Playwright）
```

### 具體執行步驟

| 優先級 | 動作 | 資源需求 |
|---|---|---|
| **P0** | 申請 The Odds API 免費 Key（500 次/月）| 無成本 |
| **P0** | 申請 TheRundown 免費 Key（20,000 點/天）| 無成本 |
| **P1** | 評估 JBot MLB 穩定性（觀察 1-2 週）| 無成本 |
| **P1** | 若 JBot 不穩定：評估 Bright Data 台灣代理 | ~$15/月 |
| **P2** | 建立美國盘口 → 台灣運彩格式轉換邏輯 | 開發時間 |

---

## 5. 赔率轉換邏輯（美國 → 台灣運彩）

台灣運彩的赔率計算方式與美國盘口不同：

**美國盘口（以 DraftKings 為例）：**
- Moneyline：+120 = 投注 $100 贏 $120
- Run Line（讓分）：-1.5 @ +150

**台灣運彩讓分計算（棒球）：**
- 讓分值以 "+1+30" 等格式表示
- 計算涉及讓分比例 × 赔率 × 本金
- 詳見：https://vocus.cc/article/613911adfd89780001934d0c

**轉換策略：**
1. 從 The Odds API 取得 DraftKings/Fanduel 美國赔率
2. 計算隱含概率（Implied Probability）
3. 對比台灣運彩盘口，識別溢價幅度
4. 溢價大於門檻值（如 5%）→ 價值投注信號

---

## 6. 風險評估

| 風險 | 等級 | 說明 |
|---|---|---|
| JBot MLB 免費版不穩定 | 🔴 高 | 背景已記載，需持續監控 |
| API Key 申請失敗 | 🟡 中 | 需國外 email，The Odds API 和 TheRundown 均可能失敗 |
| 免費額度不足 | 🟡 中 | 500 次/月（The Odds API）對完整賽季吃緊，需優化查詢 |
| 台灣運彩官网封鎖 | 🔴 高 | 目前無法直連，需代理方案 |
| 美國赔率與台灣運彩差异 | 🟡 中 | 盘口計算方式不同，需轉換邏輯 |
| 住宅代理成本 | 🟢 低 | 若需台灣代理，$15-50/月屬合理範圍 |

---

## 7. 結論與行動建議

**✅ 主要結論：**

1. **The Odds API 未被封鎖**，可直連使用，是最具成本效益的 MLB 赔率來源
2. **JBot API 未被封鎖**，但 MLB 免費版服務不穩定，需觀察或升級
3. **台灣運彩官網**無法直連，需台灣代理方案
4. **推薦採用「三源分層」策略**：JBot 主源 + The Odds API/TheRundown 參考備援

**🎯 下一步行動（按優先級）：**

| 優先級 | 行動 |
|---|---|
| **立即** | 至 the-odds-api.com 申請免費 API Key |
| **立即** | 至 therundown.io/api 申請免費 API Key |
| **本週** | 建立 `odds_retriever.py`：整合 The Odds API + JBot API |
| **本週** | 評估 JBot MLB 穩定性（每日觀察 1-2 週）|
| **評估中** | 若 JBot 不穩定：評估 Bright Data 台灣住宅代理方案 |

---

**附錄：相關文件**
- `taiwan_lottery_crawler.py` — 台灣運彩爬蟲框架
- `fetch_odds_api.py` — The Odds API 赔率抓取腳本（MLB）
- `daily_predictor.py` — 每日預測 + 赔率整合輸出
- `docs/Data-Research.md` — 數據研究策略
- `docs/Platoon-Splits-Strategy.md` — Platoon Splits 數據方案

---

## 8. The Odds API 整合實作（已落地）

### 8.1 抓取腳本

```bash
# 需先設定環境變數
export THE_ODDS_API_KEY=your_key

# 產出 JSON：data/odds/the-odds-api_YYYY-MM-DD.json
python fetch_odds_api.py --date 2026-03-24
```

**輸出格式（GameOdds）：**
- `market`: moneyline / spread / total
- `selection`: home / away / over / under
- `price`: American odds（+105 / -120）
- `line`: 讓分或大小分數值
- `source`: the_odds_api:bookmaker_key

### 8.2 daily_predictor 使用方式

```bash
# 同時使用台灣運彩 (DB) + The Odds API
python daily_predictor.py --date 2026-03-24 --odds-api

# 只使用 The Odds API 取代台灣運彩
python daily_predictor.py --date 2026-03-24 --odds-api only
```

**行為說明：**
- `--odds-api` 會讀取 `data/odds/the-odds-api_YYYY-MM-DD.json`
- 若 DB 無台灣運彩赔率，會自動使用 The Odds API
- `--odds-api only` 會直接取代台灣運彩赔率

### 8.3 限速與額度

- 免費版 **500 次/月**，建議每日只拉 1 次
- `fetch_odds_api.py` 預設若檔案已存在則跳過，可用 `--force` 強制更新
- API 回應 header 會顯示剩餘次數（x-requests-remaining）
