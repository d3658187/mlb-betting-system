
# MLB 投注預測之最優化數據策略研究報告 (初稿)

**報告目標：** 本報告旨在找出對 MLB 比賽結果預測最具價值的數據維度，分析現有數據策略，並提出具體改進建議，以提升預測模型的準確性。

**更新：** 已透過 Tavily API 完成網路研究。

---

## 1. MLB 預測研究與最佳實踐 (文獻回顧)

### FiveThirtyEight 預測系統核心發現

 FiveThirtyEight 的 MLB 預測系統採用 Elo 等級評分模型，核心調整因素包括：

| 因素 | 重要性 | 說明 |
|------|--------|------|
| **先發投手** | **最高** | 每位投手有獨立的 Elo 等級（如 Pedro Martinez 2000 年對紅襪 worth 109 Elo points） |
| 主場優勢 | 高 | 主場比賽固定加成 |
| 移動距離 | 中高 | 長途旅行後表現通常下滑 |
| 休息天數 | 中 | 休息不足影響投打表現 |

### 學術研究發現

機器學習研究（Li et al., 2022）收集 2015-2019 MLB 數據，發現：
- **SVM（支持向量機）在分類預測中表現最佳**
- 團隊整體數據比個人球員數據更具預測力
- 滾動窗口數據（近幾場表現）比賽季平均更有價值

### 業界最佳實踐

Baseball Prospectus 的 PECOTA 系統：
- 使用 `StuffPro` / `ArsenalPro` 度量投手球種與出手角度
- 左右打者 breakdown 是關鍵預測變數
- xwOBA 比傳統 wOBA 更具預測力

---

## 2. 現有數據資產分析

根據任務說明，目前系統已整合以下數據：

- **打者滾動統計 (Batter Rolling Stats)**：
  - **優點**：能夠捕捉球員近期的狀態（Hot/Cold streaks），比整個賽季的靜態數據更具即時性。
  - **潛在盲點**：滾動窗口的大小（例如，近7場 vs 近30場）對預測能力有顯著影響。需要確定最佳窗口期。
- **投手統計 (Pitcher Stats)**：
  - **分析**：這是預測比賽勝負最關鍵的因素之一。除了傳統的 ERA、WHIP 外，高階數據如 FIP (Fielding Independent Pitching)、xFIP (Expected FIP)、SIERA (Skill-Interactive ERA) 更能反映投手獨立於防守的真實能力。
  - **建議**：應優先使用 FIP/xFIP/SIERA，而非傳統 ERA。
- **牛棚統計 (Bullpen Stats)**：
  - **分析**：在比賽後期，牛棚實力往往是勝負手。應評估整個牛棚的綜合實力，而非單一救援投手。
  - **建議**：使用牛棚的綜合 FIP、K/9 (每九局三振)、BB/9 (每九局保送) 和 LOB% (殘壘率) 作為關鍵指標。
- **FanGraphs / Statcast 數據**：
  - **優點**：提供了最前沿的數據維度。
    - **Statcast - 投球**：投手球速 (Velocity)、轉速 (Spin Rate) 對於評估其壓制力至關重要。
    - **Statcast - 打擊**：打者擊球初速 (Exit Velocity)、擊球仰角 (Launch Angle) 是預測長打能力和整體進攻貢獻的核心。xwOBA (預期加權上壘率) 是一個非常有力的綜合指標。
  - **結論**：這些是當前系統中最具預測潛力的數據，應作為模型的核心特徵。
- **天氣 (Weather)**：
  - **分析**：風速和風向對全壘打數量有直接影響。氣溫和濕度也會影響球的飛行距離。
  - **建議**：需要將天氣數據與球場朝向結合，量化其對比賽總得分 (Over/Under) 的影響。
- **賠率 (Odds)**：
  - **分析**：賠率本身就隱含了市場對比賽結果的預測（Implied Probability）。它可以作為一個強大的特徵或用於校準模型的預測結果。

---

## 3. 核心預測數據維度識別（研究驗證）

基於 FiveThirtyEight、PECOTA 研究驗證，以下數據維度最具預測力：

- **投手層面**:
  - **核心**：xFIP, SIERA
  - **進階**：K-BB% (三振率減保送率), Ground Ball Rate (滾地球率), Pitch Velocity/Spin Rate
- **打者層面**:
  - **核心**：wOBA (加權上壘率), xwOBA (預期加權上壘率)
  - **進階**：Barrel %, Exit Velocity, Launch Angle, Plate Discipline (O-Swing%, Z-Swing%)
- **團隊層面**:
  - **牛棚**：Bullpen xFIP, K-BB%
  - **防守**：DRS (Defensive Runs Saved), UZR (Ultimate Zone Rating)

---

## 4. 數據缺口分析 (現有 vs. 理想)

根據研究驗證，目前系統的數據缺口如下：

- **投手 vs. 左/右手打者數據 (Platoon Splits)**：
  - **重要性**：**極高**
  - 很多投手對左、右打者的表現差異巨大
  - 建議：從 FanGraphs Splits Leaderboard 抓取

- **球場因素 (Park Factors)**：
  - **重要性**：**極高**
  - 不同球場的尺寸、海拔對得分影響天壤之別
  - 建議：使用 FanGraphs 主/客場分開統計

- **疲憊程度 (Fatigue Level)**：
  - **重要性**：**中高**
  - 先發投手近期投球數、牛棚投手近期出賽頻率
  - 已有部分牛棚數據，需加強投手疲憊追蹤

- **裁判因素 (Umpire Data)**：
  - **重要性**：**中等**
  - 主審好球帶傾向影響三振/保送
  - 付費數據源，優先級較低

---

## 5. 數據擴充建議 (來源與優先級)

### 第一優先級 (高影響力，易獲取)

| 數據類型 | 說明 | 來源 |
|----------|------|------|
| **Platoon Splits** | 投手對左/右打者表現差異 | FanGraphs Splits Leaderboard、Baseball-Reference |
| **Park Factors** | 球場對得分/全壘打的影響校正 | FanGraphs Team Batting Stats（主客場分開） |
| **xFIP / SIERA** | 投手真實能力（排除防守干擾） | FanGraphs Pitching Stats |

### 第二優先級 (中高影響力)

| 數據類型 | 說明 | 來源 |
|----------|------|------|
| **疲憊程度** | 先發投手近3場投球數、休息天數 | MLB Stats API（game_log） |
| **裁判好球帶** | 主審偏好影響三振/保送 |umpsessions.com（付費） |

### 第三優先級 (補充性)

| 數據類型 | 說明 | 來源 |
|----------|------|------|
| **防守數據 (DRS/UZR)** | 團隊防守能力 | FanGraphs Team Defense |

### FiveThirtyEight 的關鍵洞見

> 「先發投手對比賽結果的影響比其他任何因素都大。」

**投手 ID 不是重點，重點是投手的實力評級（Rating）**

現有系統的問題：
- 投手 ID 覆蓋率有限（新秀/春訓球員沒有歷史數據）
- 需要建立投手的 **實力評級**，而不只是依賴 ID

**建議：**
1. 用 xFIP/FIP 作為投手實力的核心特徵（比 ERA 更穩定）
2. 建立投手滾動實力評級（類似 FiveThirtyEight 的 Elo）
3. 針對「未知投手」建立降級模型（team-level fallback）

---
**研究來源：**
- FiveThirtyEight MLB Predictions Methodology (https://fivethirtyeight.com/methodology/how-our-mlb-predictions-work/)
- Baseball Prospectus PECOTA 2026
- Li et al. (2022) "Exploring and Selecting Features to Predict the Next Game"
