# MLB Betting System v2.0 開發清單

## 模型預測目標擴增
- [ ] 資料庫/ETL/特徵工程預留 label：`f5_home_win`（前五局勝負）、`over_under`（全場大小分）。
  - 規劃對應的結果欄位/標籤欄位與歷史資料回填流程。
- [ ] 針對 `f5_home_win` 建立獨立預測模型與訓練流程。
- [ ] 針對 `over_under` 建立獨立預測模型與訓練流程。

## LLM 分析摘要模組
- [ ] 在 `daily_predictor.py` 或 `discord_notifier.py` 預留函式：
  - 將高價值特徵（例：FIP、牛棚疲勞度）轉成 Prompt。
  - 呼叫 LLM 產出短評，並在送出 Discord Webhook 前附加到訊息內容。
