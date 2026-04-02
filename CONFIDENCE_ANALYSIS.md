# MLB 模型信心維度改善報告（2026-04-02）

## 範圍
本次完成 4 個 phase：
1. 71 場已完賽追蹤資料的信心區間分析
2. LR 機率校準（isotonic / sigmoid）
3. `daily_predictor.py` 新增 `confidence_tier` 分層與 CSV 欄位
4. `backtest.py` 策略 A/B/C 回測（含 ROI）

---

## Phase 1：信心區間 vs 準確率（71 場）

資料來源：`data/performance_tracker.csv`（只取 `actual_outcome` 非空，共 71 場）

整體：41 勝 30 敗，**57.75%**

分箱規則（依 `ml_model_prob`）：`<40%`, `40-50%`, `50-60%`, `60-70%`, `>70%`

| Bin | 場次 | Bin 準確率 | 平均模型機率 | 相對整體(57.75%) |
|---|---:|---:|---:|---:|
| <40% | 0 | N/A | N/A | N/A |
| 40-50% | 3 | 0.00% | 49.25% | -57.75% |
| 50-60% | 51 | 60.78% | 56.92% | +3.04% |
| 60-70% | 15 | 53.33% | 64.74% | -4.41% |
| >70% | 2 | 100.00% | 70.99% | +42.25% |

### 判讀
- 當前樣本主要擠在 **50-70%**，和你描述一致。
- `>70%`、`40-50%` 有明顯偏離，但樣本太小（2/3 場），目前只能視為早期訊號。
- `60-70%` 並沒有比整體更好，代表「高機率 ≠ 高信心」現象確實存在。

---

## Phase 2：LR 機率校準（Calibration）

已在 `scripts/v10_lr_daily_predict.py` 導入：
- `CalibratedClassifierCV`
- `--calibration-method isotonic|sigmoid`
- `--calibration-cv`（預設 5）

核心實作：
```python
base_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("lr", LogisticRegression(max_iter=1000, C=1.0)),
])
calibrated = CalibratedClassifierCV(base_lr, method="isotonic", cv=5)
calibrated.fit(X_train, y_train)
probs = calibrated.predict_proba(X_test)[:, 1]
```

### 校準比較（`data/training_features_v10.csv` 時序切分）
- Train: 11012
- Test: 439

| 方法 | AUC | Brier(越低越好) | 機率範圍 |
|---|---:|---:|---:|
| Raw LR | 0.5690 | 0.2447 | 0.430 ~ 0.623 |
| Isotonic | 0.5675 | **0.2433** | **0.380 ~ 0.710** |
| Sigmoid(Platt) | 0.5693 | 0.2444 | 0.420 ~ 0.633 |

### 分層效果（Isotonic）
- HIGH（>65% 或 <35%）：3 場，**100%**
- MEDIUM：171 場，60.23%
- LOW（45-55%）：265 場，52.83%（接近 random）

> 結論：isotonic 在 AUC 幾乎持平下，改善 Brier 並拉開機率分布；雖 HIGH 樣本仍少，但已開始具備信心分層效果。

---

## Phase 3：`daily_predictor.py` 信心分層落地

已新增：
- `confidence_tier(prob)`
  - HIGH: `prob > 0.65 or prob < 0.35`
  - MEDIUM: `prob > 0.55 or prob < 0.45`
  - LOW: 其餘

### 已套用位置
1. Offline predictions CSV 輸出新增 `confidence_tier`
   - 檔案欄位現在包含：`... predicted_winner, confidence_tier, model_source`
2. 台彩 market rows 輸出新增 `confidence_tier`

驗證檔案：
- `data/results/predictions_2026-03-31_conf.csv`（`confidence_tier` 已存在）

---

## Phase 4：策略 A/B/C 回測（`backtest.py`）

`backtest.py` 已擴充：
- 自動偵測 tracker schema（`ml_model_prob + actual_outcome`）進入 tracker mode
- 支援 `confidence_tier` 與策略摘要輸出
- 支援策略 B：`|model_prob - market_prob| > threshold`
- 支援 `--market-odds-dir` 由 odds 快照回填缺失 market_prob（以 `date/home/away` 匹配，允許 ±2 天偏移）

執行：
```bash
python3 backtest.py \
  --data data/performance_tracker.csv \
  --start 2026-03-28 --end 2026-04-01 \
  --out data/results/confidence_backtest_report.csv \
  --summary data/results/confidence_backtest_summary.json \
  --market-odds-dir data/odds \
  --bet-threshold 0.5 \
  --high-threshold 0.65 --low-threshold 0.35 \
  --market-edge-threshold 0.10
```

### 回測結果
| 策略 | 條件 | 場次 | 準確率 | ROI（even odds） |
|---|---|---:|---:|---:|
| A | 只看 HIGH | 7 | 57.14% | 14.29% |
| B | `|模型-市場| > 10%` | 16 | **62.50%** | **25.00%** |
| C | 全部比賽 | 71 | 57.75% | 15.49% |

### 判讀
- 目前最佳為 **策略 B**（偏離市場 >10%），顯著優於全量策略。
- 策略 A 因 HIGH 樣本仍少（7 場），暫時沒有壓倒性優勢。

---

## 本次修改檔案
- `daily_predictor.py`
  - 新增 `confidence_tier(prob)`
  - Offline CSV 新增 `confidence_tier` 欄位
  - 台彩 market rows 新增 `confidence_tier` 欄位
- `scripts/v10_lr_daily_predict.py`
  - 加入 `CalibratedClassifierCV`（isotonic/sigmoid）
  - 輸出 raw vs calibrated（AUC/Brier/分層表現）
  - predictions CSV 新增 `confidence_tier`
- `backtest.py`
  - 新增 tracker mode
  - 新增策略 A/B/C 摘要與 ROI
  - 新增 odds snapshot 回填 market_prob

---

## 產出檔案
- `CONFIDENCE_ANALYSIS.md`（本報告）
- `data/results/confidence_backtest_report.csv`
- `data/results/confidence_backtest_summary.json`
- `data/results/predictions_2026-03-31_conf.csv`

