#!/usr/bin/env python3
"""Monthly backtest (retrain once per month) for MLB model.

Usage:
  python backtest_monthly.py --data ./data/training_2022_2025_enhanced_v6.csv \
    --start 2022-04-01 --end 2025-12-31 \
    --target home_win --out ./data/backtest_2022_2025_report.csv \
    --summary ./data/backtest_2022_2025_summary.json

Notes:
- If odds columns are not available, ROI assumes even odds (win=+1, loss=-1).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import model_trainer
from sklearn.ensemble import HistGradientBoostingClassifier


def american_to_decimal(odds: float) -> float:
    if odds == 0 or pd.isna(odds):
        return np.nan
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def calc_profit(is_win: bool, odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds):
        return 1.0 if is_win else -1.0
    dec = american_to_decimal(float(odds))
    if pd.isna(dec):
        return 1.0 if is_win else -1.0
    return (dec - 1.0) if is_win else -1.0


def _pick_game_date_column(df: pd.DataFrame) -> str:
    if "game_date" in df.columns:
        return "game_date"
    for cand in ("game_date_x", "game_date_y"):
        if cand in df.columns:
            return cand
    raise ValueError("No game_date column found in data")


def find_odds_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    candidates = [
        ("home_price", "away_price"),
        ("home_odds", "away_odds"),
        ("home_ml", "away_ml"),
    ]
    for home_col, away_col in candidates:
        if home_col in df.columns and away_col in df.columns:
            return home_col, away_col
    return None, None


@dataclass
class BacktestConfig:
    data_path: str
    target: str
    start: Optional[date]
    end: Optional[date]
    min_train_size: int
    bet_threshold: float
    out_csv: Optional[str]
    out_summary: Optional[str]


def month_range(start: date, end: date):
    cur = date(start.year, start.month, 1)
    while cur <= end:
        # month end
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        period_end = min(end, date(nxt.year, nxt.month, 1) - pd.Timedelta(days=1))
        yield cur, period_end
        cur = nxt


def load_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = _pick_game_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    if target not in df.columns:
        if target == "home_win" and "home_score" in df.columns and "away_score" in df.columns:
            df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
            df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
            df[target] = (df["home_score"] > df["away_score"]).astype(float)
        else:
            raise ValueError(f"Target column '{target}' not found in data")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()

    if date_col != "game_date":
        df["game_date"] = df[date_col]

    return df


def run_backtest(cfg: BacktestConfig) -> Tuple[pd.DataFrame, dict]:
    df = load_data(cfg.data_path, cfg.target)

    if cfg.start:
        df = df[df["game_date"] >= cfg.start]
    if cfg.end:
        df = df[df["game_date"] <= cfg.end]

    if df.empty:
        raise SystemExit("No data after date filtering")

    home_odds_col, away_odds_col = find_odds_columns(df)

    results = []

    for period_start, period_end in month_range(cfg.start, cfg.end):
        train_df = df[df["game_date"] < period_start]
        test_df = df[(df["game_date"] >= period_start) & (df["game_date"] <= period_end)]
        if len(train_df) < cfg.min_train_size or test_df.empty:
            continue

        feature_cols = model_trainer.infer_feature_columns(train_df, cfg.target)
        feature_cols = model_trainer.drop_leakage_columns(feature_cols)

        X_train = train_df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
        y_train = pd.to_numeric(train_df[cfg.target], errors="coerce").fillna(0).astype(int)

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=200,
            l2_regularization=0.0,
        )
        model.fit(X_train, y_train)

        X_test = test_df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
        probs = model.predict_proba(X_test)[:, 1]

        for offset, (_, row) in enumerate(test_df.iterrows()):
            prob = float(probs[offset])
            pick = "home" if prob >= 0.5 else "away"
            actual_home_win = int(row[cfg.target])
            is_win = (pick == "home" and actual_home_win == 1) or (pick == "away" and actual_home_win == 0)

            if prob >= cfg.bet_threshold or prob <= (1 - cfg.bet_threshold):
                odds = None
                if pick == "home" and home_odds_col:
                    odds = row.get(home_odds_col)
                if pick == "away" and away_odds_col:
                    odds = row.get(away_odds_col)
                profit = calc_profit(is_win, odds)
            else:
                profit = 0.0

            results.append(
                {
                    "game_date": row["game_date"],
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "home_win_prob": prob,
                    "pick": pick,
                    "actual_home_win": actual_home_win,
                    "is_win": int(is_win),
                    "profit": profit,
                }
            )

    report = pd.DataFrame(results)
    if report.empty:
        raise SystemExit("No backtest results (check min_train_size or date range)")

    total = len(report)
    accuracy = report["is_win"].mean()
    bet_df = report[report["profit"] != 0]
    n_bets = len(bet_df)
    win_rate = bet_df["is_win"].mean() if n_bets else np.nan
    roi = bet_df["profit"].sum() / n_bets if n_bets else np.nan

    summary = {
        "rows": int(total),
        "bet_count": int(n_bets),
        "accuracy": float(accuracy),
        "win_rate": float(win_rate) if n_bets else None,
        "roi": float(roi) if n_bets else None,
        "start": str(report["game_date"].min()),
        "end": str(report["game_date"].max()),
        "target": cfg.target,
        "data_path": cfg.data_path,
        "min_train_size": cfg.min_train_size,
        "bet_threshold": cfg.bet_threshold,
        "retrain_frequency": "monthly",
    }

    return report, summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", dest="data_path", required=True)
    p.add_argument("--target", default="home_win")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--min-train-size", type=int, default=300)
    p.add_argument("--bet-threshold", type=float, default=0.5)
    p.add_argument("--out", dest="out_csv", required=True)
    p.add_argument("--summary", dest="out_summary", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = BacktestConfig(
        data_path=args.data_path,
        target=args.target,
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        min_train_size=args.min_train_size,
        bet_threshold=args.bet_threshold,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
    )

    report, summary = run_backtest(cfg)

    out_path = Path(cfg.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    sum_path = Path(cfg.out_summary)
    sum_path.parent.mkdir(parents=True, exist_ok=True)
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
