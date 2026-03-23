#!/usr/bin/env python3
"""Historical backtest for MLB model.

Features:
- Walk-forward training (train only on data before each game date)
- Computes accuracy, win rate, ROI
- Outputs detailed per-game report + summary

Usage:
  python backtest.py --data ./data/pybaseball/historical_features_v4.csv --start 2023-03-30 --end 2025-10-01 \
    --target home_win --out ./data/backtest_report.csv --summary ./data/backtest_summary.json

Notes:
- If odds columns are not available, ROI assumes even odds (win=+1, loss=-1).
- Supports American odds if columns found (home_price/away_price or home_odds/away_odds).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import model_trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


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
    feature_cols: Optional[List[str]] = None


# ---------------------
# Odds helpers
# ---------------------

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


# ---------------------
# Data helpers
# ---------------------

def _pick_game_date_column(df: pd.DataFrame) -> str:
    if "game_date" in df.columns:
        return "game_date"
    for cand in ("game_date_x", "game_date_y"):
        if cand in df.columns:
            return cand
    raise ValueError("No game_date column found in data")


def load_backtest_data(path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = _pick_game_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    # Normalize target column
    if target not in df.columns:
        if target == "home_win" and "home_score" in df.columns and "away_score" in df.columns:
            df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
            df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
            df[target] = (df["home_score"] > df["away_score"]).astype(float)
        else:
            raise ValueError(f"Target column '{target}' not found in data")

    df[target] = pd.to_numeric(df[target], errors="coerce")

    # Drop rows without target
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    # Keep canonical date column
    if date_col != "game_date":
        df["game_date"] = df[date_col]

    return df


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


# ---------------------
# Backtest
# ---------------------

def run_backtest(cfg: BacktestConfig) -> Tuple[pd.DataFrame, dict]:
    df = load_backtest_data(cfg.data_path, cfg.target)

    if cfg.start:
        df = df[df["game_date"] >= cfg.start]
    if cfg.end:
        df = df[df["game_date"] <= cfg.end]

    if df.empty:
        raise SystemExit("No data after date filtering")

    home_odds_col, away_odds_col = find_odds_columns(df)

    unique_dates = sorted(df["game_date"].unique())
    results: List[dict] = []

    for current_date in unique_dates:
        train_df = df[df["game_date"] < current_date]
        test_df = df[df["game_date"] == current_date]
        if len(train_df) < cfg.min_train_size or test_df.empty:
            continue

        if cfg.feature_cols:
            feature_cols = [c for c in cfg.feature_cols if c in train_df.columns]
        else:
            feature_cols = model_trainer.infer_feature_columns(train_df, cfg.target)
            feature_cols = model_trainer.drop_leakage_columns(feature_cols)

        X_train = train_df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
        y_train = pd.to_numeric(train_df[cfg.target], errors="coerce").fillna(0).astype(int)

        model = model_trainer.build_model("classification")
        model.fit(X_train, y_train)

        X_test = test_df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
        probs = model.predict_proba(X_test)[:, 1]

        for offset, (idx, row) in enumerate(test_df.iterrows()):
            prob = float(probs[offset])
            pick = "home" if prob >= 0.5 else "away"
            actual_home_win = int(row[cfg.target])
            is_win = (pick == "home" and actual_home_win == 1) or (pick == "away" and actual_home_win == 0)

            # Apply bet threshold
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

    # Bet stats (profit != 0)
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
    }

    return report, summary


# ---------------------
# CLI
# ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", dest="data_path", default="./data/pybaseball/historical_features_v4.csv")
    p.add_argument("--target", default="home_win")
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--min-train-size", type=int, default=300)
    p.add_argument("--bet-threshold", type=float, default=0.5, help="Predict home if prob>=threshold; away if <=1-threshold")
    p.add_argument("--out", dest="out_csv", help="Output CSV path for detailed report")
    p.add_argument("--summary", dest="out_summary", help="Output JSON path for summary")
    p.add_argument("--feature-cols", help="Comma-separated feature columns to use")
    p.add_argument("--feature-cols-file", help="JSON file containing feature columns list")
    return p.parse_args()


def main():
    args = parse_args()
    feature_cols = None
    if args.feature_cols_file:
        feature_cols = json.loads(Path(args.feature_cols_file).read_text())
    elif args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    cfg = BacktestConfig(
        data_path=args.data_path,
        target=args.target,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
        min_train_size=args.min_train_size,
        bet_threshold=args.bet_threshold,
        out_csv=args.out_csv,
        out_summary=args.out_summary,
        feature_cols=feature_cols,
    )

    report, summary = run_backtest(cfg)

    if cfg.out_csv:
        out_path = Path(cfg.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)
        logging.info("Saved report to %s", out_path)
    else:
        logging.info("No --out specified; report not saved.")

    if cfg.out_summary:
        out_path = Path(cfg.out_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        logging.info("Saved summary to %s", out_path)
    else:
        logging.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
