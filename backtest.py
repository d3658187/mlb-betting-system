#!/usr/bin/env python3
"""Historical backtest for MLB model.

Modes:
1) Walk-forward mode (default):
   - Train on games before each date, test on current date.
2) Tracker mode (auto-detected for performance_tracker schema):
   - Uses existing per-game model probabilities directly.

Outputs:
- Detailed per-game report (CSV)
- Summary JSON, including strategy A/B/C breakdown:
  A) HIGH confidence only
  B) |model_prob - market_prob| > edge threshold
  C) All games

Usage:
  python backtest.py --data ./data/pybaseball/historical_features_v4.csv --start 2023-03-30 --end 2025-10-01 \
    --target home_win --out ./data/backtest_report.csv --summary ./data/backtest_summary.json

  python backtest.py --data ./data/performance_tracker.csv --start 2026-03-28 --end 2026-04-01 \
    --out ./data/results/confidence_backtest_report.csv --summary ./data/results/confidence_backtest_summary.json \
    --market-odds-dir ./data/odds

Notes:
- If odds columns are not available, ROI assumes even odds (win=+1, loss=-1).
- Supports American odds if columns found (home_price/away_price or home_odds/away_odds).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

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
    high_threshold: float = 0.65
    low_threshold: float = 0.35
    market_edge_threshold: float = 0.10
    market_odds_dir: Optional[str] = None


# ---------------------
# Odds helpers
# ---------------------

def american_to_decimal(odds: float) -> float:
    if odds == 0 or pd.isna(odds):
        return np.nan
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def implied_prob_from_american(odds: float) -> float:
    if odds == 0 or pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def calc_profit(is_win: bool, odds: Optional[float]) -> float:
    if odds is None or pd.isna(odds):
        return 1.0 if is_win else -1.0
    dec = american_to_decimal(float(odds))
    if pd.isna(dec):
        return 1.0 if is_win else -1.0
    return (dec - 1.0) if is_win else -1.0


def market_home_prob_from_american(home_odds: Optional[float], away_odds: Optional[float]) -> float:
    if home_odds is None or away_odds is None or pd.isna(home_odds) or pd.isna(away_odds):
        return np.nan
    home_imp = implied_prob_from_american(float(home_odds))
    away_imp = implied_prob_from_american(float(away_odds))
    denom = home_imp + away_imp
    if pd.isna(denom) or denom <= 0:
        return home_imp
    return home_imp / denom


# ---------------------
# Confidence helpers
# ---------------------

def confidence_tier(prob: float, high_threshold: float = 0.65, low_threshold: float = 0.35) -> str:
    if pd.isna(prob):
        return "LOW"
    if prob > high_threshold or prob < low_threshold:
        return "HIGH"
    if prob > 0.55 or prob < 0.45:
        return "MEDIUM"
    return "LOW"


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


def detect_tracker_mode(path: str) -> bool:
    cols = pd.read_csv(path, nrows=1).columns.tolist()
    required = {"ml_model_prob", "actual_outcome"}
    return required.issubset(set(cols))


def _norm_team_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def build_market_prob_map_from_odds(odds_dir: str) -> Dict[Tuple[str, str, str], float]:
    base = Path(odds_dir)
    if not base.exists():
        logging.warning("market odds directory not found: %s", base)
        return {}

    files = sorted(base.glob("the-odds-api_*.json"))
    if not files:
        logging.warning("No odds JSON found under %s", base)
        return {}

    bucket: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    for fp in files:
        try:
            payload = json.loads(fp.read_text())
        except Exception as exc:
            logging.warning("Skip malformed odds file %s: %s", fp, exc)
            continue

        if not isinstance(payload, list):
            continue

        for game in payload:
            game_date = str(game.get("game_date") or "")
            home_team = game.get("home_team")
            away_team = game.get("away_team")
            markets = game.get("markets", [])

            if not game_date or not home_team or not away_team or not isinstance(markets, list):
                continue

            home_prices: List[float] = []
            away_prices: List[float] = []

            for item in markets:
                if str(item.get("market", "")).lower() != "moneyline":
                    continue
                sel = str(item.get("selection", "")).lower()
                price = pd.to_numeric(item.get("price"), errors="coerce")
                if pd.isna(price):
                    continue
                if sel == "home":
                    home_prices.append(float(price))
                elif sel == "away":
                    away_prices.append(float(price))

            if not home_prices or not away_prices:
                continue

            home_price = float(np.mean(home_prices))
            away_price = float(np.mean(away_prices))
            market_prob = market_home_prob_from_american(home_price, away_price)
            if pd.isna(market_prob):
                continue

            key = (game_date, _norm_team_name(home_team), _norm_team_name(away_team))
            bucket[key].append(float(market_prob))

    # Robust aggregate across snapshots/bookmakers
    return {k: float(pd.Series(v).median()) for k, v in bucket.items() if v}


def fill_tracker_market_prob(df: pd.DataFrame, odds_dir: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if "market_prob" not in out.columns:
        out["market_prob"] = np.nan

    out["market_prob"] = pd.to_numeric(out["market_prob"], errors="coerce")
    missing_mask = out["market_prob"].isna()
    if not missing_mask.any() or not odds_dir:
        return out

    prob_map = build_market_prob_map_from_odds(odds_dir)
    if not prob_map:
        return out

    offsets = [0, -1, 1, -2, 2]
    filled = 0

    for idx, row in out[missing_mask].iterrows():
        try:
            d = date.fromisoformat(str(row["date"]))
        except Exception:
            continue
        home = _norm_team_name(row.get("home_team"))
        away = _norm_team_name(row.get("away_team"))

        val = None
        for off in offsets:
            k = ((d + timedelta(days=off)).isoformat(), home, away)
            if k in prob_map:
                val = prob_map[k]
                break

        if val is not None:
            out.at[idx, "market_prob"] = float(val)
            filled += 1

    if filled:
        logging.info("Filled %d missing market_prob rows using odds snapshots", filled)

    return out


# ---------------------
# Summaries
# ---------------------

def _strategy_stats(df: pd.DataFrame, mask: pd.Series) -> dict:
    sub = df[mask].copy()
    if sub.empty:
        return {
            "games": 0,
            "bet_count": 0,
            "accuracy": None,
            "win_rate": None,
            "roi": None,
        }

    bet_df = sub[sub["profit"] != 0]
    n_bets = int(len(bet_df))

    return {
        "games": int(len(sub)),
        "bet_count": n_bets,
        "accuracy": float(sub["is_win"].mean()),
        "win_rate": float(bet_df["is_win"].mean()) if n_bets else None,
        "roi": float(bet_df["profit"].sum() / n_bets) if n_bets else None,
    }


def build_summary(report: pd.DataFrame, cfg: BacktestConfig, mode: str) -> dict:
    total = len(report)
    accuracy = report["is_win"].mean()

    bet_df = report[report["profit"] != 0]
    n_bets = len(bet_df)
    win_rate = bet_df["is_win"].mean() if n_bets else np.nan
    roi = bet_df["profit"].sum() / n_bets if n_bets else np.nan

    high_mask = report["confidence_tier"] == "HIGH"
    edge_mask = report["market_home_prob"].notna() & (report["abs_market_edge"] > cfg.market_edge_threshold)
    all_mask = pd.Series(True, index=report.index)

    summary = {
        "mode": mode,
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
        "high_threshold": cfg.high_threshold,
        "low_threshold": cfg.low_threshold,
        "market_edge_threshold": cfg.market_edge_threshold,
        "strategies": {
            "A_high_confidence": _strategy_stats(report, high_mask),
            "B_market_edge_gt_threshold": _strategy_stats(report, edge_mask),
            "C_all_games": _strategy_stats(report, all_mask),
        },
    }

    return summary


# ---------------------
# Backtest (tracker mode)
# ---------------------

def run_tracker_backtest(cfg: BacktestConfig) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(cfg.data_path)

    required = ["date", "home_team", "away_team", "ml_model_prob", "actual_outcome"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Tracker mode requires columns: {required}; missing={missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["ml_model_prob"] = pd.to_numeric(df["ml_model_prob"], errors="coerce")
    df["actual_outcome"] = pd.to_numeric(df["actual_outcome"], errors="coerce")

    if "correct_ml" in df.columns:
        df["correct_ml"] = pd.to_numeric(df["correct_ml"], errors="coerce")
    else:
        df["correct_ml"] = np.nan

    df = df.dropna(subset=["date", "ml_model_prob", "actual_outcome"]).copy()

    if cfg.start:
        df = df[df["date"] >= cfg.start]
    if cfg.end:
        df = df[df["date"] <= cfg.end]

    if df.empty:
        raise SystemExit("No tracker rows after date filtering")

    df = fill_tracker_market_prob(df, cfg.market_odds_dir)

    # Compute derived fields
    df["pick"] = np.where(df["ml_model_prob"] >= 0.5, "home", "away")
    df["actual_home_win"] = df["actual_outcome"].astype(int)

    computed_is_win = (
        ((df["pick"] == "home") & (df["actual_home_win"] == 1))
        | ((df["pick"] == "away") & (df["actual_home_win"] == 0))
    ).astype(int)

    df["is_win"] = pd.to_numeric(df["correct_ml"], errors="coerce").fillna(computed_is_win).astype(int)

    qualifies = (df["ml_model_prob"] >= cfg.bet_threshold) | (df["ml_model_prob"] <= (1 - cfg.bet_threshold))
    df["profit"] = np.where(qualifies, np.where(df["is_win"] == 1, 1.0, -1.0), 0.0)

    df["market_home_prob"] = pd.to_numeric(df.get("market_prob"), errors="coerce")
    df["abs_market_edge"] = (df["ml_model_prob"] - df["market_home_prob"]).abs()
    df["confidence_tier"] = df["ml_model_prob"].apply(
        lambda p: confidence_tier(p, high_threshold=cfg.high_threshold, low_threshold=cfg.low_threshold)
    )

    report = pd.DataFrame(
        {
            "game_date": df["date"],
            "home_team": df["home_team"],
            "away_team": df["away_team"],
            "home_win_prob": df["ml_model_prob"],
            "market_home_prob": df["market_home_prob"],
            "abs_market_edge": df["abs_market_edge"],
            "confidence_tier": df["confidence_tier"],
            "pick": df["pick"],
            "actual_home_win": df["actual_home_win"],
            "is_win": df["is_win"],
            "profit": df["profit"],
        }
    ).sort_values("game_date")

    summary = build_summary(report, cfg, mode="tracker")
    return report, summary


# ---------------------
# Backtest (walk-forward mode)
# ---------------------

def run_walkforward_backtest(cfg: BacktestConfig) -> Tuple[pd.DataFrame, dict]:
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

        for offset, (_, row) in enumerate(test_df.iterrows()):
            prob = float(probs[offset])
            pick = "home" if prob >= 0.5 else "away"
            actual_home_win = int(row[cfg.target])
            is_win = (pick == "home" and actual_home_win == 1) or (pick == "away" and actual_home_win == 0)

            odds = None
            if pick == "home" and home_odds_col:
                odds = row.get(home_odds_col)
            if pick == "away" and away_odds_col:
                odds = row.get(away_odds_col)

            market_home_prob = np.nan
            if "market_prob" in row.index:
                market_home_prob = pd.to_numeric(row.get("market_prob"), errors="coerce")
            elif home_odds_col and away_odds_col:
                market_home_prob = market_home_prob_from_american(row.get(home_odds_col), row.get(away_odds_col))

            qualifies = prob >= cfg.bet_threshold or prob <= (1 - cfg.bet_threshold)
            profit = calc_profit(is_win, odds) if qualifies else 0.0

            results.append(
                {
                    "game_date": row["game_date"],
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "home_win_prob": prob,
                    "market_home_prob": market_home_prob,
                    "abs_market_edge": abs(prob - market_home_prob) if pd.notna(market_home_prob) else np.nan,
                    "confidence_tier": confidence_tier(
                        prob,
                        high_threshold=cfg.high_threshold,
                        low_threshold=cfg.low_threshold,
                    ),
                    "pick": pick,
                    "actual_home_win": actual_home_win,
                    "is_win": int(is_win),
                    "profit": profit,
                }
            )

    report = pd.DataFrame(results)
    if report.empty:
        raise SystemExit("No backtest results (check min_train_size or date range)")

    summary = build_summary(report, cfg, mode="walkforward")
    return report, summary


def run_seasonal_walk_forward_report(
    data_path: str,
    target: str,
    feature_cols: Optional[List[str]],
    min_train_size: int,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "game_date" not in df.columns:
        raise SystemExit("Seasonal walk-forward requires 'game_date' column")
    if target not in df.columns:
        raise SystemExit(f"Seasonal walk-forward requires target column '{target}'")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date", target]).copy()
    if df.empty:
        raise SystemExit("No valid rows for seasonal walk-forward")

    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(int)
    df["season"] = df["game_date"].dt.year.astype(int)

    seasons = sorted(df["season"].unique())
    if len(seasons) < 2:
        raise SystemExit("Need at least 2 seasons for walk-forward")

    rows: List[dict] = []
    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for season in seasons[1:]:
        train_df = df[df["season"] < season].copy()
        test_df = df[df["season"] == season].copy()

        if len(train_df) < min_train_size or test_df.empty:
            continue

        if feature_cols:
            cols = [c for c in feature_cols if c in train_df.columns]
        else:
            cols = model_trainer.infer_feature_columns(train_df, target)
            cols = model_trainer.drop_leakage_columns(cols)

        if not cols:
            continue

        X_train = train_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_train = train_df[target].astype(int)
        X_test = test_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_test = test_df[target].astype(int)

        model = model_trainer.build_model("classification")
        model.fit(X_train, y_train)
        probs = np.clip(model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)

        preds = (probs >= 0.5).astype(int)

        auc = np.nan
        if y_test.nunique() >= 2:
            auc = float(roc_auc_score(y_test, probs))

        rows.append(
            {
                "season": int(season),
                "train_games": int(len(train_df)),
                "test_games": int(len(test_df)),
                "accuracy": float((preds == y_test.values).mean()),
                "brier_score": float(brier_score_loss(y_test, probs)),
                "log_loss": float(log_loss(y_test, probs, labels=[0, 1])),
                "auc": auc,
            }
        )

        all_probs.append(probs)
        all_y.append(y_test.values)

    if not rows:
        raise SystemExit("No seasonal walk-forward rows generated")

    y_all = np.concatenate(all_y)
    p_all = np.clip(np.concatenate(all_probs), 1e-6, 1 - 1e-6)
    auc_all = float(roc_auc_score(y_all, p_all)) if len(np.unique(y_all)) >= 2 else np.nan

    rows.append(
        {
            "season": "OVERALL",
            "train_games": int(np.sum([r["train_games"] for r in rows])),
            "test_games": int(len(y_all)),
            "accuracy": float(((p_all >= 0.5).astype(int) == y_all).mean()),
            "brier_score": float(brier_score_loss(y_all, p_all)),
            "log_loss": float(log_loss(y_all, p_all, labels=[0, 1])),
            "auc": auc_all,
        }
    )

    return pd.DataFrame(rows)


# ---------------------
# Entry
# ---------------------

def run_backtest(cfg: BacktestConfig) -> Tuple[pd.DataFrame, dict]:
    if detect_tracker_mode(cfg.data_path):
        logging.info("Detected tracker schema -> running tracker-mode backtest")
        return run_tracker_backtest(cfg)
    logging.info("Running walk-forward backtest")
    return run_walkforward_backtest(cfg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", dest="data_path", default="./data/pybaseball/historical_features_v4.csv")
    p.add_argument("--target", default="home_win")
    p.add_argument("--start", help="YYYY-MM-DD")
    p.add_argument("--end", help="YYYY-MM-DD")
    p.add_argument("--min-train-size", type=int, default=300)
    p.add_argument("--bet-threshold", type=float, default=0.5, help="Predict home if prob>=threshold; away if <=1-threshold")
    p.add_argument("--high-threshold", type=float, default=0.65, help="HIGH confidence if prob>=high_threshold or <=low_threshold")
    p.add_argument("--low-threshold", type=float, default=0.35, help="HIGH confidence if prob>=high_threshold or <=low_threshold")
    p.add_argument("--market-edge-threshold", type=float, default=0.10, help="Strategy B threshold: |model-market| > this value")
    p.add_argument("--market-odds-dir", default=None, help="Optional odds JSON directory used to fill tracker market_prob")
    p.add_argument("--out", dest="out_csv", help="Output CSV path for detailed report")
    p.add_argument("--summary", dest="out_summary", help="Output JSON path for summary")
    p.add_argument("--feature-cols", help="Comma-separated feature columns to use")
    p.add_argument("--feature-cols-file", help="JSON file containing feature columns list")
    p.add_argument("--seasonal-walk-forward", action="store_true", help="Run season-based walk-forward (train seasons 1..N-1, test season N)")
    p.add_argument("--walk-forward-report", default="data/results/walk_forward_report.csv", help="Seasonal walk-forward CSV output path")
    return p.parse_args()


def main():
    args = parse_args()
    feature_cols = None
    if args.feature_cols_file:
        feature_cols = json.loads(Path(args.feature_cols_file).read_text())
    elif args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    if args.seasonal_walk_forward:
        report_df = run_seasonal_walk_forward_report(
            data_path=args.data_path,
            target=args.target,
            feature_cols=feature_cols,
            min_train_size=args.min_train_size,
        )
        out_path = Path(args.walk_forward_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(out_path, index=False)
        logging.info("Saved seasonal walk-forward report to %s", out_path)
        return

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
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        market_edge_threshold=args.market_edge_threshold,
        market_odds_dir=args.market_odds_dir,
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
