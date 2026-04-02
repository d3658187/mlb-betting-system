#!/usr/bin/env python3
"""Utilities for MLB performance tracker maintenance.

Responsibilities:
1) Keep tracker schema consistent.
2) Deduplicate rows by (date, home_team, away_team), preferring rows with ml_model_prob.
3) Upsert daily predictions into tracker.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TRACKER_COLUMNS = [
    "date",
    "game_id",
    "home_team",
    "away_team",
    "ml_model_prob",
    "market_prob",
    "actual_outcome",
    "correct_ml",
]

KEY_COLUMNS = ["date", "home_team", "away_team"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _empty_tracker() -> pd.DataFrame:
    return pd.DataFrame(columns=TRACKER_COLUMNS)


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_tracker()

    out = df.copy()
    for col in TRACKER_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["date"] = out["date"].astype(str)
    out["game_id"] = out["game_id"].astype(str)
    out["home_team"] = out["home_team"].astype(str)
    out["away_team"] = out["away_team"].astype(str)

    for col in ["ml_model_prob", "market_prob", "actual_outcome", "correct_ml"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out[TRACKER_COLUMNS]


def clean_tracker_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by key and keep row with non-null ml_model_prob first."""
    out = _ensure_schema(df)
    if out.empty:
        return out

    out = out.copy()
    out["_ml_priority"] = out["ml_model_prob"].notna().astype(int)
    out["_row_seq"] = np.arange(len(out))

    # Sort so drop_duplicates(keep='last') picks:
    # 1) row with ml_model_prob (priority=1)
    # 2) newest row when same priority
    out = out.sort_values(KEY_COLUMNS + ["_ml_priority", "_row_seq"])
    out = out.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    out = out.sort_values(["date", "home_team", "away_team", "_row_seq"]).drop(columns=["_ml_priority", "_row_seq"])

    return _ensure_schema(out)


def load_tracker(tracker_path: Path) -> pd.DataFrame:
    if not tracker_path.exists():
        return _empty_tracker()
    return _ensure_schema(pd.read_csv(tracker_path))


def save_tracker(df: pd.DataFrame, tracker_path: Path):
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_schema(df).to_csv(tracker_path, index=False)


def build_tracker_rows(
    target_date: str,
    games_df: pd.DataFrame,
    features_df: pd.DataFrame,
    market_probs: Optional[pd.Series] = None,
) -> pd.DataFrame:
    if games_df is None or games_df.empty or features_df is None or features_df.empty:
        return _empty_tracker()

    game_info = games_df[["game_id", "home_team_name", "away_team_name"]].drop_duplicates().copy()
    game_info["game_id"] = game_info["game_id"].astype(str)

    feature_probs = features_df[["game_id", "home_win_prob"]].copy()
    feature_probs["game_id"] = feature_probs["game_id"].astype(str)
    feature_probs = feature_probs.rename(columns={"home_win_prob": "ml_model_prob"})

    rows = game_info.merge(feature_probs, on="game_id", how="left")
    rows["date"] = str(target_date)

    if market_probs is not None and len(market_probs) > 0:
        rows["market_prob"] = rows["game_id"].map(market_probs)
    else:
        rows["market_prob"] = pd.NA

    rows["actual_outcome"] = pd.NA
    rows["correct_ml"] = pd.NA

    rows = rows.rename(columns={"home_team_name": "home_team", "away_team_name": "away_team"})
    return _ensure_schema(rows)


def upsert_tracker_rows(tracker_path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    existing = load_tracker(tracker_path)
    incoming = _ensure_schema(new_rows)
    if existing.empty:
        combined = incoming.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    cleaned = clean_tracker_dataframe(combined)
    save_tracker(cleaned, tracker_path)
    return cleaned


def clean_tracker_file(tracker_path: Path) -> tuple[int, int]:
    before_df = load_tracker(tracker_path)
    before_count = len(before_df)
    after_df = clean_tracker_dataframe(before_df)
    after_count = len(after_df)
    save_tracker(after_df, tracker_path)
    return before_count, after_count


def build_rows_from_predictions(predictions_path: Path) -> pd.DataFrame:
    pred = pd.read_csv(predictions_path)

    game_id_col = "game_id" if "game_id" in pred.columns else None
    date_col = "prediction_date" if "prediction_date" in pred.columns else ("date" if "date" in pred.columns else None)
    home_col = "home_team" if "home_team" in pred.columns else "home_team_name"
    away_col = "away_team" if "away_team" in pred.columns else "away_team_name"
    prob_col = "home_win_prob" if "home_win_prob" in pred.columns else "ml_model_prob"
    market_col = "market_home_prob" if "market_home_prob" in pred.columns else "market_prob"

    missing = [
        name
        for name, col in {
            "home_team": home_col,
            "away_team": away_col,
            "ml_model_prob": prob_col,
        }.items()
        if col is None or col not in pred.columns
    ]
    if missing:
        raise ValueError(f"Prediction file missing required fields: {missing}")

    rows = pd.DataFrame()
    rows["date"] = pred[date_col].astype(str) if date_col and date_col in pred.columns else pd.NA
    rows["game_id"] = pred[game_id_col].astype(str) if game_id_col and game_id_col in pred.columns else pd.NA
    rows["home_team"] = pred[home_col].astype(str)
    rows["away_team"] = pred[away_col].astype(str)
    rows["ml_model_prob"] = pd.to_numeric(pred[prob_col], errors="coerce")
    rows["market_prob"] = pd.to_numeric(pred[market_col], errors="coerce") if market_col in pred.columns else pd.NA
    rows["actual_outcome"] = pd.NA
    rows["correct_ml"] = pd.NA

    return _ensure_schema(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Deduplicate/normalize performance tracker")
    parser.add_argument("--tracker", default="data/performance_tracker.csv", help="Path to tracker CSV")
    parser.add_argument("--predictions", help="Optional predictions CSV to upsert into tracker")
    return parser.parse_args()


def main():
    args = parse_args()
    tracker_path = Path(args.tracker)

    if args.predictions:
        pred_path = Path(args.predictions)
        new_rows = build_rows_from_predictions(pred_path)
        if new_rows["date"].isna().all():
            raise ValueError("Predictions CSV must include 'prediction_date' (or 'date') column")
        updated = upsert_tracker_rows(tracker_path, new_rows)
        logging.info("Tracker upserted from %s -> %s (%d rows total)", pred_path, tracker_path, len(updated))
        return

    before_count, after_count = clean_tracker_file(tracker_path)
    logging.info("Tracker cleaned: %s | before=%d after=%d removed=%d", tracker_path, before_count, after_count, before_count - after_count)


if __name__ == "__main__":
    main()
