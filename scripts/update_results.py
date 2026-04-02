#!/usr/bin/env python3
"""Update performance tracker outcomes from MLB Stats API final scores.

Default target date is yesterday (local timezone).
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import update_tracker

MLB_SCHEDULE_API = "https://statsapi.mlb.com/api/v1/schedule"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def fetch_final_games(target_date: str, timeout: int = 20) -> List[Dict]:
    params = {
        "sportId": 1,
        "date": target_date,
    }
    response = requests.get(MLB_SCHEDULE_API, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json() or {}

    finals: List[Dict] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            status = (((game.get("status") or {}).get("abstractGameState")) or "").strip().lower()
            if status != "final":
                continue

            teams = game.get("teams") or {}
            home = teams.get("home") or {}
            away = teams.get("away") or {}
            home_team = ((home.get("team") or {}).get("name"))
            away_team = ((away.get("team") or {}).get("name"))
            home_score = home.get("score")
            away_score = away.get("score")

            if home_team is None or away_team is None:
                continue
            if home_score is None or away_score is None:
                continue

            home_score = int(home_score)
            away_score = int(away_score)
            if home_score == away_score:
                # MLB regular season should not tie, skip defensively.
                continue

            finals.append(
                {
                    "date": target_date,
                    "game_id": str(game.get("gamePk") or ""),
                    "home_team": home_team,
                    "away_team": away_team,
                    "actual_outcome": 1 if home_score > away_score else 0,
                }
            )

    return finals


def _calc_correct_ml(series: pd.Series, actual_outcome: int) -> pd.Series:
    probs = pd.to_numeric(series, errors="coerce")
    preds = probs >= 0.5
    correct = (preds.astype(float) == float(actual_outcome)).astype("float")
    correct[probs.isna()] = np.nan
    return correct


def update_tracker_results(tracker_df: pd.DataFrame, finals: List[Dict]) -> tuple[pd.DataFrame, int]:
    if tracker_df.empty or not finals:
        return tracker_df, 0

    updated_rows = 0
    out = tracker_df.copy()
    out["date"] = out["date"].astype(str)
    out["game_id"] = out["game_id"].astype(str)

    for game in finals:
        game_id = str(game["game_id"])
        target_date = str(game["date"])
        home_team = game["home_team"]
        away_team = game["away_team"]
        actual_outcome = int(game["actual_outcome"])

        mask_by_id = out["game_id"] == game_id
        mask_by_key = (
            (out["date"] == target_date)
            & (out["home_team"] == home_team)
            & (out["away_team"] == away_team)
        )
        mask = mask_by_id | mask_by_key

        if not mask.any():
            continue

        out.loc[mask, "actual_outcome"] = actual_outcome
        out.loc[mask, "correct_ml"] = _calc_correct_ml(out.loc[mask, "ml_model_prob"], actual_outcome)
        updated_rows += int(mask.sum())

    return out, updated_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Update tracker outcomes from MLB Stats API")
    parser.add_argument("--date", help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--tracker", default="data/performance_tracker.csv", help="Tracker CSV path")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else (date.today() - timedelta(days=1))
    target_date_str = target_date.isoformat()

    tracker_path = Path(args.tracker)
    tracker_df = update_tracker.load_tracker(tracker_path)
    if tracker_df.empty:
        logging.warning("Tracker is empty: %s", tracker_path)
        return

    finals = fetch_final_games(target_date_str, timeout=args.timeout)
    if not finals:
        logging.info("No final games returned for %s", target_date_str)
        return

    updated_df, updated_rows = update_tracker_results(tracker_df, finals)
    cleaned_df = update_tracker.clean_tracker_dataframe(updated_df)
    update_tracker.save_tracker(cleaned_df, tracker_path)

    logging.info(
        "Results updated for %s: finals=%d tracker_rows_updated=%d total_rows=%d",
        target_date_str,
        len(finals),
        updated_rows,
        len(cleaned_df),
    )


if __name__ == "__main__":
    main()
