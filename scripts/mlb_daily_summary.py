#!/usr/bin/env python3
"""Generate daily tracker summary text for Discord-friendly posting."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def _safe_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "N/A"
    return f"{(numerator / denominator) * 100:.2f}%"


def _calc_correct(df: pd.DataFrame) -> pd.Series:
    correct_col = pd.to_numeric(df.get("correct_ml"), errors="coerce")
    if correct_col.notna().any():
        return correct_col

    probs = pd.to_numeric(df.get("ml_model_prob"), errors="coerce")
    actual = pd.to_numeric(df.get("actual_outcome"), errors="coerce")
    pred_home = (probs >= 0.5).astype(float)
    out = (pred_home == actual).astype(float)
    out[probs.isna() | actual.isna()] = pd.NA
    return out


def build_summary_text(tracker_path: Path, target_date: date) -> str:
    if not tracker_path.exists():
        return f"📊 MLB Daily Summary ({target_date.isoformat()})\nTracker file not found: {tracker_path}"

    df = pd.read_csv(tracker_path)
    if df.empty:
        return f"📊 MLB Daily Summary ({target_date.isoformat()})\nTracker is empty."

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.date
    df["actual_outcome"] = pd.to_numeric(df.get("actual_outcome"), errors="coerce")
    df["ml_model_prob"] = pd.to_numeric(df.get("ml_model_prob"), errors="coerce")
    df["correct_ml"] = _calc_correct(df)

    day_df = df[df["date"] == target_date].copy()
    day_total = int(len(day_df))
    day_scored = day_df.dropna(subset=["actual_outcome", "ml_model_prob"])
    day_correct = int(pd.to_numeric(day_scored["correct_ml"], errors="coerce").fillna(0).sum())
    day_games = int(len(day_scored))

    all_scored = df.dropna(subset=["actual_outcome", "ml_model_prob"]).copy()
    all_correct = int(pd.to_numeric(all_scored["correct_ml"], errors="coerce").fillna(0).sum())
    all_games = int(len(all_scored))

    pending_today = day_total - day_games

    lines = [
        f"📊 MLB Daily Summary｜{target_date.isoformat()}",
        f"昨日預測場次: {day_total}",
        f"昨日已結算: {day_games}",
        f"昨日命中: {day_correct}",
        f"昨日勝率: {_safe_pct(day_correct, day_games)}",
    ]

    if pending_today > 0:
        lines.append(f"昨日待回填: {pending_today}")

    lines.extend(
        [
            "-" * 24,
            f"累計已結算場次: {all_games}",
            f"累計命中: {all_correct}",
            f"累計勝率: {_safe_pct(all_correct, all_games)}",
        ]
    )

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MLB daily summary from performance tracker")
    parser.add_argument("--date", help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--tracker", default="data/performance_tracker.csv", help="Tracker CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else (date.today() - timedelta(days=1))
    text = build_summary_text(Path(args.tracker), target_date)
    print(text)


if __name__ == "__main__":
    main()
