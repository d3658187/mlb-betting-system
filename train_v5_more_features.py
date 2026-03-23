#!/usr/bin/env python3
"""Train v5 model with richer feature set (rolling + batting + pitching + bullpen + H2H).

Usage:
  python train_v5_more_features.py \
    --csv ./data/pybaseball/historical_features_v4.csv \
    --target home_win \
    --out-dir ./models \
    --model-name mlb_v5_rich
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import model_trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="./data/pybaseball/historical_features_v4.csv")
    p.add_argument("--target", default="home_win")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--out-dir", default="./models")
    p.add_argument("--model-name", default="mlb_v5_rich")
    p.add_argument("--h2h-windows", default="5,10", help="comma-separated windows")
    return p.parse_args()


def ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in df.columns:
        return df
    for cand in ("game_date_x", "game_date_y"):
        if cand in df.columns:
            df = df.copy()
            df["game_date"] = df[cand]
            return df
    raise ValueError("game_date not found in dataset")


def add_h2h_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    df = ensure_game_date(df)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values("game_date")

    # Directional matchup (home vs away). This avoids mixing venue effects.
    matchup_key = df["home_team"].astype(str) + "__" + df["away_team"].astype(str)
    df["matchup_key"] = matchup_key

    for w in windows:
        col = f"h2h_home_win_pct_{w}"
        df[col] = (
            df.groupby("matchup_key")["home_win"]
            .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
    return df.drop(columns=["matchup_key"])


def build_feature_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    def exists(cols):
        return [c for c in cols if c in df.columns]

    starter_cols = exists([
        "home_starter_era", "home_starter_whip", "home_starter_k_pct", "home_starter_bb_pct",
        "home_starter_kbb_pct", "home_starter_fip", "home_starter_xfip", "home_starter_siera",
        "home_starter_war", "home_starter_ip",
        "away_starter_era", "away_starter_whip", "away_starter_k_pct", "away_starter_bb_pct",
        "away_starter_kbb_pct", "away_starter_fip", "away_starter_xfip", "away_starter_siera",
        "away_starter_war", "away_starter_ip",
    ])

    bullpen_cols = exists([
        "home_bp_ERA", "home_bp_WHIP", "home_bp_K/9", "home_bp_BB/9", "home_bp_K%", "home_bp_BB%", "home_bp_FIP",
        "away_bp_ERA", "away_bp_WHIP", "away_bp_K/9", "away_bp_BB/9", "away_bp_K%", "away_bp_BB%", "away_bp_FIP",
        "home_bullpen_prev_runs_allowed", "home_bullpen_ra_last3",
        "away_bullpen_prev_runs_allowed", "away_bullpen_ra_last3",
    ])

    batting_cols = exists([
        "home_bat_AVG", "home_bat_OBP", "home_bat_SLG", "home_bat_OPS", "home_bat_ISO",
        "home_bat_wOBA", "home_bat_wRC+", "home_bat_BB%", "home_bat_K%", "home_bat_HR", "home_bat_R",
        "away_bat_AVG", "away_bat_OBP", "away_bat_SLG", "away_bat_OPS", "away_bat_ISO",
        "away_bat_wOBA", "away_bat_wRC+", "away_bat_BB%", "away_bat_K%", "away_bat_HR", "away_bat_R",
    ])

    pitching_cols = exists([
        "home_pit_ERA", "home_pit_FIP", "home_pit_xFIP", "home_pit_SIERA", "home_pit_WHIP",
        "home_pit_K%", "home_pit_BB%", "home_pit_K/9", "home_pit_BB/9", "home_pit_HR/9",
        "away_pit_ERA", "away_pit_FIP", "away_pit_xFIP", "away_pit_SIERA", "away_pit_WHIP",
        "away_pit_K%", "away_pit_BB%", "away_pit_K/9", "away_pit_BB/9", "away_pit_HR/9",
    ])

    rolling_cols = exists([
        "home_roll5_run_diff_mean", "home_roll15_run_diff_mean", "home_roll30_run_diff_mean",
        "home_roll5_runs_scored_mean", "home_roll5_runs_allowed_mean", "home_roll5_win_mean",
        "home_roll15_runs_scored_mean", "home_roll15_runs_allowed_mean", "home_roll15_win_mean",
        "home_roll30_runs_scored_mean", "home_roll30_runs_allowed_mean", "home_roll30_win_mean",
        "away_roll5_run_diff_mean", "away_roll15_run_diff_mean", "away_roll30_run_diff_mean",
        "away_roll5_runs_scored_mean", "away_roll5_runs_allowed_mean", "away_roll5_win_mean",
        "away_roll15_runs_scored_mean", "away_roll15_runs_allowed_mean", "away_roll15_win_mean",
        "away_roll30_runs_scored_mean", "away_roll30_runs_allowed_mean", "away_roll30_win_mean",
        "home_ha_roll5_runs_scored_mean", "home_ha_roll5_runs_allowed_mean", "home_ha_roll5_win_mean",
        "home_ha_roll15_runs_scored_mean", "home_ha_roll15_runs_allowed_mean", "home_ha_roll15_win_mean",
        "home_ha_roll30_runs_scored_mean", "home_ha_roll30_runs_allowed_mean", "home_ha_roll30_win_mean",
        "away_ha_roll5_runs_scored_mean", "away_ha_roll5_runs_allowed_mean", "away_ha_roll5_win_mean",
        "away_ha_roll15_runs_scored_mean", "away_ha_roll15_runs_allowed_mean", "away_ha_roll15_win_mean",
        "away_ha_roll30_runs_scored_mean", "away_ha_roll30_runs_allowed_mean", "away_ha_roll30_win_mean",
        "home_rest_days", "away_rest_days",
    ])

    h2h_cols = exists([c for c in df.columns if c.startswith("h2h_home_win_pct_")])

    # Build diff features for core stats
    diff_pairs = [
        ("home_starter_era", "away_starter_era", "diff_starter_era"),
        ("home_starter_whip", "away_starter_whip", "diff_starter_whip"),
        ("home_starter_kbb_pct", "away_starter_kbb_pct", "diff_starter_kbb_pct"),
        ("home_bat_wOBA", "away_bat_wOBA", "diff_bat_wOBA"),
        ("home_bat_wRC+", "away_bat_wRC+", "diff_bat_wRC+"),
        ("home_bat_OPS", "away_bat_OPS", "diff_bat_OPS"),
        ("home_pit_FIP", "away_pit_FIP", "diff_pit_FIP"),
        ("home_pit_SIERA", "away_pit_SIERA", "diff_pit_SIERA"),
        ("home_roll15_run_diff_mean", "away_roll15_run_diff_mean", "diff_roll15_run_diff_mean"),
    ]

    diff_cols = []
    for h, a, d in diff_pairs:
        if h in df.columns and a in df.columns:
            df[d] = pd.to_numeric(df[h], errors="coerce") - pd.to_numeric(df[a], errors="coerce")
            diff_cols.append(d)

    base_cols = starter_cols + bullpen_cols + batting_cols + pitching_cols + rolling_cols + h2h_cols + diff_cols
    base_cols = [c for c in base_cols if c in df.columns]

    # Guard against possible leakage in rolling features (if not shifted in source)
    filtered = []
    for c in base_cols:
        lc = c.lower()
        if "roll" in lc and ("run" in lc or "win" in lc or "score" in lc):
            continue
        if any(token in lc for token in ["score", "run_margin", "run_diff", "total_points"]):
            continue
        filtered.append(c)
    base_cols = filtered

    # Minimal fallbacks to ensure model can train
    if not base_cols:
        base_cols = model_trainer.infer_feature_columns(df, target="home_win")

    return df, base_cols


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("No training data found")

    if args.target in df.columns:
        df = df[df[args.target].notna()].copy()

    windows = [int(x) for x in args.h2h_windows.split(",") if x.strip()]
    if "home_win" in df.columns:
        df = add_h2h_features(df, windows)

    df, feature_cols = build_feature_cols(df)

    model, metrics, feature_cols, feature_importance = model_trainer.train_model(
        df, target=args.target, task=args.task, feature_cols=feature_cols
    )

    logging.info("Training metrics: %s", metrics)
    logging.info("Top feature importance: %s", feature_importance[:15])

    model_trainer.save_artifacts(
        model,
        metrics,
        feature_cols,
        feature_importance,
        out_dir=args.out_dir,
        target=args.target,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
