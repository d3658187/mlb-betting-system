#!/usr/bin/env python3
"""Build full training dataset from DB (games + starting_pitchers + game_results)
and enrich with pitcher + team batting stats.

Usage:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  .venv/bin/python build_training_v6_dataset.py \
    --start-date 2022-01-01 --end-date 2024-12-31 \
    --out-base ./data/training_2022_2024.csv \
    --out ./data/training_2022_2024_enhanced_v6.csv \
    --pybaseball-dir ./data/pybaseball
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import create_engine, text


# ---------------------
# DB helpers
# ---------------------

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    return create_engine(db_url, pool_pre_ping=True)


def load_base_games(engine, start_date: str, end_date: str, require_results: bool = True) -> pd.DataFrame:
    where_clause = "g.game_date BETWEEN :start_date AND :end_date"
    if require_results:
        where_clause += " AND r.home_score IS NOT NULL AND r.away_score IS NOT NULL"

    sql = text(
        f"""
        SELECT g.game_date,
               th.abbreviation AS home_team,
               ta.abbreviation AS away_team,
               sp_home.pitcher_mlb_id AS home_pitcher_mlbam,
               sp_away.pitcher_mlb_id AS away_pitcher_mlbam,
               EXTRACT(YEAR FROM g.game_date)::int AS season,
               r.home_score AS home_runs,
               r.away_score AS away_runs,
               r.home_win AS home_win
          FROM games g
          LEFT JOIN teams th ON th.id = g.home_team_id
          LEFT JOIN teams ta ON ta.id = g.away_team_id
          LEFT JOIN starting_pitchers sp_home
                 ON sp_home.game_id = g.mlb_game_id AND sp_home.is_home = TRUE
          LEFT JOIN starting_pitchers sp_away
                 ON sp_away.game_id = g.mlb_game_id AND sp_away.is_home = FALSE
          LEFT JOIN game_results r ON r.game_id = g.id
         WHERE {where_clause}
        """
    )
    df = pd.read_sql(sql, engine, params={"start_date": start_date, "end_date": end_date})

    if df.empty:
        return df

    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    for col in ["home_team", "away_team"]:
        df[col] = df[col].astype(str)
    for col in ["home_pitcher_mlbam", "away_pitcher_mlbam"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    # Drop rows missing team abbreviations
    df = df.dropna(subset=["home_team", "away_team"])
    df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")

    return df


# ---------------------
# CSV stats loaders
# ---------------------

def load_pitcher_stats(pybaseball_dir: Path, season: int) -> pd.DataFrame:
    p = pybaseball_dir / f"pitcher_stats_{season}.csv"
    df = pd.read_csv(p)
    keep = [
        "mlbam_id", "ERA", "WHIP", "K%", "BB%", "K-BB%", "FIP", "xFIP", "SIERA", "WAR", "IP"
    ]
    df = df[keep].copy()
    df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce").astype("Int64")
    return df


def load_team_batting(pybaseball_dir: Path, season: int) -> pd.DataFrame:
    p = pybaseball_dir / f"team_batting_{season}.csv"
    df = pd.read_csv(p)
    keep = [
        "team", "AVG", "OBP", "SLG", "OPS", "ISO", "wOBA", "wRC+", "BB%", "K%", "R", "HR", "SB"
    ]
    df = df[keep].copy()
    df = df[df["team"].notna()]
    df = df[df["team"] != "- - -"]
    return df


# ---------------------
# Rolling / H2H features
# ---------------------


def _build_team_game_log(base: pd.DataFrame) -> pd.DataFrame:
    games = base[[
        "game_date",
        "home_team",
        "away_team",
        "home_runs",
        "away_runs",
        "home_win",
        "season",
    ]].copy()

    games["game_key"] = (
        games["game_date"].astype(str)
        + "_"
        + games["home_team"].astype(str)
        + "_"
        + games["away_team"].astype(str)
    )

    home = games.copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["runs_scored"] = pd.to_numeric(home["home_runs"], errors="coerce")
    home["runs_allowed"] = pd.to_numeric(home["away_runs"], errors="coerce")
    home["win"] = pd.to_numeric(home["home_win"], errors="coerce")
    home["is_home"] = True

    away = games.copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["runs_scored"] = pd.to_numeric(away["away_runs"], errors="coerce")
    away["runs_allowed"] = pd.to_numeric(away["home_runs"], errors="coerce")
    away["win"] = 1 - pd.to_numeric(away["home_win"], errors="coerce")
    away["is_home"] = False

    team_game = pd.concat([home, away], ignore_index=True)
    team_game["run_diff"] = team_game["runs_scored"] - team_game["runs_allowed"]
    team_game = team_game.sort_values(["team", "game_date", "game_key"]).reset_index(drop=True)
    return team_game


def attach_rolling_features(base: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    if base.empty:
        return base

    team_game = _build_team_game_log(base)
    grouped = team_game.groupby("team", group_keys=False)

    metrics = ["runs_scored", "runs_allowed", "run_diff", "win"]
    feat = team_game[["game_key", "team", "game_date", "is_home"]].copy()
    for col in metrics:
        rolled_mean = (
            grouped[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        feat[f"roll{window}_{col}_mean"] = rolled_mean.values

    # home/away split
    grouped_ha = team_game.groupby(["team", "is_home"], group_keys=False)
    for col in ["runs_scored", "runs_allowed", "win"]:
        rolled = (
            grouped_ha[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=[0, 1], drop=True)
        )
        feat[f"ha_roll{window}_{col}_mean"] = rolled.values

    # merge to base
    base = base.copy()
    base["game_key"] = (
        base["game_date"].astype(str)
        + "_"
        + base["home_team"].astype(str)
        + "_"
        + base["away_team"].astype(str)
    )

    home_feat = feat[feat["is_home"]].drop(columns=["is_home"], errors="ignore")
    away_feat = feat[~feat["is_home"]].drop(columns=["is_home"], errors="ignore")

    home_feat = home_feat.rename(
        columns={c: f"home_{c}" for c in home_feat.columns if c not in {"game_key", "team", "game_date"}}
    )
    away_feat = away_feat.rename(
        columns={c: f"away_{c}" for c in away_feat.columns if c not in {"game_key", "team", "game_date"}}
    )

    base = base.merge(home_feat, how="left", left_on=["game_key", "home_team"], right_on=["game_key", "team"]).drop(columns=["team"], errors="ignore")
    base = base.merge(away_feat, how="left", left_on=["game_key", "away_team"], right_on=["game_key", "team"]).drop(columns=["team"], errors="ignore")

    return base


def attach_h2h_features(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return base

    team_game = _build_team_game_log(base)
    team_game = team_game.sort_values(["team", "opponent", "game_date", "game_key"]).reset_index(drop=True)

    group = team_game.groupby(["team", "opponent"], group_keys=False)
    team_game["h2h_games"] = group.cumcount()

    team_game["win_num"] = pd.to_numeric(team_game["win"], errors="coerce").fillna(0)
    team_game["runs_scored_num"] = pd.to_numeric(team_game["runs_scored"], errors="coerce").fillna(0)
    team_game["runs_allowed_num"] = pd.to_numeric(team_game["runs_allowed"], errors="coerce").fillna(0)

    team_game["h2h_wins"] = group["win_num"].cumsum().shift(1)
    team_game["h2h_runs_scored_sum"] = group["runs_scored_num"].cumsum().shift(1)
    team_game["h2h_runs_allowed_sum"] = group["runs_allowed_num"].cumsum().shift(1)

    team_game["h2h_win_pct"] = team_game["h2h_wins"] / team_game["h2h_games"].replace(0, pd.NA)
    team_game["h2h_runs_scored_avg"] = team_game["h2h_runs_scored_sum"] / team_game["h2h_games"].replace(0, pd.NA)
    team_game["h2h_runs_allowed_avg"] = team_game["h2h_runs_allowed_sum"] / team_game["h2h_games"].replace(0, pd.NA)

    feat = team_game[[
        "game_key",
        "team",
        "h2h_games",
        "h2h_win_pct",
        "h2h_runs_scored_avg",
        "h2h_runs_allowed_avg",
    ]].copy()

    base = base.copy()
    base["game_key"] = (
        base["game_date"].astype(str)
        + "_"
        + base["home_team"].astype(str)
        + "_"
        + base["away_team"].astype(str)
    )

    home_feat = feat.rename(
        columns={c: f"home_{c}" for c in feat.columns if c not in {"game_key", "team", "game_date"}}
    )
    away_feat = feat.rename(
        columns={c: f"away_{c}" for c in feat.columns if c not in {"game_key", "team", "game_date"}}
    )

    base = base.merge(home_feat, how="left", left_on=["game_key", "home_team"], right_on=["game_key", "team"]).drop(columns=["team"], errors="ignore")
    base = base.merge(away_feat, how="left", left_on=["game_key", "away_team"], right_on=["game_key", "team"]).drop(columns=["team"], errors="ignore")

    return base


# ---------------------
# Enrichment
# ---------------------

def enrich_with_stats(base: pd.DataFrame, py_dir: Path) -> pd.DataFrame:
    base = base.copy()
    base["season"] = pd.to_numeric(base["season"], errors="coerce").astype("Int64")
    seasons = sorted(base["season"].dropna().unique())

    pitcher_map: Dict[int, pd.DataFrame] = {}
    batting_map: Dict[int, pd.DataFrame] = {}

    for season in seasons:
        try:
            pitcher_map[season] = load_pitcher_stats(py_dir, season)
            batting_map[season] = load_team_batting(py_dir, season)
        except FileNotFoundError:
            continue

    enriched_parts = []
    for season in seasons:
        if season not in pitcher_map or season not in batting_map:
            continue
        df = base[base["season"] == season].copy()

        pit = pitcher_map[season]
        bat = batting_map[season]

        df["home_pitcher_mlbam"] = pd.to_numeric(df["home_pitcher_mlbam"], errors="coerce").astype("Int64")
        df["away_pitcher_mlbam"] = pd.to_numeric(df["away_pitcher_mlbam"], errors="coerce").astype("Int64")

        # Join pitcher stats
        df = df.merge(pit, how="left", left_on="home_pitcher_mlbam", right_on="mlbam_id")
        df = df.rename(columns={
            "ERA": "home_p_ERA", "WHIP": "home_p_WHIP", "K%": "home_p_K%", "BB%": "home_p_BB%",
            "K-BB%": "home_p_K-BB%", "FIP": "home_p_FIP", "xFIP": "home_p_xFIP", "SIERA": "home_p_SIERA",
            "WAR": "home_p_WAR", "IP": "home_p_IP",
        }).drop(columns=["mlbam_id"])

        df = df.merge(pit, how="left", left_on="away_pitcher_mlbam", right_on="mlbam_id")
        df = df.rename(columns={
            "ERA": "away_p_ERA", "WHIP": "away_p_WHIP", "K%": "away_p_K%", "BB%": "away_p_BB%",
            "K-BB%": "away_p_K-BB%", "FIP": "away_p_FIP", "xFIP": "away_p_xFIP", "SIERA": "away_p_SIERA",
            "WAR": "away_p_WAR", "IP": "away_p_IP",
        }).drop(columns=["mlbam_id"])

        # Normalize team keys for merge
        df["home_team"] = df["home_team"].astype(str)
        df["away_team"] = df["away_team"].astype(str)

        # Join team batting
        df = df.merge(bat, how="left", left_on="home_team", right_on="team")
        df = df.rename(columns={
            "AVG": "home_bat_AVG", "OBP": "home_bat_OBP", "SLG": "home_bat_SLG", "OPS": "home_bat_OPS",
            "ISO": "home_bat_ISO", "wOBA": "home_bat_wOBA", "wRC+": "home_bat_wRC+",
            "BB%": "home_bat_BB%", "K%": "home_bat_K%", "R": "home_bat_R", "HR": "home_bat_HR", "SB": "home_bat_SB",
        }).drop(columns=["team"])

        df = df.merge(bat, how="left", left_on="away_team", right_on="team")
        df = df.rename(columns={
            "AVG": "away_bat_AVG", "OBP": "away_bat_OBP", "SLG": "away_bat_SLG", "OPS": "away_bat_OPS",
            "ISO": "away_bat_ISO", "wOBA": "away_bat_wOBA", "wRC+": "away_bat_wRC+",
            "BB%": "away_bat_BB%", "K%": "away_bat_K%", "R": "away_bat_R", "HR": "away_bat_HR", "SB": "away_bat_SB",
        }).drop(columns=["team"])

        # Diff features
        diff_pairs = [
            ("home_p_ERA", "away_p_ERA", "diff_p_ERA"),
            ("home_p_WHIP", "away_p_WHIP", "diff_p_WHIP"),
            ("home_p_K-BB%", "away_p_K-BB%", "diff_p_K-BB%"),
            ("home_bat_wOBA", "away_bat_wOBA", "diff_bat_wOBA"),
            ("home_bat_wRC+", "away_bat_wRC+", "diff_bat_wRC+"),
            ("home_bat_OPS", "away_bat_OPS", "diff_bat_OPS"),
        ]
        for h, a, d in diff_pairs:
            df[d] = pd.to_numeric(df[h], errors="coerce") - pd.to_numeric(df[a], errors="coerce")

        enriched_parts.append(df)

    if not enriched_parts:
        return pd.DataFrame()

    return pd.concat(enriched_parts, ignore_index=True)


def fill_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    exclude_cols = {"home_runs", "away_runs", "home_win"}
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude_cols]

    for col in numeric_cols:
        series = df[col]
        if series.isna().all():
            df[col] = 0
        else:
            df[col] = series.fillna(series.median())
    return df


# ---------------------
# CLI
# ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--out-base", default="./data/training_2022_2024.csv")
    p.add_argument("--out", default="./data/training_2022_2024_enhanced_v6.csv")
    p.add_argument("--pybaseball-dir", default="./data/pybaseball")
    p.add_argument("--no-fillna", action="store_true", help="Do not fill missing numeric features")
    p.add_argument("--no-results-filter", action="store_true", help="Include games without results")
    return p.parse_args()


def main():
    args = parse_args()
    engine = get_engine()

    base = load_base_games(engine, args.start_date, args.end_date, require_results=not args.no_results_filter)
    if base.empty:
        raise SystemExit("No base games found for given date range")

    out_base = Path(args.out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out_base, index=False)
    print(f"Wrote base: {out_base} rows={len(base)} cols={len(base.columns)}")

    py_dir = Path(args.pybaseball_dir)

    base_with_roll = attach_rolling_features(base, window=5)
    base_with_h2h = attach_h2h_features(base_with_roll)

    enriched = enrich_with_stats(base_with_h2h, py_dir)
    if enriched.empty:
        raise SystemExit("Enriched dataset is empty (missing pybaseball stats for seasons?)")

    if not args.no_fillna:
        enriched = fill_missing_features(enriched)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"Wrote enriched: {out_path} rows={len(enriched)} cols={len(enriched.columns)}")


if __name__ == "__main__":
    main()
