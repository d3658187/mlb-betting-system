#!/usr/bin/env python3
"""Bulk import MLB historical data using pybaseball.

Outputs (CSV by default):
- schedule/games (home/away, score, win)
- team batting season stats
- team pitching season stats

Usage:
  python import_historical_data.py --season-range 2023-2025 --out-dir ./data/pybaseball
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pybaseball import schedule_and_record, batting_stats, pitching_stats, team_ids
except Exception:  # pragma: no cover
    schedule_and_record = None
    batting_stats = None
    pitching_stats = None
    team_ids = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Fallback team abbreviations (used when team_ids() not available)
TEAM_ABBRS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]


# ---------------------
# Helpers
# ---------------------

def _safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _normalize_schedule_columns(df: pd.DataFrame, season: int) -> pd.DataFrame:
    df = df.copy()
    # Normalize date
    date_col = None
    for cand in ["Date", "date", "GAME_DATE", "game_date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        raise RuntimeError("schedule_and_record output missing date column")
    raw_dates = df[date_col].astype(str).str.replace(r"\s*\(.*\)$", "", regex=True)
    if not raw_dates.str.contains(r"\d{4}").any():
        raw_dates = raw_dates + f" {season}"
    df["game_date"] = pd.to_datetime(raw_dates, errors="coerce").dt.date

    # Normalize home/away
    ha_col = None
    for cand in ["Home/Away", "Home_Away", "home_away", "homeAway", "H/A"]:
        if cand in df.columns:
            ha_col = cand
            break
    if ha_col:
        df["is_home"] = df[ha_col].astype(str).str.upper().isin(["HOME", "H", "TRUE", "1"])
    else:
        # fallback: assume @ indicates away
        if "Opp" in df.columns:
            df["is_home"] = ~df["Opp"].astype(str).str.startswith("@")
        else:
            df["is_home"] = pd.NA

    # Normalize opponent/team
    team_col = None
    for cand in ["Tm", "Team", "team"]:
        if cand in df.columns:
            team_col = cand
            break
    opp_col = None
    for cand in ["Opp", "Opponent", "opp"]:
        if cand in df.columns:
            opp_col = cand
            break

    if team_col:
        df["team"] = df[team_col]
    if opp_col:
        df["opponent"] = df[opp_col].astype(str).str.replace("@", "", regex=False)

    # Normalize scores
    for cand in ["R", "Runs", "runs"]:
        if cand in df.columns:
            df["runs_scored"] = pd.to_numeric(df[cand], errors="coerce")
            break
    for cand in ["RA", "RunsAllowed", "runs_allowed", "R/RA"]:
        if cand in df.columns:
            df["runs_allowed"] = pd.to_numeric(df[cand], errors="coerce")
            break

    # Win/loss
    wl_col = None
    for cand in ["W/L", "WL", "Result", "result"]:
        if cand in df.columns:
            wl_col = cand
            break
    if wl_col:
        df["win"] = df[wl_col].astype(str).str.startswith("W")
    else:
        df["win"] = df["runs_scored"] > df["runs_allowed"]

    return df


def _load_team_abbrs() -> List[str]:
    if team_ids is None:
        return TEAM_ABBRS
    try:
        df = team_ids()
        for cand in ["abbrev", "abbreviation", "team_abbrev", "teamAbbrev", "Team", "team"]:
            if cand in df.columns:
                vals = df[cand].dropna().astype(str).tolist()
                if vals:
                    return sorted(set(vals))
    except Exception:
        pass
    return TEAM_ABBRS


# ---------------------
# Data loaders (pybaseball)
# ---------------------

def fetch_schedule_for_season(season: int) -> pd.DataFrame:
    if schedule_and_record is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")

    rows: List[pd.DataFrame] = []
    team_abbrs = _load_team_abbrs()

    for abbr in team_abbrs:
        try:
            df = schedule_and_record(season, abbr)
        except Exception as exc:
            logging.warning("schedule_and_record failed for %s %s: %s", season, abbr, exc)
            print(f"WARNING: Skipping data for {abbr} in {season} due to data error.")
            continue
        if df is None or df.empty:
            continue
        df = _normalize_schedule_columns(df, season)
        df["team_abbr"] = abbr
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    schedule = pd.concat(rows, ignore_index=True)

    # Build home games only to avoid duplicates (one row per game)
    home_games = schedule[schedule["is_home"] == True].copy()
    if home_games.empty:
        # fallback: build by de-dup on date + team/opponent
        home_games = schedule.copy()

    home_games["home_team"] = home_games["team"]
    home_games["away_team"] = home_games["opponent"]
    home_games["home_score"] = pd.to_numeric(home_games.get("runs_scored"), errors="coerce")
    home_games["away_score"] = pd.to_numeric(home_games.get("runs_allowed"), errors="coerce")
    home_games["home_win"] = home_games["home_score"] > home_games["away_score"]

    games = home_games[[
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
    ]].copy()
    games["season"] = season

    games = games.dropna(subset=["home_team", "away_team", "game_date"])
    games = games.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")

    return games


def fetch_team_batting(season: int) -> pd.DataFrame:
    if batting_stats is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")
    try:
        df = batting_stats(season, season, qual=0)
    except TypeError:
        df = batting_stats(season, qual=0)
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize team column
    team_col = None
    for cand in ["Team", "Tm", "team"]:
        if cand in df.columns:
            team_col = cand
            break
    if team_col is None:
        return pd.DataFrame()

    df = df.copy()
    df["team"] = df[team_col]
    df["season"] = season

    numeric_cols = [c for c in df.columns if c not in {team_col, "team", "season", "Name", "playerid", "ID"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate to team-level (sum counting stats, mean rate stats)
    agg_map: Dict[str, str] = {}
    for col in numeric_cols:
        if col.upper() in {"G", "AB", "H", "HR", "R", "RBI", "BB", "SO"}:
            agg_map[col] = "sum"
        else:
            agg_map[col] = "mean"

    team_df = df.groupby("team", as_index=False).agg(agg_map)
    team_df["season"] = season
    return team_df


def fetch_team_pitching(season: int) -> pd.DataFrame:
    if pitching_stats is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")
    try:
        df = pitching_stats(season, season, qual=0)
    except TypeError:
        df = pitching_stats(season, qual=0)
    if df is None or df.empty:
        return pd.DataFrame()

    team_col = None
    for cand in ["Team", "Tm", "team"]:
        if cand in df.columns:
            team_col = cand
            break
    if team_col is None:
        return pd.DataFrame()

    df = df.copy()
    df["team"] = df[team_col]
    df["season"] = season

    numeric_cols = [c for c in df.columns if c not in {team_col, "team", "season", "Name", "playerid", "ID"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    agg_map: Dict[str, str] = {}
    for col in numeric_cols:
        if col.upper() in {"G", "IP", "H", "ER", "HR", "BB", "SO", "R"}:
            agg_map[col] = "sum"
        else:
            agg_map[col] = "mean"

    team_df = df.groupby("team", as_index=False).agg(agg_map)
    team_df["season"] = season
    return team_df


# ---------------------
# Runner
# ---------------------

def run(seasons: Sequence[int], out_dir: str) -> None:
    _safe_mkdir(out_dir)

    all_games: List[pd.DataFrame] = []
    all_batting: List[pd.DataFrame] = []
    all_pitching: List[pd.DataFrame] = []

    for season in seasons:
        logging.info("Fetching schedule for %s", season)
        games = fetch_schedule_for_season(season)
        if not games.empty:
            games.to_csv(os.path.join(out_dir, f"games_{season}.csv"), index=False)
            all_games.append(games)
            logging.info("Saved %d games for %s", len(games), season)

        logging.info("Fetching team batting for %s", season)
        bat = fetch_team_batting(season)
        if not bat.empty:
            bat.to_csv(os.path.join(out_dir, f"team_batting_{season}.csv"), index=False)
            all_batting.append(bat)
            logging.info("Saved %d team batting rows for %s", len(bat), season)

        logging.info("Fetching team pitching for %s", season)
        pit = fetch_team_pitching(season)
        if not pit.empty:
            pit.to_csv(os.path.join(out_dir, f"team_pitching_{season}.csv"), index=False)
            all_pitching.append(pit)
            logging.info("Saved %d team pitching rows for %s", len(pit), season)

    if all_games:
        pd.concat(all_games, ignore_index=True).to_csv(os.path.join(out_dir, "games_2023_2025.csv"), index=False)
    if all_batting:
        pd.concat(all_batting, ignore_index=True).to_csv(os.path.join(out_dir, "team_batting_2023_2025.csv"), index=False)
    if all_pitching:
        pd.concat(all_pitching, ignore_index=True).to_csv(os.path.join(out_dir, "team_pitching_2023_2025.csv"), index=False)

    logging.info("Historical import complete. Output dir: %s", out_dir)


# ---------------------
# CLI
# ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, help="Season year (e.g., 2024)")
    p.add_argument("--season-range", default="2023-2025", help="Season range like 2023-2025")
    p.add_argument("--out-dir", default="./data/pybaseball", help="Output directory for CSV")
    return p.parse_args()


def main():
    args = parse_args()

    if args.season:
        seasons = [args.season]
    else:
        try:
            start_year, end_year = [int(x) for x in args.season_range.split("-")]
        except ValueError as exc:
            raise SystemExit("--season-range must look like 2023-2025") from exc
        seasons = list(range(start_year, end_year + 1))

    run(seasons, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
