#!/usr/bin/env python3
"""Daily pybaseball crawler for schedule + team batting/pitching stats.

Outputs (CSV):
- games_YYYY-MM-DD.csv (home/away schedule + scores)
- probable_starters_YYYY-MM-DD.csv (optional)
- team_batting_{season}.csv (team-level batting stats)
- team_pitching_{season}.csv (team-level pitching stats)

Usage:
  python pybaseball_daily_crawler.py --date 2026-03-13 --out-dir ./data/pybaseball/daily --include-probable-starters
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    from pybaseball import schedule_and_record, team_batting, team_pitching, team_ids
except Exception:  # pragma: no cover
    schedule_and_record = None
    team_batting = None
    team_pitching = None
    team_ids = None

try:
    from pybaseball import probable_starters
except Exception:  # pragma: no cover
    probable_starters = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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


def _cache_path(out_dir: str) -> str:
    return os.path.join(out_dir, "team_fetch_cache.csv")


def _load_team_cache(path: str) -> Dict[str, date]:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    team_col = None
    for cand in ["team_abbr", "team", "abbr", "abbrev"]:
        if cand in df.columns:
            team_col = cand
            break
    if team_col is None or "last_fetched" not in df.columns:
        return {}
    cache: Dict[str, date] = {}
    for _, row in df.iterrows():
        team = str(row.get(team_col, "")).strip()
        if not team:
            continue
        last_val = str(row.get("last_fetched", "")).strip()
        try:
            last_date = date.fromisoformat(last_val)
        except Exception:
            continue
        cache[team] = last_date
    return cache


def _save_team_cache(path: str, cache: Dict[str, date]) -> None:
    if not cache:
        return
    rows = [
        {"team_abbr": team, "last_fetched": last_date.isoformat()}
        for team, last_date in sorted(cache.items())
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


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

    # Home/Away
    ha_col = None
    for cand in ["Home/Away", "Home_Away", "home_away", "homeAway", "H/A"]:
        if cand in df.columns:
            ha_col = cand
            break
    if ha_col:
        df["is_home"] = df[ha_col].astype(str).str.upper().isin(["HOME", "H", "TRUE", "1"])
    else:
        if "Opp" in df.columns:
            df["is_home"] = ~df["Opp"].astype(str).str.startswith("@")
        else:
            df["is_home"] = pd.NA

    # Team/Opponent
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

    # Scores
    for cand in ["R", "Runs", "runs"]:
        if cand in df.columns:
            df["runs_scored"] = pd.to_numeric(df[cand], errors="coerce")
            break
    for cand in ["RA", "RunsAllowed", "runs_allowed", "R/RA"]:
        if cand in df.columns:
            df["runs_allowed"] = pd.to_numeric(df[cand], errors="coerce")
            break

    # Win/Loss
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


# ---------------------
# Fetchers
# ---------------------

def fetch_games_for_date(
    target_date: date,
    cache: Optional[Dict[str, date]] = None,
) -> Tuple[pd.DataFrame, Dict[str, date]]:
    if schedule_and_record is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")

    season = target_date.year
    team_abbrs = _load_team_abbrs()
    rows: List[pd.DataFrame] = []
    cache = cache or {}
    skipped = 0

    for abbr in team_abbrs:
        last_fetched = cache.get(abbr)
        if last_fetched and last_fetched >= target_date:
            skipped += 1
            continue
        try:
            df = schedule_and_record(season, abbr)
        except Exception as exc:
            logging.warning("schedule_and_record failed for %s %s: %s", season, abbr, exc)
            continue
        cache[abbr] = target_date
        if df is None or df.empty:
            continue
        df = _normalize_schedule_columns(df, season)
        df = df[df["game_date"] == target_date]
        if df.empty:
            continue
        df["team_abbr"] = abbr
        rows.append(df)

    if skipped:
        logging.info("Skipped %d teams due to cache", skipped)

    if not rows:
        return pd.DataFrame(), cache

    schedule = pd.concat(rows, ignore_index=True)
    home_games = schedule[schedule["is_home"] == True].copy()
    if home_games.empty:
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
    return games, cache


def fetch_team_batting(season: int) -> pd.DataFrame:
    if team_batting is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")
    try:
        df = team_batting(season)
    except Exception as exc:
        logging.warning("team_batting failed for %s: %s", season, exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["season"] = season
    return df


def fetch_team_pitching(season: int) -> pd.DataFrame:
    if team_pitching is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")
    try:
        df = team_pitching(season)
    except Exception as exc:
        logging.warning("team_pitching failed for %s: %s", season, exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["season"] = season
    return df


def fetch_probable_starters(target_date: date) -> pd.DataFrame:
    if probable_starters is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")

    year = target_date.year
    month = target_date.month
    try:
        df = probable_starters(year, month)
    except Exception as exc:
        logging.warning("probable_starters failed for %s-%s: %s", year, month, exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # normalize date column
    date_col = None
    for cand in ["game_date", "Game Date", "Date", "date"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col:
        df["game_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    else:
        df["game_date"] = pd.NaT

    if "game_date" in df.columns:
        df = df[df["game_date"] == target_date]

    # normalize team/opponent columns
    team_col = None
    for cand in ["team", "Team", "tm", "Tm"]:
        if cand in df.columns:
            team_col = cand
            break
    opp_col = None
    for cand in ["opponent", "Opponent", "Opp", "opp"]:
        if cand in df.columns:
            opp_col = cand
            break

    if team_col:
        df["team"] = df[team_col]
    if opp_col:
        df["opponent"] = df[opp_col]
        df["opponent"] = df["opponent"].astype(str).str.replace("@", "", regex=False)

    # derive home/away if possible
    if opp_col:
        df["is_home"] = ~df[opp_col].astype(str).str.startswith("@")
    else:
        df["is_home"] = pd.NA

    df = df.reset_index(drop=True)
    return df


# ---------------------
# Runner
# ---------------------

def run(
    target_date: date,
    out_dir: str,
    skip_team_stats: bool = False,
    include_probable_starters: bool = False,
) -> None:
    _safe_mkdir(out_dir)
    season = target_date.year

    cache_file = _cache_path(out_dir)
    cache = _load_team_cache(cache_file)

    logging.info("Fetching games for %s", target_date)
    games, cache = fetch_games_for_date(target_date, cache=cache)
    _save_team_cache(cache_file, cache)
    if not games.empty:
        games.to_csv(os.path.join(out_dir, f"games_{target_date.isoformat()}.csv"), index=False)
        logging.info("Saved %d games", len(games))
    else:
        logging.info("No games found for %s", target_date)

    if include_probable_starters:
        logging.info("Fetching probable starters for %s", target_date)
        starters = fetch_probable_starters(target_date)
        if not starters.empty:
            starters.to_csv(
                os.path.join(out_dir, f"probable_starters_{target_date.isoformat()}.csv"),
                index=False,
            )
            logging.info("Saved probable starters %d rows", len(starters))
        else:
            logging.info("No probable starters found for %s", target_date)

    if skip_team_stats:
        return

    logging.info("Fetching team batting for %s", season)
    bat = fetch_team_batting(season)
    if not bat.empty:
        bat.to_csv(os.path.join(out_dir, f"team_batting_{season}.csv"), index=False)
        logging.info("Saved team batting %d rows", len(bat))

    logging.info("Fetching team pitching for %s", season)
    pit = fetch_team_pitching(season)
    if not pit.empty:
        pit.to_csv(os.path.join(out_dir, f"team_pitching_{season}.csv"), index=False)
        logging.info("Saved team pitching %d rows", len(pit))


# ---------------------
# CLI
# ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--out-dir", default="./data/pybaseball/daily", help="Output directory for CSV")
    p.add_argument("--skip-team-stats", action="store_true", help="Skip team batting/pitching stats")
    p.add_argument("--include-probable-starters", action="store_true", help="Fetch probable starters for target date")
    return p.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else date.today()
    run(
        target_date,
        out_dir=args.out_dir,
        skip_team_stats=args.skip_team_stats,
        include_probable_starters=args.include_probable_starters,
    )


if __name__ == "__main__":
    main()
