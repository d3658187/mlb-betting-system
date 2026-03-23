#!/usr/bin/env python3
"""Import historical starting pitchers + pitcher season stats via pybaseball.

Outputs (CSV):
- pitcher_stats_{season}.csv (season-level pitching stats with MLBAM id)
- starting_pitchers_{season}.csv (game-level probable starters from Retrosheet)
- merged combined CSVs for 2023-2025

Usage:
  python import_pitcher_data.py --season-range 2023-2025 --out-dir ./data/pybaseball
"""
from __future__ import annotations

import argparse
import logging
import os
import io
import zipfile
from io import StringIO
from typing import Iterable, List, Dict

import pandas as pd
import requests

try:
    from pybaseball import pitching_stats, playerid_reverse_lookup, team_ids
    from pybaseball import retrosheet
    from pybaseball.utils import get_text_file
except Exception:  # pragma: no cover
    pitching_stats = None
    playerid_reverse_lookup = None
    team_ids = None
    retrosheet = None
    get_text_file = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------
# Helpers
# ---------------------

def _safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _reverse_lookup(ids: List[str], key_type: str) -> pd.DataFrame:
    if playerid_reverse_lookup is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")
    frames = []
    for chunk in _chunked(ids, 1000):
        frames.append(playerid_reverse_lookup(chunk, key_type=key_type))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _retro_team_map() -> Dict[str, str]:
    if team_ids is None:
        return {}
    try:
        df = team_ids()
        if df.empty:
            return {}
        last_year = df["yearID"].max()
        df = df[df["yearID"] == last_year].dropna(subset=["teamIDretro", "teamIDBR"])
        return dict(zip(df["teamIDretro"], df["teamIDBR"]))
    except Exception:
        return {}


# ---------------------
# Pitcher season stats
# ---------------------

def fetch_pitcher_stats(season: int) -> pd.DataFrame:
    if pitching_stats is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")
    logging.info("Fetching pitching_stats for %s", season)
    try:
        df = pitching_stats(season, season, qual=0)
    except TypeError:
        df = pitching_stats(season, qual=0)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["Season"] = season

    fg_ids = df["IDfg"].dropna().astype(int).unique().tolist()
    fg_map = _reverse_lookup(fg_ids, key_type="fangraphs")
    if fg_map.empty:
        return pd.DataFrame()

    fg_map = fg_map.copy()
    fg_map["key_fangraphs"] = pd.to_numeric(fg_map["key_fangraphs"], errors="coerce").astype("Int64")
    fg_to_mlbam = fg_map.set_index("key_fangraphs")["key_mlbam"].to_dict()
    df["mlbam_id"] = df["IDfg"].astype(int).map(fg_to_mlbam)

    keep_cols = [
        "mlbam_id",
        "IDfg",
        "Season",
        "Name",
        "Team",
        "ERA",
        "FIP",
        "xFIP",
        "SIERA",
        "WHIP",
        "K%",
        "BB%",
        "K-BB%",
        "WAR",
        "IP",
    ]
    existing = [c for c in keep_cols if c in df.columns]
    out = df[existing].copy()
    out = out.dropna(subset=["mlbam_id"]).copy()
    out["mlbam_id"] = pd.to_numeric(out["mlbam_id"], errors="coerce")
    return out


# ---------------------
# Starting pitchers (Retrosheet)
# ---------------------

def _get_retrosheet_gamelog_text(season: int) -> str:
    if retrosheet is None or get_text_file is None:
        return ""
    url = retrosheet.gamelog_url.format(season)
    try:
        text = get_text_file(url)
    except Exception:
        text = ""

    if text and not text.startswith("404"):
        return text

    zip_url = f"https://www.retrosheet.org/gamelogs/gl{season}.zip"
    try:
        resp = requests.get(zip_url, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            target = f"gl{season}.txt"
            name = next((n for n in zf.namelist() if n.lower() == target), None)
            if not name:
                name = next((n for n in zf.namelist() if n.lower().endswith(target)), None)
            if not name:
                name = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
            if not name:
                raise RuntimeError("Retrosheet zip missing gamelog txt")
            with zf.open(name) as f:
                return f.read().decode("latin-1")
    except Exception as exc:
        logging.warning("Retrosheet fallback failed for %s: %s", season, exc)
        return text or ""


def fetch_starting_pitchers(season: int) -> pd.DataFrame:
    if retrosheet is None or get_text_file is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")

    logging.info("Fetching Retrosheet gamelog for %s", season)
    text = _get_retrosheet_gamelog_text(season)
    if not text or text.startswith("404"):
        logging.warning("Retrosheet gamelog unavailable for %s", season)
        return pd.DataFrame()
    df = pd.read_csv(StringIO(text), header=None, names=retrosheet.gamelog_columns)
    if df.empty:
        return pd.DataFrame()

    retro_map = _retro_team_map()

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
    df["home_team"] = df["home_team"].map(retro_map).fillna(df["home_team"])
    df["away_team"] = df["visiting_team"].map(retro_map).fillna(df["visiting_team"])

    df["home_pitcher_retro"] = df["home_starting_pitcher_id"].astype(str)
    df["away_pitcher_retro"] = df["visiting_starting_pitcher_id"].astype(str)

    retro_ids = pd.concat([df["home_pitcher_retro"], df["away_pitcher_retro"]]).dropna().unique().tolist()
    retro_ids = [rid for rid in retro_ids if rid and rid != "nan"]

    retro_map_df = _reverse_lookup(retro_ids, key_type="retro")
    retro_to_mlbam = retro_map_df.set_index("key_retro")["key_mlbam"].to_dict() if not retro_map_df.empty else {}

    df["home_pitcher_mlbam"] = df["home_pitcher_retro"].map(retro_to_mlbam)
    df["away_pitcher_mlbam"] = df["away_pitcher_retro"].map(retro_to_mlbam)

    out = df[[
        "game_date",
        "home_team",
        "away_team",
        "home_pitcher_mlbam",
        "away_pitcher_mlbam",
        "home_pitcher_retro",
        "away_pitcher_retro",
        "home_starting_pitcher_name",
        "visiting_starting_pitcher_name",
    ]].copy()
    out = out.rename(columns={
        "home_starting_pitcher_name": "home_pitcher_name",
        "visiting_starting_pitcher_name": "away_pitcher_name",
    })
    out["season"] = season
    out["home_pitcher_mlbam"] = pd.to_numeric(out["home_pitcher_mlbam"], errors="coerce")
    out["away_pitcher_mlbam"] = pd.to_numeric(out["away_pitcher_mlbam"], errors="coerce")
    return out


# ---------------------
# Runner
# ---------------------

def run(seasons: List[int], out_dir: str) -> None:
    _safe_mkdir(out_dir)

    all_pitcher_stats = []
    all_starters = []

    for season in seasons:
        stats = fetch_pitcher_stats(season)
        if not stats.empty:
            stats.to_csv(os.path.join(out_dir, f"pitcher_stats_{season}.csv"), index=False)
            all_pitcher_stats.append(stats)
            logging.info("Saved pitcher stats %s rows", len(stats))

        starters = fetch_starting_pitchers(season)
        if not starters.empty:
            starters.to_csv(os.path.join(out_dir, f"starting_pitchers_{season}.csv"), index=False)
            all_starters.append(starters)
            logging.info("Saved starting pitchers %s rows", len(starters))

    if all_pitcher_stats:
        pd.concat(all_pitcher_stats, ignore_index=True).to_csv(
            os.path.join(out_dir, "pitcher_stats_2023_2025.csv"), index=False
        )
    if all_starters:
        pd.concat(all_starters, ignore_index=True).to_csv(
            os.path.join(out_dir, "starting_pitchers_2023_2025.csv"), index=False
        )

    logging.info("Pitcher import complete. Output dir: %s", out_dir)


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
        start_year, end_year = [int(x) for x in args.season_range.split("-")]
        seasons = list(range(start_year, end_year + 1))

    run(seasons, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
