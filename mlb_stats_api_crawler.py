#!/usr/bin/env python3
"""MLB Stats API crawler (no key required) for daily schedule/results and teams.

Outputs (CSV):
- games_YYYY-MM-DD.csv (home/away schedule + scores)
- teams_mlb.csv (team_id/name/abbreviation/league/division)

Usage:
  python mlb_stats_api_crawler.py --date 2026-03-13 --out-dir ./data/mlb_stats_api/daily
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def request_json(url: str, params: Optional[Dict] = None) -> Dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------
# Teams
# ---------------------

def fetch_teams() -> List[Dict]:
    data = request_json(f"{BASE_URL}/teams", params={"sportId": 1})
    teams = []
    for t in data.get("teams", []):
        teams.append({
            "team_id": t.get("id"),
            "name": t.get("name"),
            "abbreviation": t.get("abbreviation"),
            "team_name": t.get("teamName"),
            "location_name": t.get("locationName"),
            "league": (t.get("league") or {}).get("name"),
            "division": (t.get("division") or {}).get("name"),
        })
    return teams


def build_team_abbr_map(teams: List[Dict]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for t in teams:
        tid = t.get("team_id")
        if tid is None:
            continue
        abbr = t.get("abbreviation") or t.get("team_name") or t.get("name")
        if abbr:
            mapping[tid] = abbr
    return mapping


# ---------------------
# Schedule/Games
# ---------------------

def fetch_schedule(target_date: date) -> Dict:
    params = {
        "sportId": 1,
        "date": target_date.isoformat(),
    }
    return request_json(f"{BASE_URL}/schedule", params=params)


def parse_games(schedule_json: Dict, team_abbr: Dict[int, str], season: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for date_block in schedule_json.get("dates", []):
        game_date = date_block.get("date")
        for g in date_block.get("games", []):
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_team_id = (home.get("team") or {}).get("id")
            away_team_id = (away.get("team") or {}).get("id")

            home_team = team_abbr.get(home_team_id) or (home.get("team") or {}).get("name")
            away_team = team_abbr.get(away_team_id) or (away.get("team") or {}).get("name")

            home_score = home.get("score")
            away_score = away.get("score")

            home_win = None
            if home_score is not None and away_score is not None:
                try:
                    home_win = float(home_score) > float(away_score)
                except Exception:
                    home_win = None

            rows.append({
                "game_date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_win,
                "season": season,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["home_team", "away_team", "game_date"])
    df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")
    return df


# ---------------------
# Runner
# ---------------------

def run(target_date: date, out_dir: str, skip_teams: bool = False) -> None:
    _safe_mkdir(out_dir)
    season = target_date.year

    logging.info("Fetching teams")
    teams = fetch_teams()
    if teams and not skip_teams:
        teams_df = pd.DataFrame(teams)
        teams_df.to_csv(os.path.join(out_dir, "teams_mlb.csv"), index=False)
        logging.info("Saved %d teams", len(teams_df))

    team_abbr = build_team_abbr_map(teams)

    logging.info("Fetching games for %s", target_date)
    schedule = fetch_schedule(target_date)
    games = parse_games(schedule, team_abbr, season)

    if not games.empty:
        games.to_csv(os.path.join(out_dir, f"games_{target_date.isoformat()}.csv"), index=False)
        logging.info("Saved %d games", len(games))
    else:
        logging.info("No games found for %s", target_date)


# ---------------------
# CLI
# ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--out-dir", default="./data/mlb_stats_api/daily", help="Output directory for CSV")
    p.add_argument("--skip-teams", action="store_true", help="Skip teams CSV output")
    return p.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else date.today()
    run(target_date, out_dir=args.out_dir, skip_teams=args.skip_teams)


if __name__ == "__main__":
    main()
