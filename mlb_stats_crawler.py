#!/usr/bin/env python3
"""MLB Stats API crawler (free) for daily schedule, probable pitchers, and lineup batting stats.

This is the A-layer (primary) data source in the A/B hybrid pipeline.

Usage:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  python mlb_stats_crawler.py --date 2026-03-13

Notes:
- Uses the official MLB Stats API (no key required): https://statsapi.mlb.com/api/
- This is a framework; adjust column mappings to match your actual schema.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from sqlalchemy import create_engine, text

BASE_URL = "https://statsapi.mlb.com/api/v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------
# DB
# ---------------------

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    return create_engine(db_url, pool_pre_ping=True)


def upsert_teams(conn, rows: Iterable[Dict]):
    if not rows:
        return
    sql = text(
        """
        INSERT INTO teams (mlb_team_id, name, abbreviation)
        VALUES (:mlb_team_id, :name, :abbreviation)
        ON CONFLICT (mlb_team_id) DO UPDATE
          SET name = EXCLUDED.name,
              abbreviation = EXCLUDED.abbreviation,
              updated_at = now();
        """
    )
    conn.execute(sql, list(rows))


def upsert_games(conn, rows: Iterable[Dict]):
    if not rows:
        return
    sql = text(
        """
        INSERT INTO games (
          mlb_game_id, game_date, game_datetime,
          home_team_id, away_team_id, venue, status
        )
        VALUES (
          :mlb_game_id, :game_date, :game_datetime,
          :home_team_id, :away_team_id, :venue, :status
        )
        ON CONFLICT (mlb_game_id) DO UPDATE
          SET game_date = EXCLUDED.game_date,
              game_datetime = EXCLUDED.game_datetime,
              home_team_id = EXCLUDED.home_team_id,
              away_team_id = EXCLUDED.away_team_id,
              venue = EXCLUDED.venue,
              status = EXCLUDED.status,
              updated_at = now();
        """
    )
    conn.execute(sql, list(rows))


def upsert_stats_pitching(conn, rows: Iterable[Dict]):
    if not rows:
        return
    # NOTE: Adjust columns to your actual stats_pitching schema
    sql = text(
        """
        INSERT INTO stats_pitching (
          game_id, pitcher_mlb_id, team_mlb_id, is_home,
          innings_pitched, hits, runs, earned_runs,
          walks, strikeouts, era, whip, pitches, strikes
        )
        VALUES (
          :game_id, :pitcher_mlb_id, :team_mlb_id, :is_home,
          :innings_pitched, :hits, :runs, :earned_runs,
          :walks, :strikeouts, :era, :whip, :pitches, :strikes
        )
        ON CONFLICT (game_id, pitcher_mlb_id) DO UPDATE
          SET innings_pitched = EXCLUDED.innings_pitched,
              hits = EXCLUDED.hits,
              runs = EXCLUDED.runs,
              earned_runs = EXCLUDED.earned_runs,
              walks = EXCLUDED.walks,
              strikeouts = EXCLUDED.strikeouts,
              era = EXCLUDED.era,
              whip = EXCLUDED.whip,
              pitches = EXCLUDED.pitches,
              strikes = EXCLUDED.strikes;
        """
    )
    conn.execute(sql, list(rows))


def upsert_stats_batting(conn, rows: Iterable[Dict]):
    if not rows:
        return
    # NOTE: Adjust columns to your actual stats_batting schema
    sql = text(
        """
        INSERT INTO stats_batting (
          game_id, batter_mlb_id, team_mlb_id, is_home, batting_order,
          position, at_bats, hits, runs, rbi, walks, strikeouts,
          avg, obp, slg, ops,
          season_ab, season_hits, season_hr, season_bb, season_so,
          season_avg, season_obp, season_slg, season_ops
        )
        VALUES (
          :game_id, :batter_mlb_id, :team_mlb_id, :is_home, :batting_order,
          :position, :at_bats, :hits, :runs, :rbi, :walks, :strikeouts,
          :avg, :obp, :slg, :ops,
          :season_ab, :season_hits, :season_hr, :season_bb, :season_so,
          :season_avg, :season_obp, :season_slg, :season_ops
        )
        ON CONFLICT (game_id, batter_mlb_id) DO UPDATE
          SET batting_order = EXCLUDED.batting_order,
              position = EXCLUDED.position,
              at_bats = EXCLUDED.at_bats,
              hits = EXCLUDED.hits,
              runs = EXCLUDED.runs,
              rbi = EXCLUDED.rbi,
              walks = EXCLUDED.walks,
              strikeouts = EXCLUDED.strikeouts,
              avg = EXCLUDED.avg,
              obp = EXCLUDED.obp,
              slg = EXCLUDED.slg,
              ops = EXCLUDED.ops,
              season_ab = EXCLUDED.season_ab,
              season_hits = EXCLUDED.season_hits,
              season_hr = EXCLUDED.season_hr,
              season_bb = EXCLUDED.season_bb,
              season_so = EXCLUDED.season_so,
              season_avg = EXCLUDED.season_avg,
              season_obp = EXCLUDED.season_obp,
              season_slg = EXCLUDED.season_slg,
              season_ops = EXCLUDED.season_ops;
        """
    )
    conn.execute(sql, list(rows))


# ---------------------
# API helpers
# ---------------------

def request_json(url: str, params: Optional[Dict] = None) -> Dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_schedule(target_date: date) -> Dict:
    params = {
        "sportId": 1,  # MLB
        "date": target_date.isoformat(),
        "hydrate": "probablePitcher",  # include probable pitcher info
    }
    return request_json(f"{BASE_URL}/schedule", params)


def fetch_game_boxscore(game_pk: int) -> Dict:
    # boxscore endpoint includes player seasonStats and battingOrder
    return request_json(f"{BASE_URL}/game/{game_pk}/boxscore")


def fetch_game_feed(game_pk: int) -> Dict:
    # feed/live has additional context if needed
    return request_json(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")


# ---------------------
# Parsers
# ---------------------

def parse_schedule(schedule_json: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Return (teams, games, probable_pitchers)."""
    teams: List[Dict] = []
    games: List[Dict] = []
    pitchers: List[Dict] = []

    def normalize_team(team: Dict) -> Optional[Dict]:
        team_id = team.get("id")
        if team_id is None:
            logging.warning("Skipping team with missing id: %s", team)
            return None
        name = team.get("name") or "UNKNOWN"
        abbr = team.get("abbreviation") or "N/A"
        return {
            "mlb_team_id": team_id,
            "name": name,
            "abbreviation": abbr,
        }

    for date_block in schedule_json.get("dates", []):
        for g in date_block.get("games", []):
            game_pk = g.get("gamePk")
            game_date = date_block.get("date")
            game_datetime = g.get("gameDate")
            status = g.get("status", {}).get("detailedState")
            venue = g.get("venue", {}).get("name")

            home = g.get("teams", {}).get("home", {}).get("team", {})
            away = g.get("teams", {}).get("away", {}).get("team", {})

            home_team = normalize_team(home)
            away_team = normalize_team(away)
            if home_team:
                teams.append(home_team)
            if away_team:
                teams.append(away_team)

            games.append({
                "mlb_game_id": game_pk,
                "game_date": game_date,
                "game_datetime": game_datetime,
                "home_team_id": home.get("id"),  # temp: raw MLB id; remap later
                "away_team_id": away.get("id"),
                "venue": venue,
                "status": status,
            })

            for side in ["home", "away"]:
                p = g.get("teams", {}).get(side, {}).get("probablePitcher")
                if not p:
                    continue
                pitchers.append({
                    "mlb_game_id": game_pk,
                    "pitcher_mlb_id": p.get("id"),
                    "pitcher_name": p.get("fullName"),
                    "team_mlb_id": home.get("id") if side == "home" else away.get("id"),
                    "is_home": side == "home",
                })

    # de-dup teams by mlb_team_id
    uniq = {}
    for t in teams:
        if t.get("mlb_team_id") is None:
            continue
        uniq[t["mlb_team_id"]] = t
    teams = list(uniq.values())
    return teams, games, pitchers


def parse_boxscore_for_pitchers(boxscore: Dict, game_id: int) -> List[Dict]:
    """Extract pitching stats for each pitcher (if available)."""
    rows: List[Dict] = []
    for side in ["home", "away"]:
        team = boxscore.get("teams", {}).get(side, {})
        team_id = team.get("team", {}).get("id")
        pitchers = team.get("pitchers", [])
        players = team.get("players", {})
        for pid in pitchers:
            p = players.get(f"ID{pid}", {})
            stats = p.get("stats", {}).get("pitching", {})
            rows.append({
                "game_id": game_id,
                "pitcher_mlb_id": pid,
                "team_mlb_id": team_id,
                "is_home": side == "home",
                "innings_pitched": stats.get("inningsPitched"),
                "hits": stats.get("hits"),
                "runs": stats.get("runs"),
                "earned_runs": stats.get("earnedRuns"),
                "walks": stats.get("baseOnBalls"),
                "strikeouts": stats.get("strikeOuts"),
                "era": stats.get("era"),
                "whip": stats.get("whip"),
                "pitches": stats.get("pitchesThrown"),
                "strikes": stats.get("strikes"),
            })
    return rows


def parse_boxscore_for_lineup(boxscore: Dict, game_id: int) -> List[Dict]:
    """Extract batting stats for each player in the boxscore."""
    rows: List[Dict] = []
    for side in ["home", "away"]:
        team = boxscore.get("teams", {}).get(side, {})
        team_id = team.get("team", {}).get("id")
        batting_order = team.get("battingOrder", [])
        players = team.get("players", {})

        # batting_order is list of player IDs in order
        order_map = {pid: idx + 1 for idx, pid in enumerate(batting_order)}

        for pid, pdata in players.items():
            # pid format: "ID123456"
            player_id = int(pid.replace("ID", ""))
            batting = pdata.get("stats", {}).get("batting", {})
            season = pdata.get("seasonStats", {}).get("batting", {})
            position = pdata.get("position", {}).get("abbreviation")

            rows.append({
                "game_id": game_id,
                "batter_mlb_id": player_id,
                "team_mlb_id": team_id,
                "is_home": side == "home",
                "batting_order": order_map.get(player_id),
                "position": position,
                "at_bats": batting.get("atBats"),
                "hits": batting.get("hits"),
                "runs": batting.get("runs"),
                "rbi": batting.get("rbi"),
                "walks": batting.get("baseOnBalls"),
                "strikeouts": batting.get("strikeOuts"),
                "avg": batting.get("avg"),
                "obp": batting.get("obp"),
                "slg": batting.get("slg"),
                "ops": batting.get("ops"),
                "season_ab": season.get("atBats"),
                "season_hits": season.get("hits"),
                "season_hr": season.get("homeRuns"),
                "season_bb": season.get("baseOnBalls"),
                "season_so": season.get("strikeOuts"),
                "season_avg": season.get("avg"),
                "season_obp": season.get("obp"),
                "season_slg": season.get("slg"),
                "season_ops": season.get("ops"),
            })
    return rows


# ---------------------
# ETL flow
# ---------------------

def run(target_date: date):
    schedule = fetch_schedule(target_date)
    teams, games, probable_pitchers = parse_schedule(schedule)

    engine = get_engine()
    with engine.begin() as conn:
        # teams
        upsert_teams(conn, teams)

        # map MLB team ids -> local UUIDs (for games table)
        team_id_map = {
            row["mlb_team_id"]: row["id"]
            for row in conn.execute(text("SELECT id, mlb_team_id FROM teams")).mappings()
        }

        # transform games team IDs to local UUID
        for g in games:
            g["home_team_id"] = team_id_map.get(g["home_team_id"])
            g["away_team_id"] = team_id_map.get(g["away_team_id"])

        upsert_games(conn, games)

        # map mlb_game_id -> local game UUID
        game_id_map = {
            row["mlb_game_id"]: row["id"]
            for row in conn.execute(text("SELECT id, mlb_game_id FROM games")).mappings()
        }

        # probable pitchers (schedule-level)
        # Option: store into stats_pitching as placeholders if needed
        # Here we load actual pitching stats from boxscore instead.

        # fetch boxscore per game and load pitching + batting
        pitching_rows: List[Dict] = []
        batting_rows: List[Dict] = []
        for g in games:
            mlb_game_id = g["mlb_game_id"]
            game_uuid = game_id_map.get(mlb_game_id)
            if not game_uuid:
                continue
            try:
                boxscore = fetch_game_boxscore(mlb_game_id)
            except Exception as e:
                logging.warning("boxscore fetch failed for %s: %s", mlb_game_id, e)
                continue

            pitching_rows.extend(parse_boxscore_for_pitchers(boxscore, game_uuid))
            batting_rows.extend(parse_boxscore_for_lineup(boxscore, game_uuid))

        upsert_stats_pitching(conn, pitching_rows)
        upsert_stats_batting(conn, batting_rows)

    logging.info("MLB Stats API crawl complete for %s", target_date)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    return p.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else date.today()
    run(target_date)


if __name__ == "__main__":
    main()
