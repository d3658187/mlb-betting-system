#!/usr/bin/env python3
"""Daily ETL skeleton for MLB betting system.

Usage:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  python etl_daily.py --date 2026-03-13

This is a scaffold. Plug in real data sources in fetch_* functions.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional

from sqlalchemy import create_engine, text

# Taiwan Sports Lottery odds (free apidata)
import taiwan_lottery_crawler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    return create_engine(db_url, pool_pre_ping=True)


# ---------------------
# Fetch stubs
# ---------------------

def fetch_teams(target_date: date) -> List[Dict]:
    """Return list of teams. Replace with real API call."""
    # TODO: integrate with MLB stats API or another provider
    return []


def fetch_games(target_date: date) -> List[Dict]:
    """Return list of games for target_date. Replace with real API call."""
    return []


def load_game_lookup(target_date: date, engine=None, conn=None) -> Dict:
    """Build (game_date, away_team_name, home_team_name) -> game_id lookup."""
    sql = text(
        """
        SELECT g.id as game_id,
               g.game_date,
               th.name as home_team_name,
               ta.name as away_team_name
          FROM games g
          JOIN teams th ON g.home_team_id = th.id
          JOIN teams ta ON g.away_team_id = ta.id
         WHERE g.game_date = :target_date
        """
    )
    close_conn = False
    if conn is None:
        engine = engine or get_engine()
        conn = engine.connect()
        close_conn = True
    try:
        rows = conn.execute(sql, {"target_date": target_date}).mappings().all()
    finally:
        if close_conn:
            conn.close()
    lookup = {}
    for r in rows:
        key = (r["game_date"].isoformat(), r["away_team_name"], r["home_team_name"])
        lookup[key] = r["game_id"]
    return lookup


def fetch_odds(target_date: date, engine=None, conn=None) -> List[Dict]:
    """Fetch MLB odds from Taiwan Sports Lottery apidata (free).

    Returns rows compatible with insert_odds().
    """
    if conn is None:
        engine = engine or get_engine()

    # Fetch from free apidata endpoint
    try:
        items = taiwan_lottery_crawler.fetch_pre_games(
            sport_id=taiwan_lottery_crawler.BASEBALL_SPORT_ID,
            lang="en",
        )
        games = taiwan_lottery_crawler.parse_pre_games(items, target_date)
    except Exception as exc:
        logging.warning("Failed to fetch Taiwan odds: %s", exc)
        return []

    if not games:
        logging.warning("No Taiwan odds parsed for %s", target_date)
        return []

    lookup = load_game_lookup(target_date, engine=engine, conn=conn)
    rows = taiwan_lottery_crawler.format_for_db(games, lookup)
    if not rows:
        logging.warning("No odds rows matched to games. Check team name mapping.")
        return []
    return rows


def fetch_results(target_date: date) -> List[Dict]:
    """Return list of completed game results. Replace with real API call."""
    return []


# ---------------------
# Loaders (upserts)
# ---------------------

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


def insert_odds(conn, rows: Iterable[Dict]):
    if not rows:
        return
    sql = text(
        """
        INSERT INTO odds (
          game_id, sportsbook, market, selection, price, line, retrieved_at
        )
        VALUES (
          :game_id, :sportsbook, :market, :selection, :price, :line, :retrieved_at
        )
        ON CONFLICT DO NOTHING;
        """
    )
    conn.execute(sql, list(rows))


def upsert_results(conn, rows: Iterable[Dict]):
    if not rows:
        return
    sql = text(
        """
        INSERT INTO game_results (
          game_id, home_score, away_score, home_win, total_points
        )
        VALUES (
          :game_id, :home_score, :away_score, :home_win, :total_points
        )
        ON CONFLICT (game_id) DO UPDATE
          SET home_score = EXCLUDED.home_score,
              away_score = EXCLUDED.away_score,
              home_win = EXCLUDED.home_win,
              total_points = EXCLUDED.total_points,
              updated_at = now();
        """
    )
    conn.execute(sql, list(rows))


# ---------------------
# ETL runner
# ---------------------

def run_etl(target_date: date):
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO etl_runs (run_date, status, started_at)
                VALUES (:run_date, 'running', now())
                """
            ),
            {"run_date": target_date},
        )

        try:
            teams = fetch_teams(target_date)
            games = fetch_games(target_date)

            # Upsert teams/games first so odds can map to game_id
            upsert_teams(conn, teams)
            upsert_games(conn, games)

            odds = fetch_odds(target_date, engine=engine, conn=conn)
            results = fetch_results(target_date)

            insert_odds(conn, odds)
            upsert_results(conn, results)

            conn.execute(
                text(
                    """
                    UPDATE etl_runs
                       SET status = 'success', finished_at = now()
                     WHERE run_date = :run_date AND status = 'running'
                    """
                ),
                {"run_date": target_date},
            )
            logging.info("ETL success for %s", target_date)
        except Exception as e:
            conn.execute(
                text(
                    """
                    UPDATE etl_runs
                       SET status = 'failed', finished_at = now(), message = :message
                     WHERE run_date = :run_date AND status = 'running'
                    """
                ),
                {"run_date": target_date, "message": str(e)},
            )
            logging.exception("ETL failed for %s", target_date)
            raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    return p.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else date.today()
    run_etl(target_date)


if __name__ == "__main__":
    main()
