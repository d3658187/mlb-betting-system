#!/usr/bin/env python3
"""Bullpen fatigue scoring based on recent relief usage.

Usage:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  python bullpen_fatigue.py --date 2026-03-13 --out ./data/bullpen_fatigue_2026-03-13.csv

Notes:
- Uses stats_pitching joined to games to get game_date.
- Excludes starting pitchers via starting_pitchers table if present.
- Computes per-pitcher fatigue over the last N days, then sums to team index.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass
class FatigueWeights:
    appearance_weight: float = 1.0
    pitch_weight: float = 1.0 / 30.0  # every 30 pitches adds ~1 fatigue
    rest_weight: float = 1.5
    rest_days_cap: int = 3


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    return create_engine(db_url, pool_pre_ping=True)


def table_exists(conn, table: str) -> bool:
    sql = text(
        """
        SELECT EXISTS (
          SELECT 1 FROM information_schema.tables
          WHERE table_schema = 'public' AND table_name = :table
        )
        """
    )
    return bool(conn.execute(sql, {"table": table}).scalar())


def ensure_bullpen_table(conn):
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS bullpen_fatigue (
              id BIGSERIAL PRIMARY KEY,
              game_date DATE NOT NULL,
              team_mlb_id INTEGER NOT NULL,
              bullpen_fatigue_index NUMERIC(10,3),
              bullpen_pitch_count NUMERIC(10,3),
              bullpen_appearance_days NUMERIC(10,3),
              bullpen_pitcher_count INTEGER,
              bullpen_avg_rest_days NUMERIC(10,3),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              UNIQUE (game_date, team_mlb_id)
            );
            """
        )
    )


def upsert_bullpen_fatigue(conn, rows: List[Dict]):
    if not rows:
        return
    sql = text(
        """
        INSERT INTO bullpen_fatigue (
          game_date,
          team_mlb_id,
          bullpen_fatigue_index,
          bullpen_pitch_count,
          bullpen_appearance_days,
          bullpen_pitcher_count,
          bullpen_avg_rest_days
        ) VALUES (
          :game_date,
          :team_mlb_id,
          :bullpen_fatigue_index,
          :bullpen_pitch_count,
          :bullpen_appearance_days,
          :bullpen_pitcher_count,
          :bullpen_avg_rest_days
        )
        ON CONFLICT (game_date, team_mlb_id) DO UPDATE
          SET bullpen_fatigue_index = EXCLUDED.bullpen_fatigue_index,
              bullpen_pitch_count = EXCLUDED.bullpen_pitch_count,
              bullpen_appearance_days = EXCLUDED.bullpen_appearance_days,
              bullpen_pitcher_count = EXCLUDED.bullpen_pitcher_count,
              bullpen_avg_rest_days = EXCLUDED.bullpen_avg_rest_days,
              updated_at = now();
        """
    )
    conn.execute(sql, rows)


def load_pitching(engine, since_date: date, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT g.game_date,
               p.game_id,
               p.pitcher_mlb_id,
               p.team_mlb_id,
               p.pitches
          FROM stats_pitching p
          JOIN games g ON g.id = p.game_id
         WHERE g.game_date >= :since_date
           AND g.game_date < :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": since_date, "target_date": target_date})


def load_starting(engine, since_date: date, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT g.game_date,
               s.game_id,
               s.pitcher_mlb_id
          FROM starting_pitchers s
          JOIN games g ON g.mlb_game_id = s.game_id
         WHERE g.game_date >= :since_date
           AND g.game_date < :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": since_date, "target_date": target_date})


def compute_pitcher_fatigue(
    pitching: pd.DataFrame,
    target_date: date,
    weights: FatigueWeights,
) -> pd.DataFrame:
    if pitching.empty:
        return pitching

    pitching = pitching.copy()
    pitching["game_date"] = pd.to_datetime(pitching["game_date"]).dt.date
    pitching["pitches"] = pd.to_numeric(pitching["pitches"], errors="coerce").fillna(0)

    groups = pitching.groupby("pitcher_mlb_id", dropna=False)
    appearance_days = groups["game_date"].nunique().rename("appearance_days")
    total_pitches = groups["pitches"].sum().rename("total_pitches")
    last_game_date = groups["game_date"].max().rename("last_game_date")

    # most recent team in window
    pitching_sorted = pitching.sort_values(["pitcher_mlb_id", "game_date"]).dropna(subset=["team_mlb_id"])
    latest_team = pitching_sorted.groupby("pitcher_mlb_id").tail(1).set_index("pitcher_mlb_id")["team_mlb_id"]

    fatigue_df = pd.concat([appearance_days, total_pitches, last_game_date], axis=1).reset_index()
    fatigue_df["team_mlb_id"] = fatigue_df["pitcher_mlb_id"].map(latest_team)

    fatigue_df["rest_days"] = fatigue_df["last_game_date"].apply(
        lambda d: max(0, (target_date - d).days - 1) if pd.notna(d) else weights.rest_days_cap
    )

    rest_penalty = (weights.rest_days_cap - fatigue_df["rest_days"]).clip(lower=0) * weights.rest_weight

    fatigue_df["fatigue_score"] = (
        fatigue_df["appearance_days"] * weights.appearance_weight
        + fatigue_df["total_pitches"] * weights.pitch_weight
        + rest_penalty
    )
    return fatigue_df


def load_game_dates(engine, start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[date]:
    sql = "SELECT DISTINCT game_date FROM games"
    params: Dict[str, date] = {}
    if start_date and end_date:
        sql += " WHERE game_date BETWEEN :start_date AND :end_date"
        params = {"start_date": start_date, "end_date": end_date}
    sql += " ORDER BY game_date"
    df = pd.read_sql(text(sql), engine, params=params)
    if df.empty:
        return []
    return pd.to_datetime(df["game_date"]).dt.date.tolist()


def compute_team_fatigue(
    engine,
    target_date: date,
    window_days: int = 5,
    weights: Optional[FatigueWeights] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if weights is None:
        weights = FatigueWeights()

    since_date = target_date - timedelta(days=window_days)
    pitching = load_pitching(engine, since_date, target_date)
    if pitching.empty:
        return pd.DataFrame(), pd.DataFrame()

    with engine.begin() as conn:
        has_starting = table_exists(conn, "starting_pitchers")

    if has_starting:
        starters = load_starting(engine, since_date, target_date)
        if not starters.empty:
            starters = starters[["game_id", "pitcher_mlb_id"]].drop_duplicates()
            pitching = pitching.merge(starters, on=["game_id", "pitcher_mlb_id"], how="left", indicator=True)
            pitching = pitching[pitching["_merge"] == "left_only"].drop(columns=["_merge"])

    pitcher_fatigue = compute_pitcher_fatigue(pitching, target_date, weights)
    if pitcher_fatigue.empty:
        return pd.DataFrame(), pd.DataFrame()

    team_fatigue = (
        pitcher_fatigue.groupby("team_mlb_id", dropna=False)
        .agg(
            bullpen_fatigue_index=("fatigue_score", "sum"),
            bullpen_pitch_count=("total_pitches", "sum"),
            bullpen_appearance_days=("appearance_days", "sum"),
            bullpen_pitcher_count=("pitcher_mlb_id", "nunique"),
            bullpen_avg_rest_days=("rest_days", "mean"),
        )
        .reset_index()
    )
    team_fatigue["game_date"] = target_date

    return team_fatigue, pitcher_fatigue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--season", type=int, help="Season year (e.g., 2024)")
    parser.add_argument("--season-range", help="Season range like 2022-2024")
    parser.add_argument("--all", action="store_true", help="Process all game dates in DB")
    parser.add_argument("--window", type=int, default=5, help="Lookback days")
    parser.add_argument("--out", type=str, help="Output CSV for team fatigue index (single date only)")
    parser.add_argument("--out-pitchers", type=str, help="Output CSV for pitcher fatigue details (single date only)")
    parser.add_argument("--store-db", action="store_true", help="Write bullpen fatigue to DB")
    args = parser.parse_args()

    engine = get_engine()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        team_fatigue, pitcher_fatigue = compute_team_fatigue(engine, target_date, window_days=args.window)
        if team_fatigue.empty:
            logging.warning("No bullpen fatigue data for %s", target_date)
            return

        if args.out:
            out_dir = os.path.dirname(args.out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            team_fatigue.to_csv(args.out, index=False)
            logging.info("Wrote team fatigue -> %s", args.out)

        if args.out_pitchers:
            out_dir = os.path.dirname(args.out_pitchers)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            pitcher_fatigue.to_csv(args.out_pitchers, index=False)
            logging.info("Wrote pitcher fatigue -> %s", args.out_pitchers)

        if args.store_db:
            with engine.begin() as conn:
                ensure_bullpen_table(conn)
                upsert_bullpen_fatigue(conn, team_fatigue.to_dict("records"))
                logging.info("Stored bullpen fatigue for %s", target_date)
        logging.info("Done.")
        return

    # range processing
    date_list: List[date] = []
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        date_list = load_game_dates(engine, start_date, end_date)
    elif args.season:
        start_date = date(args.season, 1, 1)
        end_date = date(args.season, 12, 31)
        date_list = load_game_dates(engine, start_date, end_date)
    elif args.season_range:
        try:
            start_year, end_year = [int(x) for x in args.season_range.split("-")]
        except ValueError as exc:
            raise SystemExit("--season-range must look like 2022-2024") from exc
        for year in range(start_year, end_year + 1):
            date_list.extend(load_game_dates(engine, date(year, 1, 1), date(year, 12, 31)))
    elif args.all:
        date_list = load_game_dates(engine)

    if not date_list:
        raise SystemExit("Provide --date or a range (--start-date/--end-date, --season, --season-range, --all)")

    if args.store_db:
        with engine.begin() as conn:
            ensure_bullpen_table(conn)

    for target_date in date_list:
        team_fatigue, _ = compute_team_fatigue(engine, target_date, window_days=args.window)
        if team_fatigue.empty:
            continue
        if args.store_db:
            with engine.begin() as conn:
                upsert_bullpen_fatigue(conn, team_fatigue.to_dict("records"))
        logging.info("Processed bullpen fatigue for %s", target_date)

    logging.info("Done.")


if __name__ == "__main__":
    main()
