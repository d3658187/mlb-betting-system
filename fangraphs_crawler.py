#!/usr/bin/env python3
"""PyBaseball-based FanGraphs stats loader for advanced stats.

This is the B-layer (on-demand补值) data source in the A/B hybrid pipeline.
Data source now uses pybaseball (FanGraphs data) instead of direct FanGraphs API.

Usage:
  python fangraphs_crawler.py --season 2026 --mode pitchers --out ./data/fangraphs_pitchers_2026.csv
  python fangraphs_crawler.py --season 2026 --mode teams --out ./data/fangraphs_teams_2026.csv
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
    python fangraphs_crawler.py --season 2026 --mode both --store-db
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from sqlalchemy import create_engine, text

try:
    import cloudscraper
except Exception:  # pragma: no cover
    cloudscraper = None

from pybaseball import fg_pitching_data, fg_team_batting_data, playerid_reverse_lookup

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

FANGRAPHS_API_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
FANGRAPHS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

FG_TEAM_ALIAS = {
    "ARI": "AZ",
    "TBR": "TB",
    "SDP": "SD",
    "KCR": "KC",
    "SFG": "SF",
    "WSN": "WSH",
    "CHW": "CWS",
}


@dataclass
class PyBaseballParams:
    season: int
    season1: Optional[int] = None
    qual: int = 0


# ---------------------
# DB helpers
# ---------------------

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    return create_engine(db_url, pool_pre_ping=True)


def ensure_tables(conn):
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS fangraphs_pitchers (
              id BIGSERIAL PRIMARY KEY,
              season INTEGER NOT NULL,
              player_id INTEGER,
              name TEXT,
              team TEXT,
              fip NUMERIC(6,3),
              xfip NUMERIC(6,3),
              k_per_9 NUMERIC(6,3),
              bb_per_9 NUMERIC(6,3),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              UNIQUE (season, player_id)
            );
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS fangraphs_team_batting (
              id BIGSERIAL PRIMARY KEY,
              season INTEGER NOT NULL,
              team_id INTEGER,
              team TEXT,
              wrc_plus NUMERIC(6,2),
              woba NUMERIC(6,4),
              xwoba NUMERIC(6,4),
              ops_plus NUMERIC(6,2),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              UNIQUE (season, team_id)
            );
            """
        )
    )

    # Ensure columns exist for older deployments
    conn.execute(text("ALTER TABLE fangraphs_pitchers ADD COLUMN IF NOT EXISTS k_per_9 NUMERIC(6,3)"))
    conn.execute(text("ALTER TABLE fangraphs_pitchers ADD COLUMN IF NOT EXISTS bb_per_9 NUMERIC(6,3)"))
    conn.execute(text("ALTER TABLE fangraphs_team_batting ADD COLUMN IF NOT EXISTS woba NUMERIC(6,4)"))
    conn.execute(text("ALTER TABLE fangraphs_team_batting ADD COLUMN IF NOT EXISTS xwoba NUMERIC(6,4)"))
    conn.execute(text("ALTER TABLE fangraphs_team_batting ADD COLUMN IF NOT EXISTS ops_plus NUMERIC(6,2)"))


def upsert_pitchers(conn, rows: Iterable[Dict]):
    if not rows:
        return
    sql = text(
        """
        INSERT INTO fangraphs_pitchers (
          season, player_id, name, team, fip, xfip, k_per_9, bb_per_9
        ) VALUES (
          :season, :player_id, :name, :team, :fip, :xfip, :k_per_9, :bb_per_9
        )
        ON CONFLICT (season, player_id) DO UPDATE
          SET name = EXCLUDED.name,
              team = EXCLUDED.team,
              fip = EXCLUDED.fip,
              xfip = EXCLUDED.xfip,
              k_per_9 = EXCLUDED.k_per_9,
              bb_per_9 = EXCLUDED.bb_per_9,
              updated_at = now();
        """
    )
    conn.execute(sql, list(rows))


def upsert_teams(conn, rows: Iterable[Dict]):
    if not rows:
        return
    sql = text(
        """
        INSERT INTO fangraphs_team_batting (
          season, team_id, team, wrc_plus, woba, xwoba, ops_plus
        ) VALUES (
          :season, :team_id, :team, :wrc_plus, :woba, :xwoba, :ops_plus
        )
        ON CONFLICT (season, team_id) DO UPDATE
          SET team = EXCLUDED.team,
              wrc_plus = EXCLUDED.wrc_plus,
              woba = EXCLUDED.woba,
              xwoba = EXCLUDED.xwoba,
              ops_plus = EXCLUDED.ops_plus,
              updated_at = now();
        """
    )
    conn.execute(sql, list(rows))


def map_team_ids(conn, rows: Iterable[Dict]) -> List[Dict]:
    rows = list(rows)
    if not rows:
        return rows
    team_rows = list(
        conn.execute(text("SELECT mlb_team_id, abbreviation, name FROM teams")).mappings()
    )
    abbr_map = {
        (r["abbreviation"] or "").upper(): r["mlb_team_id"]
        for r in team_rows
        if r.get("abbreviation")
    }
    name_map = {
        (r["name"] or "").lower(): r["mlb_team_id"]
        for r in team_rows
        if r.get("name")
    }
    missing = 0
    for row in rows:
        team_label = (row.get("team") or "").strip()
        if not team_label:
            continue
        team_key = team_label.upper()
        mapped = abbr_map.get(team_key)
        if mapped is None:
            alias = FG_TEAM_ALIAS.get(team_key)
            if alias:
                mapped = abbr_map.get(alias)
        if mapped is None:
            mapped = name_map.get(team_label.lower())
        if mapped is not None:
            row["team_id"] = int(mapped)
        else:
            missing += 1
    if missing:
        logging.warning("Team ID mapping missing for %d rows (FanGraphs team labels not matched)", missing)
    return rows


# ---------------------
# FanGraphs API helpers
# ---------------------

def _strip_html(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    # Strip anchor tags and HTML entities
    text = re.sub(r"<[^>]+>", "", str(value))
    return text.replace("&amp;", "&").strip()


def _fetch_fg_api(params: Dict) -> List[Dict]:
    session = None
    if cloudscraper is not None:
        session = cloudscraper.create_scraper()
    client = session or requests
    resp = client.get(FANGRAPHS_API_URL, params=params, headers=FANGRAPHS_HEADERS, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"FanGraphs API error {resp.status_code}: {resp.text[:200]}")
    payload = resp.json()
    return payload.get("data") or []


# ---------------------
# PyBaseball helpers
# ---------------------

def _to_float(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _pick_value(row, keys: List[str]):
    for key in keys:
        if isinstance(row, dict):
            if key in row:
                return row.get(key)
        else:
            if key in row.index:
                return row.get(key)
    return None


def _chunked(seq: List[int], size: int = 500) -> List[List[int]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _map_fangraphs_to_mlbam(fg_ids: List[int]) -> Dict[int, Optional[int]]:
    if not fg_ids:
        return {}
    mapping: Dict[int, Optional[int]] = {}
    for chunk in _chunked(sorted(set(int(x) for x in fg_ids if x is not None))):
        df = playerid_reverse_lookup(chunk, key_type="fangraphs")
        if df.empty:
            continue
        for _, row in df.iterrows():
            fg_id = int(row.get("key_fangraphs")) if row.get("key_fangraphs") is not None else None
            mlb_id = int(row.get("key_mlbam")) if row.get("key_mlbam") is not None else None
            if fg_id is not None:
                mapping[fg_id] = mlb_id
    return mapping


def _to_int(val) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def fetch_pitcher_fip_xfip_api(params: PyBaseballParams) -> List[Dict]:
    query = {
        "pos": "all",
        "stats": "pit",
        "lg": "all",
        "qual": params.qual,
        "type": 8,
        "season": params.season,
        "month": 0,
        "season1": params.season1 or params.season,
        "ind": 1,
        "team": 0,
        "rost": 0,
        "age": 0,
        "filter": "",
        "players": "",
        "page": "1_1000000",
    }
    data = _fetch_fg_api(query)
    if not data:
        return []
    rows: List[Dict] = []
    for row in data:
        name = _strip_html(row.get("Name"))
        team = _strip_html(row.get("Team"))
        fip = _to_float(row.get("FIP"))
        xfip = _to_float(row.get("xFIP"))
        k_per_9 = _to_float(_pick_value(row, ["K/9", "K9"]))
        bb_per_9 = _to_float(_pick_value(row, ["BB/9", "BB9"]))
        if not name:
            continue
        rows.append(
            {
                "season": params.season,
                "player_id": _to_int(row.get("xMLBAMID")),
                "name": name,
                "team": team,
                "fip": fip,
                "xfip": xfip,
                "k_per_9": k_per_9,
                "bb_per_9": bb_per_9,
            }
        )
    logging.info("Fetched FanGraphs API pitcher rows: %d", len(rows))
    return rows


def fetch_team_wrc_plus_api(params: PyBaseballParams) -> List[Dict]:
    query = {
        "pos": "all",
        "stats": "bat",
        "lg": "all",
        "qual": params.qual,
        "type": 8,
        "season": params.season,
        "month": 0,
        "season1": params.season1 or params.season,
        "ind": 1,
        "team": "0,ts",
        "rost": 0,
        "age": 0,
        "filter": "",
        "players": "",
        "page": "1_1000000",
    }
    data = _fetch_fg_api(query)
    if not data:
        return []
    rows: List[Dict] = []
    for row in data:
        team = _strip_html(row.get("Team"))
        wrc_plus = _to_float(row.get("wRC+"))
        woba = _to_float(_pick_value(row, ["wOBA", "woba"]))
        xwoba = _to_float(_pick_value(row, ["xwOBA", "xwoba"]))
        ops_plus = _to_float(_pick_value(row, ["OPS+", "OPS_plus", "OPSPlus"]))
        if not team:
            continue
        rows.append(
            {
                "season": params.season,
                "team_id": None,
                "team": team,
                "wrc_plus": wrc_plus,
                "woba": woba,
                "xwoba": xwoba,
                "ops_plus": ops_plus,
            }
        )
    logging.info("Fetched FanGraphs API team rows: %d", len(rows))
    return rows


def fetch_pitcher_fip_xfip_pybaseball(params: PyBaseballParams) -> List[Dict]:
    try:
        df = fg_pitching_data(params.season, params.season1 or params.season, qual=params.qual)
    except Exception as exc:
        logging.warning("pybaseball fg_pitching_data failed: %s", exc)
        return []
    if df is None or df.empty:
        return []

    if "IDfg" not in df.columns:
        raise RuntimeError("pybaseball fg_pitching_data missing IDfg column")

    fg_ids = df["IDfg"].dropna().astype(int).tolist()
    fg_to_mlbam = _map_fangraphs_to_mlbam(fg_ids)

    rows: List[Dict] = []
    missing_mlbam = 0
    for _, row in df.iterrows():
        fg_id = row.get("IDfg")
        if pd.isna(fg_id):
            continue
        fg_id = int(fg_id)
        mlbam_id = fg_to_mlbam.get(fg_id)
        if mlbam_id in (None, ""):
            missing_mlbam += 1
        name = row.get("Name")
        team = row.get("Team")
        fip = _to_float(row.get("FIP"))
        xfip = _to_float(row.get("xFIP"))
        k_per_9 = _to_float(_pick_value(row, ["K/9", "K9"]))
        bb_per_9 = _to_float(_pick_value(row, ["BB/9", "BB9"]))
        if not name:
            continue
        rows.append(
            {
                "season": params.season,
                "player_id": int(mlbam_id) if mlbam_id not in (None, "") else None,
                "name": str(name),
                "team": str(team) if team is not None else None,
                "fip": fip,
                "xfip": xfip,
                "k_per_9": k_per_9,
                "bb_per_9": bb_per_9,
            }
        )
    if missing_mlbam:
        logging.warning("Missing MLBAM ID mapping for %d FanGraphs pitchers", missing_mlbam)
    logging.info("Mapped FanGraphs->MLBAM for %d/%d pitchers", len(rows) - missing_mlbam, len(rows))
    return rows


def fetch_team_wrc_plus_pybaseball(params: PyBaseballParams) -> List[Dict]:
    try:
        df = fg_team_batting_data(params.season, params.season1 or params.season)
    except Exception as exc:
        logging.warning("pybaseball fg_team_batting_data failed: %s", exc)
        return []
    if df is None or df.empty:
        return []

    rows: List[Dict] = []
    for _, row in df.iterrows():
        team = row.get("Team")
        wrc_plus = _to_float(row.get("wRC+"))
        woba = _to_float(_pick_value(row, ["wOBA", "woba"]))
        xwoba = _to_float(_pick_value(row, ["xwOBA", "xwoba"]))
        ops_plus = _to_float(_pick_value(row, ["OPS+", "OPS_plus", "OPSPlus"]))
        if not team:
            continue
        rows.append(
            {
                "season": params.season,
                "team_id": None,
                "team": str(team),
                "wrc_plus": wrc_plus,
                "woba": woba,
                "xwoba": xwoba,
                "ops_plus": ops_plus,
            }
        )
    logging.info("Fetched team wRC+ rows: %d", len(rows))
    return rows


def _fallback_season_rows(params: PyBaseballParams, fetcher) -> List[Dict]:
    fallback_season = (params.season - 1) if params.season else None
    if not fallback_season or fallback_season < 1900:
        return []
    logging.warning("No FanGraphs data for season %s; falling back to %s", params.season, fallback_season)
    fb_params = PyBaseballParams(season=fallback_season, season1=fallback_season, qual=params.qual)
    rows = fetcher(fb_params)
    for row in rows:
        row["season"] = params.season
    return rows


def fetch_pitcher_fip_xfip(params: PyBaseballParams) -> List[Dict]:
    try:
        rows = fetch_pitcher_fip_xfip_api(params)
        if rows:
            return rows
        logging.warning("FanGraphs API returned no pitcher data for season %s", params.season)
    except Exception as exc:
        logging.warning("FanGraphs API pitcher fetch failed: %s", exc)

    rows = fetch_pitcher_fip_xfip_pybaseball(params)
    if rows:
        return rows
    return _fallback_season_rows(params, fetch_pitcher_fip_xfip_pybaseball)


def fetch_team_wrc_plus(params: PyBaseballParams) -> List[Dict]:
    try:
        rows = fetch_team_wrc_plus_api(params)
        if rows:
            return rows
        logging.warning("FanGraphs API returned no team data for season %s", params.season)
    except Exception as exc:
        logging.warning("FanGraphs API team fetch failed: %s", exc)

    rows = fetch_team_wrc_plus_pybaseball(params)
    if rows:
        return rows
    return _fallback_season_rows(params, fetch_team_wrc_plus_pybaseball)


# ---------------------
# IO helpers
# ---------------------

def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        logging.warning("No data to write: %s", path)
        return
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Wrote %d rows -> %s", len(rows), path)


# ---------------------
# CLI
# ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--season1", type=int)
    parser.add_argument("--mode", choices=["pitchers", "teams", "both"], default="both")
    parser.add_argument("--qual", type=int, default=0)
    parser.add_argument("--out", type=str)
    parser.add_argument("--store-db", action="store_true")
    parser.add_argument("--init-db", action="store_true")
    args = parser.parse_args()

    params = PyBaseballParams(
        season=args.season,
        season1=args.season1,
        qual=args.qual,
    )

    pitcher_rows: List[Dict] = []
    team_rows: List[Dict] = []

    if args.mode in ("pitchers", "both"):
        logging.info("Fetching pitcher FIP/xFIP via pybaseball for season %s", args.season)
        pitcher_rows = fetch_pitcher_fip_xfip(params)
        if args.out:
            out_path = args.out if args.mode == "pitchers" else args.out.replace(".csv", "_pitchers.csv")
            write_csv(pitcher_rows, out_path)

    if args.mode in ("teams", "both"):
        logging.info("Fetching team wRC+ via pybaseball for season %s", args.season)
        team_rows = fetch_team_wrc_plus(params)
        if args.out:
            out_path = args.out if args.mode == "teams" else args.out.replace(".csv", "_teams.csv")
            write_csv(team_rows, out_path)

    if args.store_db:
        engine = get_engine()
        with engine.begin() as conn:
            if args.init_db:
                ensure_tables(conn)
            if pitcher_rows:
                upsert_pitchers(conn, pitcher_rows)
            if team_rows:
                team_rows = map_team_ids(conn, team_rows)
                upsert_teams(conn, team_rows)
        logging.info("Stored data into DB")

    logging.info("Done.")


if __name__ == "__main__":
    main()
