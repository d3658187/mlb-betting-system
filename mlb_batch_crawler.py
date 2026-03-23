#!/usr/bin/env python3
"""Batch MLB Stats API crawler with concurrency, caching, and retries.

Outputs (CSV, compatible with mlb_stats_api_crawler.py):
- games_{season}.csv with columns: game_pk, game_date, home_team, away_team,
  home_score, away_score, home_win, season
- teams_mlb.csv (optional, default enabled)

Usage:
  python mlb_batch_crawler.py --start-year 2022 --end-year 2024 --out-dir ./data/mlb_stats_api/batch
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from datetime import date
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://statsapi.mlb.com/api/v1"
FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------
# Utils
# ---------------------

def _safe_mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _fmt_eta(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------
# Requests with retry
# ---------------------

def request_json(url: str, params: Optional[Dict] = None, retries: int = 6, backoff: int = 2) -> Dict:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else backoff * (2 ** attempt)
                logging.warning("Rate limited (429). Retrying in %ss...", wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= retries - 1:
                break
            wait = backoff * (2 ** attempt)
            logging.warning("Request failed (%s). Retrying in %ss...", exc, wait)
            time.sleep(wait)
    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed without exception")


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


# ---------------------
# Schedule + Game fetch
# ---------------------

def fetch_schedule_for_season(season: int) -> Dict:
    params = {"sportId": 1, "season": season}
    return request_json(f"{BASE_URL}/schedule", params=params)


def extract_game_pks(schedule_json: Dict) -> List[int]:
    pks: List[int] = []
    for date_block in schedule_json.get("dates", []):
        for g in date_block.get("games", []):
            game_pk = g.get("gamePk")
            if game_pk is None:
                continue
            pks.append(int(game_pk))
    # preserve order but de-dup
    seen = set()
    uniq = []
    for pk in pks:
        if pk in seen:
            continue
        seen.add(pk)
        uniq.append(pk)
    return uniq


def fetch_game_detail(game_pk: int) -> Dict:
    return request_json(FEED_URL.format(game_pk=game_pk))


def parse_game_detail(payload: Dict) -> Optional[Dict]:
    game_data = payload.get("gameData", {}) or {}
    live_data = payload.get("liveData", {}) or {}

    # Some payloads (e.g., 2024) store officialDate under gameData.datetime
    datetime_block = game_data.get("datetime") or {}
    official_date = (
        game_data.get("officialDate")
        or datetime_block.get("officialDate")
        or datetime_block.get("originalDate")
    )
    # Fallback: if only dateTime exists, take date portion
    if not official_date:
        dt = datetime_block.get("dateTime")
        if isinstance(dt, str) and "T" in dt:
            official_date = dt.split("T", 1)[0]
    season = game_data.get("game", {}).get("season")

    teams = game_data.get("teams", {}) or {}
    home_team = (teams.get("home") or {}).get("name")
    away_team = (teams.get("away") or {}).get("name")

    linescore = live_data.get("linescore", {}) or {}
    home_score = (linescore.get("teams", {}) or {}).get("home", {}).get("runs")
    away_score = (linescore.get("teams", {}) or {}).get("away", {}).get("runs")

    # fallback to boxscore if linescore missing
    if home_score is None or away_score is None:
        boxscore = live_data.get("boxscore", {}) or {}
        home_score = home_score if home_score is not None else (boxscore.get("teams", {}) or {}).get("home", {}).get("teamStats", {}).get("batting", {}).get("runs")
        away_score = away_score if away_score is not None else (boxscore.get("teams", {}) or {}).get("away", {}).get("teamStats", {}).get("batting", {}).get("runs")

    home_win = None
    if home_score is not None and away_score is not None:
        try:
            home_win = float(home_score) > float(away_score)
        except Exception:
            home_win = None

    if not official_date or not home_team or not away_team:
        return None

    return {
        "game_pk": game_data.get("game", {}).get("pk"),
        "game_date": official_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
        "season": int(season) if season else None,
    }


# ---------------------
# Cache
# ---------------------

def cache_path(cache_dir: str, game_pk: int) -> str:
    return os.path.join(cache_dir, f"game_{game_pk}.json")


def load_from_cache(cache_dir: str, game_pk: int) -> Optional[Dict]:
    path = cache_path(cache_dir, game_pk)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_to_cache(cache_dir: str, game_pk: int, payload: Dict) -> None:
    path = cache_path(cache_dir, game_pk)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        return


# ---------------------
# Runner
# ---------------------

def crawl_season(season: int, out_dir: str, cache_dir: str, concurrency: int, retries: int, backoff: int, force: bool) -> pd.DataFrame:
    logging.info("Fetching schedule for %s", season)
    schedule = fetch_schedule_for_season(season)
    game_pks = extract_game_pks(schedule)
    if not game_pks:
        logging.warning("No games found for season %s", season)
        return pd.DataFrame()

    total = len(game_pks)
    logging.info("Season %s: %d games", season, total)

    lock = threading.Lock()
    completed = 0
    start_ts = time.time()
    rows: List[Dict] = []

    def task(game_pk: int) -> Optional[Dict]:
        if not force:
            cached = load_from_cache(cache_dir, game_pk)
            if cached:
                return parse_game_detail(cached)
        payload = request_json(FEED_URL.format(game_pk=game_pk), params=None, retries=retries, backoff=backoff)
        save_to_cache(cache_dir, game_pk, payload)
        return parse_game_detail(payload)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(task, pk): pk for pk in game_pks}
        for fut in as_completed(futures):
            pk = futures[fut]
            try:
                row = fut.result()
                if row:
                    rows.append(row)
            except Exception as exc:
                logging.warning("Failed gamePk=%s (%s)", pk, exc)
            finally:
                with lock:
                    completed += 1
                    elapsed = time.time() - start_ts
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    if completed % 25 == 0 or completed == total:
                        logging.info("Progress %d/%d (%.1f%%) | rate %.2f/s | ETA %s",
                                     completed, total, completed * 100 / total, rate, _fmt_eta(eta))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["game_date", "home_team", "away_team"])
    df = df.drop_duplicates(subset=["game_pk"], keep="first")
    df = df.sort_values(["game_date", "game_pk"]) if "game_date" in df.columns else df
    return df


def run(start_year: int, end_year: int, out_dir: str, concurrency: int, retries: int, backoff: int, force: bool, skip_teams: bool) -> None:
    _safe_mkdir(out_dir)
    cache_dir = os.path.join(out_dir, "cache")
    _safe_mkdir(cache_dir)

    if not skip_teams:
        try:
            teams = fetch_teams()
            if teams:
                pd.DataFrame(teams).to_csv(os.path.join(out_dir, "teams_mlb.csv"), index=False)
                logging.info("Saved %d teams", len(teams))
        except Exception as exc:
            logging.warning("Teams fetch failed: %s", exc)

    all_df: List[pd.DataFrame] = []
    for season in range(start_year, end_year + 1):
        df = crawl_season(season, out_dir, cache_dir, concurrency, retries, backoff, force)
        if not df.empty:
            out_path = os.path.join(out_dir, f"games_{season}.csv")
            df.to_csv(out_path, index=False)
            all_df.append(df)
            logging.info("Saved %d games to %s", len(df), out_path)

    if all_df:
        merged = pd.concat(all_df, ignore_index=True)
        merged.to_csv(os.path.join(out_dir, f"games_{start_year}_{end_year}.csv"), index=False)
        logging.info("Saved merged games %d rows", len(merged))


# ---------------------
# CLI
# ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, required=True, help="Start season year, e.g., 2022")
    p.add_argument("--end-year", type=int, required=True, help="End season year, e.g., 2024")
    p.add_argument("--out-dir", default="./data/mlb_stats_api/batch", help="Output directory")
    p.add_argument("--concurrency", type=int, default=8, help="Concurrent requests (5-10 recommended)")
    p.add_argument("--retries", type=int, default=6, help="Max retries per request")
    p.add_argument("--backoff", type=int, default=2, help="Backoff base seconds")
    p.add_argument("--force", action="store_true", help="Ignore cache and refetch")
    p.add_argument("--skip-teams", action="store_true", help="Skip teams CSV output")
    return p.parse_args()


def main():
    args = parse_args()
    if args.end_year < args.start_year:
        raise SystemExit("end-year must be >= start-year")
    run(
        start_year=args.start_year,
        end_year=args.end_year,
        out_dir=args.out_dir,
        concurrency=args.concurrency,
        retries=args.retries,
        backoff=args.backoff,
        force=args.force,
        skip_teams=args.skip_teams,
    )


if __name__ == "__main__":
    main()
