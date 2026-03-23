#!/usr/bin/env python3
"""Fetch game-time weather from Open-Meteo and store in Postgres.

Usage:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  python weather_crawler.py --date 2026-03-13 --store-db

Notes:
- Pulls games from `games` table (mlb_game_id + game_datetime).
- Uses MLB Stats API to resolve venue IDs -> lat/lon.
- Queries Open-Meteo hourly data and picks the closest hour to game start.
"""
from __future__ import annotations

import argparse
import calendar
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from sqlalchemy import create_engine, text

MLB_API = "https://statsapi.mlb.com/api/v1"
OPEN_METEO_FORECAST_API = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------
# DB
# ---------------------

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    return create_engine(db_url, pool_pre_ping=True)


def ensure_game_weather_table(conn):
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS game_weather (
              id BIGSERIAL PRIMARY KEY,
              mlb_game_id BIGINT NOT NULL,
              game_datetime TIMESTAMPTZ,
              venue TEXT,
              latitude DOUBLE PRECISION,
              longitude DOUBLE PRECISION,
              temperature_c NUMERIC(6,2),
              relative_humidity NUMERIC(6,2),
              wind_speed NUMERIC(6,2),
              wind_direction NUMERIC(6,2),
              data_time TIMESTAMPTZ,
              source TEXT DEFAULT 'open-meteo',
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              UNIQUE (mlb_game_id)
            );
            """
        )
    )


def upsert_game_weather(conn, rows: Iterable[Dict]):
    rows = list(rows)
    if not rows:
        return
    sql = text(
        """
        INSERT INTO game_weather (
          mlb_game_id,
          game_datetime,
          venue,
          latitude,
          longitude,
          temperature_c,
          relative_humidity,
          wind_speed,
          wind_direction,
          data_time,
          source
        ) VALUES (
          :mlb_game_id,
          :game_datetime,
          :venue,
          :latitude,
          :longitude,
          :temperature_c,
          :relative_humidity,
          :wind_speed,
          :wind_direction,
          :data_time,
          :source
        )
        ON CONFLICT (mlb_game_id) DO UPDATE
          SET game_datetime = EXCLUDED.game_datetime,
              venue = EXCLUDED.venue,
              latitude = EXCLUDED.latitude,
              longitude = EXCLUDED.longitude,
              temperature_c = EXCLUDED.temperature_c,
              relative_humidity = EXCLUDED.relative_humidity,
              wind_speed = EXCLUDED.wind_speed,
              wind_direction = EXCLUDED.wind_direction,
              data_time = EXCLUDED.data_time,
              source = EXCLUDED.source,
              updated_at = now();
        """
    )
    conn.execute(sql, rows)


def ensure_weather_cache_table(conn):
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS weather_cache (
              id BIGSERIAL PRIMARY KEY,
              venue_id BIGINT NOT NULL,
              venue TEXT,
              latitude DOUBLE PRECISION,
              longitude DOUBLE PRECISION,
              month_start DATE NOT NULL,
              month_end DATE NOT NULL,
              payload TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              UNIQUE (venue_id, month_start)
            );
            """
        )
    )


def load_weather_cache(engine, venue_id: int, month_start: date) -> Optional[Dict]:
    sql = text(
        """
        SELECT payload
          FROM weather_cache
         WHERE venue_id = :venue_id
           AND month_start = :month_start
        """
    )
    with engine.connect() as conn:
        row = conn.execute(sql, {"venue_id": venue_id, "month_start": month_start}).fetchone()
    if not row:
        return None
    payload = row[0]
    if not payload:
        return None
    try:
        return json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError:
        return None


def upsert_weather_cache(
    engine,
    *,
    venue_id: int,
    venue: Optional[str],
    latitude: float,
    longitude: float,
    month_start: date,
    month_end: date,
    payload: Dict,
):
    sql = text(
        """
        INSERT INTO weather_cache (
          venue_id,
          venue,
          latitude,
          longitude,
          month_start,
          month_end,
          payload
        ) VALUES (
          :venue_id,
          :venue,
          :latitude,
          :longitude,
          :month_start,
          :month_end,
          :payload
        )
        ON CONFLICT (venue_id, month_start) DO UPDATE
          SET venue = EXCLUDED.venue,
              latitude = EXCLUDED.latitude,
              longitude = EXCLUDED.longitude,
              month_end = EXCLUDED.month_end,
              payload = EXCLUDED.payload,
              updated_at = now();
        """
    )
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "venue_id": venue_id,
                "venue": venue,
                "latitude": latitude,
                "longitude": longitude,
                "month_start": month_start,
                "month_end": month_end,
                "payload": json.dumps(payload, ensure_ascii=False),
            },
        )


def load_games(
    engine,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    only_missing: bool = True,
) -> List[Dict]:
    sql = """
    SELECT g.mlb_game_id, g.game_date, g.game_datetime, g.venue
      FROM games g
    """
    if only_missing:
        sql += " LEFT JOIN game_weather w ON g.mlb_game_id = w.mlb_game_id"
    sql += " WHERE g.game_datetime IS NOT NULL"
    if only_missing:
        sql += " AND w.mlb_game_id IS NULL"
    params: Dict[str, date] = {}
    if start_date and end_date:
        sql += " AND g.game_date BETWEEN :start_date AND :end_date"
        params = {"start_date": start_date, "end_date": end_date}
    sql += " ORDER BY g.game_date, g.game_datetime"

    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


# ---------------------
# Helpers
# ---------------------

def parse_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        v = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return None
    return None


def request_json(url: str, params: Optional[Dict] = None, retries: int = 8, backoff: int = 10) -> Dict:
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
# MLB venue lookup
# ---------------------

def fetch_schedule(target_date: date) -> Dict:
    params = {
        "sportId": 1,
        "date": target_date.isoformat(),
    }
    return request_json(f"{MLB_API}/schedule", params)


def parse_schedule_venues(schedule_json: Dict) -> Dict[int, Dict]:
    """Return mapping of gamePk -> {venue_id, venue_name}."""
    mapping: Dict[int, Dict] = {}
    for date_block in schedule_json.get("dates", []):
        for g in date_block.get("games", []):
            game_pk = g.get("gamePk")
            venue = g.get("venue", {}) or {}
            venue_id = venue.get("id")
            if not game_pk or not venue_id:
                continue
            mapping[int(game_pk)] = {
                "venue_id": int(venue_id),
                "venue_name": venue.get("name"),
            }
    return mapping


def fetch_venues(venue_ids: Iterable[int]) -> Dict[int, Dict]:
    ids = [str(v) for v in sorted(set(venue_ids)) if v is not None]
    if not ids:
        return {}
    params = {"venueIds": ",".join(ids), "hydrate": "location"}
    payload = request_json(f"{MLB_API}/venues", params)
    venues: Dict[int, Dict] = {}
    for v in payload.get("venues", []):
        vid = v.get("id")
        loc = v.get("location", {}) or {}
        coords = loc.get("defaultCoordinates", {}) or {}
        lat = loc.get("latitude") or coords.get("latitude")
        lon = loc.get("longitude") or coords.get("longitude")
        venues[int(vid)] = {
            "name": v.get("name"),
            "latitude": lat,
            "longitude": lon,
        }
    return venues


# ---------------------
# Open-Meteo
# ---------------------

def fetch_open_meteo(lat: float, lon: float, start_date: date, end_date: date) -> Dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "UTC",
    }

    api = OPEN_METEO_ARCHIVE_API
    data = request_json(api, params)
    time.sleep(60)
    return data


def month_bounds(target_date: date) -> Tuple[date, date]:
    last_day = calendar.monthrange(target_date.year, target_date.month)[1]
    return date(target_date.year, target_date.month, 1), date(target_date.year, target_date.month, last_day)


def select_closest_hour(weather: Dict, target_dt: datetime) -> Optional[Dict]:
    hourly = weather.get("hourly", {}) or {}
    times = hourly.get("time", [])
    if not times:
        return None

    def parse_hour(t: str) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(t)
        except ValueError:
            return None

    parsed = [parse_hour(t) for t in times]
    target_dt = target_dt.astimezone(timezone.utc).replace(tzinfo=None)
    best_idx = None
    best_diff = None
    for idx, dt in enumerate(parsed):
        if dt is None:
            continue
        diff = abs((dt - target_dt).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_idx = idx

    if best_idx is None:
        return None

    return {
        "data_time": parsed[best_idx],
        "temperature_c": hourly.get("temperature_2m", [None])[best_idx],
        "relative_humidity": hourly.get("relative_humidity_2m", [None])[best_idx],
        "wind_speed": hourly.get("wind_speed_10m", [None])[best_idx],
        "wind_direction": hourly.get("wind_direction_10m", [None])[best_idx],
    }


# ---------------------
# Main
# ---------------------

def build_weather_rows(games: List[Dict], engine) -> List[Dict]:
    if not games:
        return []

    # group by date -> schedule -> venue mapping
    date_map: Dict[date, Dict[int, Dict]] = {}
    venue_cache: Dict[int, Dict] = {}

    for g in games:
        g_date = g.get("game_date")
        if isinstance(g_date, str):
            g_date = datetime.strptime(g_date, "%Y-%m-%d").date()
        if g_date and g_date not in date_map:
            sched = fetch_schedule(g_date)
            date_map[g_date] = parse_schedule_venues(sched)

            venue_ids = [info["venue_id"] for info in date_map[g_date].values()]
            venue_cache.update(fetch_venues(venue_ids))

    rows: List[Dict] = []
    weather_cache: Dict[Tuple[int, date], Dict] = {}
    for g in games:
        game_pk = g.get("mlb_game_id")
        g_date = g.get("game_date")
        if isinstance(g_date, str):
            g_date = datetime.strptime(g_date, "%Y-%m-%d").date()
        g_dt = parse_datetime(g.get("game_datetime"))
        if not game_pk or not g_date or not g_dt:
            continue

        venue_info = date_map.get(g_date, {}).get(int(game_pk))
        if not venue_info:
            logging.warning("Missing venue for game %s on %s", game_pk, g_date)
            continue

        venue_id = venue_info.get("venue_id")
        if not venue_id:
            logging.warning("Missing venue id for game %s on %s", game_pk, g_date)
            continue

        venue = venue_info.get("venue_name") or g.get("venue")
        v = venue_cache.get(int(venue_id)) if venue_id else None
        if not v or v.get("latitude") is None or v.get("longitude") is None:
            logging.warning("Missing lat/lon for venue %s (game %s)", venue_id, game_pk)
            continue

        lat = float(v["latitude"])
        lon = float(v["longitude"])
        month_start, month_end = month_bounds(g_date)
        cache_key = (int(venue_id), month_start)
        weather = weather_cache.get(cache_key)
        if weather is None:
            weather = load_weather_cache(engine, int(venue_id), month_start)
            if weather is None:
                weather = fetch_open_meteo(lat, lon, month_start, month_end)
                upsert_weather_cache(
                    engine,
                    venue_id=int(venue_id),
                    venue=venue,
                    latitude=lat,
                    longitude=lon,
                    month_start=month_start,
                    month_end=month_end,
                    payload=weather,
                )
            weather_cache[cache_key] = weather
        hour = select_closest_hour(weather, g_dt)
        if not hour:
            logging.warning("No hourly weather for game %s", game_pk)
            continue

        rows.append({
            "mlb_game_id": int(game_pk),
            "game_datetime": g_dt,
            "venue": venue,
            "latitude": lat,
            "longitude": lon,
            "temperature_c": hour.get("temperature_c"),
            "relative_humidity": hour.get("relative_humidity"),
            "wind_speed": hour.get("wind_speed"),
            "wind_direction": hour.get("wind_direction"),
            "data_time": hour.get("data_time"),
            "source": "open-meteo",
        })

    return rows


def iter_month_ranges(start_date: date, end_date: date):
    current = date(start_date.year, start_date.month, 1)
    while current <= end_date:
        last_day = calendar.monthrange(current.year, current.month)[1]
        month_end = date(current.year, current.month, last_day)
        range_start = max(start_date, current)
        range_end = min(end_date, month_end)
        label = current.strftime("%Y-%m")
        yield label, range_start, range_end
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target date (YYYY-MM-DD)")
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--store-db", action="store_true", help="Write weather to DB")
    parser.add_argument("--limit", type=int, help="Limit number of games (debug)")
    args = parser.parse_args()

    engine = get_engine()

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    if args.date:
        start_date = end_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        raise SystemExit("Provide --date or --start-date/--end-date")

    with engine.begin() as conn:
        ensure_game_weather_table(conn)
        ensure_weather_cache_table(conn)

    total_rows = 0
    for label, range_start, range_end in iter_month_ranges(start_date, end_date):
        games = load_games(engine, range_start, range_end)
        if args.limit:
            games = games[: args.limit]

        if not games:
            logging.info("No games found for %s (%s to %s).", label, range_start, range_end)
            logging.info("Completed %s.", label)
            continue

        rows = build_weather_rows(games, engine)
        if not rows:
            logging.warning("No weather rows built for %s (%s to %s).", label, range_start, range_end)
            logging.info("Completed %s.", label)
            continue

        if args.store_db:
            with engine.begin() as conn:
                upsert_game_weather(conn, rows)
            logging.info(
                "Stored %d weather rows for %s (%s to %s).",
                len(rows),
                label,
                range_start,
                range_end,
            )
        else:
            logging.info(
                "Built %d weather rows for %s (%s to %s) (not stored).",
                len(rows),
                label,
                range_start,
                range_end,
            )

        total_rows += len(rows)
        logging.info("Completed %s.", label)

    if args.store_db:
        logging.info("Done. Total stored rows: %d", total_rows)
    else:
        logging.info("Done. Total built rows: %d", total_rows)


if __name__ == "__main__":
    main()
