#!/usr/bin/env python3
"""Fetch MLB odds from The Odds API and save to JSON.

Output format (list of games per sportsbook):
[
  {
    "game_date": "YYYY-MM-DD",
    "home_team": "...",
    "away_team": "...",
    "game_time": "HH:MM",
    "markets": [
      {"market": "moneyline", "selection": "home", "price": -120, "line": null},
      {"market": "spread", "selection": "away", "price": 110, "line": 1.5},
      {"market": "total", "selection": "over", "price": -105, "line": 8.5}
    ],
    "source": "the_odds_api:bookmaker_key"
  }
]

Usage:
  THE_ODDS_API_KEY=xxx python fetch_odds_api.py --date 2026-03-24
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

from taiwan_lottery_crawler import MarketOdd, GameOdds

API_ENDPOINT = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
DEFAULT_MARKETS = "h2h,spreads,totals"
DEFAULT_REGIONS = "us"
DEFAULT_DATE_FORMAT = "iso"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# -------------------------
# Helpers
# -------------------------

def _decimal_to_american(decimal_odds: float) -> Optional[int]:
    if decimal_odds is None or decimal_odds <= 1:
        return None
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))


def _normalize_price(price: Optional[float]) -> Optional[int]:
    """Normalize price into American odds.

    If price looks like decimal (<10), convert; else treat as American.
    """
    if price is None:
        return None
    try:
        val = float(price)
    except Exception:
        return None
    if abs(val) < 10 and val > 1:
        return _decimal_to_american(val)
    return int(round(val))


def _parse_commence_time(iso_ts: Optional[str]) -> (Optional[str], Optional[str]):
    if not iso_ts:
        return None, None
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        dt_et = dt.astimezone(ZoneInfo("US/Eastern"))
        return dt_et.date().isoformat(), dt_et.strftime("%H:%M")
    except Exception:
        return iso_ts.split("T")[0], None


def _build_params(api_key: str) -> Dict[str, str]:
    return {
        "apiKey": api_key,
        "regions": DEFAULT_REGIONS,
        "markets": DEFAULT_MARKETS,
        "dateFormat": DEFAULT_DATE_FORMAT,
    }


def fetch_odds(api_key: str, timeout: int = 20) -> List[Dict]:
    params = _build_params(api_key)
    resp = requests.get(API_ENDPOINT, params=params, timeout=timeout)

    # Rate limit headers
    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None:
        logging.info("The Odds API requests remaining: %s (used=%s)", remaining, used)

    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        raise RuntimeError(f"Rate limited (429). Retry-After: {retry_after}")

    resp.raise_for_status()
    return resp.json()


def parse_odds(payload: List[Dict]) -> List[GameOdds]:
    games: List[GameOdds] = []

    for item in payload or []:
        home_team = item.get("home_team")
        away_team = item.get("away_team")
        game_date, game_time = _parse_commence_time(item.get("commence_time"))

        bookmakers = item.get("bookmakers") or []
        for book in bookmakers:
            book_key = book.get("key") or "unknown"
            markets = book.get("markets") or []

            market_odds: List[MarketOdd] = []
            for m in markets:
                key = m.get("key")
                outcomes = m.get("outcomes") or []

                if key == "h2h":
                    for o in outcomes:
                        name = o.get("name")
                        price = _normalize_price(o.get("price"))
                        if price is None:
                            continue
                        selection = "home" if name == home_team else "away" if name == away_team else name
                        market_odds.append(MarketOdd("moneyline", selection, price))

                elif key == "spreads":
                    for o in outcomes:
                        name = o.get("name")
                        price = _normalize_price(o.get("price"))
                        line = o.get("point")
                        if price is None:
                            continue
                        selection = "home" if name == home_team else "away" if name == away_team else name
                        market_odds.append(MarketOdd("spread", selection, price, line))

                elif key == "totals":
                    for o in outcomes:
                        name = str(o.get("name") or "").lower()
                        price = _normalize_price(o.get("price"))
                        line = o.get("point")
                        if price is None:
                            continue
                        if "over" in name:
                            selection = "over"
                        elif "under" in name:
                            selection = "under"
                        else:
                            selection = name or "total"
                        market_odds.append(MarketOdd("total", selection, price, line))

            if market_odds:
                games.append(
                    GameOdds(
                        game_date=game_date,
                        home_team=home_team,
                        away_team=away_team,
                        game_time=game_time,
                        markets=market_odds,
                        source=f"the_odds_api:{book_key}",
                    )
                )

    return games


def save_json(games: List[GameOdds], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "game_date": g.game_date,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "game_time": g.game_time,
            "markets": [asdict(m) for m in g.markets],
            "source": g.source,
        }
        for g in games
    ]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--out", help="Output JSON file path")
    p.add_argument("--force", action="store_true", help="Overwrite existing file")
    p.add_argument("--timeout", type=int, default=20, help="Request timeout seconds")
    return p.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else date.today()

    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing THE_ODDS_API_KEY environment variable")

    out_path = Path(args.out) if args.out else Path("data/odds") / f"the-odds-api_{target_date.isoformat()}.json"
    if out_path.exists() and not args.force:
        logging.info("Output exists, skip fetch: %s", out_path)
        return

    payload = fetch_odds(api_key, timeout=args.timeout)
    games = parse_odds(payload)

    if not games:
        logging.warning("No odds returned from The Odds API")

    save_json(games, out_path)
    logging.info("Saved %d bookmaker entries to %s", len(games), out_path)


if __name__ == "__main__":
    main()
