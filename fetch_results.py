#!/usr/bin/env python3
"""Fetch completed MLB game results from The Odds API scores endpoint.

Usage:
  THE_ODDS_API_KEY=xxx python fetch_results.py --date 2026-03-24

Output:
  data/results/YYYY-MM-DD.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import requests

API_ENDPOINT = "https://api.the-odds-api.com/v4/sports/baseball_mlb/scores"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _parse_scores(item: Dict) -> Optional[Dict]:
    if not item:
        return None
    if not item.get("completed"):
        return None

    home_team = item.get("home_team")
    away_team = item.get("away_team")
    scores = item.get("scores") or []

    if not home_team or not away_team or not scores:
        return None

    home_score = None
    away_score = None
    for score in scores:
        name = score.get("name")
        val = score.get("score")
        if name == home_team:
            home_score = val
        elif name == away_team:
            away_score = val

    if home_score is None or away_score is None:
        return None

    return {
        "game_date": item.get("commence_time", "")[:10] or item.get("last_update", "")[:10],
        "home_team": home_team,
        "away_team": away_team,
        "home_score": float(home_score),
        "away_score": float(away_score),
        "completed": True,
        "commence_time": item.get("commence_time"),
        "last_update": item.get("last_update"),
    }


def fetch_results(api_key: str, target_date: str, timeout: int = 20) -> List[Dict]:
    params = {
        "apiKey": api_key,
        "date": target_date,
    }
    resp = requests.get(API_ENDPOINT, params=params, timeout=timeout)

    remaining = resp.headers.get("x-requests-remaining")
    used = resp.headers.get("x-requests-used")
    if remaining is not None:
        logging.info("The Odds API requests remaining: %s (used=%s)", remaining, used)

    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        raise RuntimeError(f"Rate limited (429). Retry-After: {retry_after}")

    resp.raise_for_status()
    payload = resp.json()

    rows: List[Dict] = []
    for item in payload or []:
        parsed = _parse_scores(item)
        if parsed:
            rows.append(parsed)
    return rows


def save_json(rows: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--out", help="Output JSON file path")
    p.add_argument("--force", action="store_true", help="Overwrite existing file")
    p.add_argument("--timeout", type=int, default=20, help="Request timeout seconds")
    return p.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date).isoformat() if args.date else date.today().isoformat()

    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("Missing THE_ODDS_API_KEY environment variable")

    out_path = Path(args.out) if args.out else Path("data/results") / f"{target_date}.json"
    if out_path.exists() and not args.force:
        logging.info("Output exists, skip fetch: %s", out_path)
        return

    rows = fetch_results(api_key, target_date, timeout=args.timeout)
    if not rows:
        logging.warning("No completed games returned for %s", target_date)

    save_json(rows, out_path)
    logging.info("Saved %d completed games to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
