#!/usr/bin/env python3
"""Taiwan Sports Lottery (台灣運彩) MLB odds crawler - scaffold.

Goal:
- Fetch MLB games and markets: moneyline (獨贏), run_line (讓分盤), total (大小分)
- Return structured data that can be inserted into PostgreSQL (see init_db.sql)

Notes:
- Taiwan Sports Lottery site is dynamic and uses JS. Playwright is recommended.
- Anti-bot considerations: rotate UA, block heavy resources, add random delays,
  use headful mode + slowMo if needed, and optionally stealth plugins.

Dependencies (suggested):
  pip install playwright beautifulsoup4 lxml
  playwright install

Usage:
  python taiwan_lottery_crawler.py --date 2026-03-13 --out odds.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import requests

# Optional: BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

# Optional: Playwright for rendering
try:
    from playwright.sync_api import sync_playwright
except Exception:  # pragma: no cover
    sync_playwright = None


BASE_URL = "https://www.sportslottery.com.tw"  # 台灣運彩官網
MLB_URL = "https://www.sportslottery.com.tw/sportsbook/daily-coupons"  # 台灣運彩賽事表
APIDATA_BASE = "https://blob3rd.sportslottery.com.tw/apidata"
BASEBALL_SPORT_ID = "34731.1"
JBOT_API_URL = "https://api.sportsbot.tech/v2/odds"

# Optional: map JBot/Taiwan lottery team names -> MLB Stats team names
# Provide a JSON file via TEAM_NAME_MAP_PATH env var:
#   {"波士頓紅襪": "Boston Red Sox", "紐約洋基": "New York Yankees", ...}
TEAM_NAME_MAP_PATH = os.getenv("TEAM_NAME_MAP_PATH")

# English -> Chinese team name map (MLB 30 teams)
EN_TO_TW_TEAM_MAP = {
    "Arizona Diamondbacks": "亞利桑那響尾蛇",
    "Atlanta Braves": "亞特蘭大勇士",
    "Baltimore Orioles": "巴爾的摩金鶯",
    "Boston Red Sox": "波士頓紅襪",
    "Chicago Cubs": "芝加哥小熊",
    "Chicago White Sox": "芝加哥白襪",
    "Cincinnati Reds": "辛辛那提紅人",
    "Cleveland Guardians": "克里夫蘭守護者",
    "Colorado Rockies": "科羅拉多洛磯",
    "Detroit Tigers": "底特律老虎",
    "Houston Astros": "休士頓太空人",
    "Kansas City Royals": "堪薩斯城皇家",
    "Los Angeles Angels": "洛杉磯天使",
    "Los Angeles Dodgers": "洛杉磯道奇",
    "Miami Marlins": "邁阿密馬林魚",
    "Milwaukee Brewers": "密爾瓦基釀酒人",
    "Minnesota Twins": "明尼蘇達雙城",
    "New York Mets": "紐約大都會",
    "New York Yankees": "紐約洋基",
    "Athletics": "運動家",
    "Philadelphia Phillies": "費城費城人",
    "Pittsburgh Pirates": "匹茲堡海盜",
    "San Diego Padres": "聖地牙哥教士",
    "San Francisco Giants": "舊金山巨人",
    "Seattle Mariners": "西雅圖水手",
    "St. Louis Cardinals": "聖路易紅雀",
    "Tampa Bay Rays": "坦帕灣光芒",
    "Texas Rangers": "德州遊騎兵",
    "Toronto Blue Jays": "多倫多藍鳥",
    "Washington Nationals": "華盛頓國民",
}

# Built-in aliases for common Taiwan lottery naming variants
DEFAULT_TEAM_NAME_MAP = {
    # Existing alias
    "Athletics (MLB)": "Athletics",

    # Arizona Diamondbacks
    "亞利桑那響尾蛇": "Arizona Diamondbacks",
    "響尾蛇": "Arizona Diamondbacks",

    # Atlanta Braves
    "亞特蘭大勇士": "Atlanta Braves",
    "勇士": "Atlanta Braves",

    # Baltimore Orioles
    "巴爾的摩金鶯": "Baltimore Orioles",
    "金鶯": "Baltimore Orioles",

    # Boston Red Sox
    "波士頓紅襪": "Boston Red Sox",
    "紅襪": "Boston Red Sox",

    # Chicago Cubs
    "芝加哥小熊": "Chicago Cubs",
    "小熊": "Chicago Cubs",

    # Chicago White Sox
    "芝加哥白襪": "Chicago White Sox",
    "白襪": "Chicago White Sox",

    # Cincinnati Reds
    "辛辛那提紅人": "Cincinnati Reds",
    "紅人": "Cincinnati Reds",

    # Cleveland Guardians (and former Indians)
    "克里夫蘭守護者": "Cleveland Guardians",
    "守護者": "Cleveland Guardians",
    "克里夫蘭印地安人": "Cleveland Guardians",
    "印地安人": "Cleveland Guardians",

    # Colorado Rockies
    "科羅拉多洛磯": "Colorado Rockies",
    "洛磯": "Colorado Rockies",

    # Detroit Tigers
    "底特律老虎": "Detroit Tigers",
    "老虎": "Detroit Tigers",

    # Houston Astros
    "休士頓太空人": "Houston Astros",
    "太空人": "Houston Astros",

    # Kansas City Royals
    "堪薩斯城皇家": "Kansas City Royals",
    "堪薩斯市皇家": "Kansas City Royals",
    "皇家": "Kansas City Royals",

    # Los Angeles Angels
    "洛杉磯天使": "Los Angeles Angels",
    "天使": "Los Angeles Angels",

    # Los Angeles Dodgers
    "洛杉磯道奇": "Los Angeles Dodgers",
    "道奇": "Los Angeles Dodgers",

    # Miami Marlins (and former Florida Marlins)
    "邁阿密馬林魚": "Miami Marlins",
    "馬林魚": "Miami Marlins",
    "佛羅里達馬林魚": "Miami Marlins",

    # Milwaukee Brewers
    "密爾瓦基釀酒人": "Milwaukee Brewers",
    "釀酒人": "Milwaukee Brewers",

    # Minnesota Twins
    "明尼蘇達雙城": "Minnesota Twins",
    "雙城": "Minnesota Twins",

    # New York Mets
    "紐約大都會": "New York Mets",
    "大都會": "New York Mets",

    # New York Yankees
    "紐約洋基": "New York Yankees",
    "洋基": "New York Yankees",

    # Oakland Athletics (now Athletics)
    "奧克蘭運動家": "Athletics",
    "運動家": "Athletics",

    # Philadelphia Phillies
    "費城費城人": "Philadelphia Phillies",
    "費城人": "Philadelphia Phillies",

    # San Diego Padres
    "聖地牙哥教士": "San Diego Padres",
    "教士": "San Diego Padres",

    # San Francisco Giants
    "舊金山巨人": "San Francisco Giants",
    "巨人": "San Francisco Giants",

    # Seattle Mariners
    "西雅圖水手": "Seattle Mariners",
    "水手": "Seattle Mariners",

    # St. Louis Cardinals
    "聖路易紅雀": "St. Louis Cardinals",
    "紅雀": "St. Louis Cardinals",

    # Tampa Bay Rays
    "坦帕灣光芒": "Tampa Bay Rays",
    "光芒": "Tampa Bay Rays",

    # Texas Rangers
    "德州遊騎兵": "Texas Rangers",
    "遊騎兵": "Texas Rangers",

    # Toronto Blue Jays
    "多倫多藍鳥": "Toronto Blue Jays",
    "藍鳥": "Toronto Blue Jays",

    # Washington Nationals
    "華盛頓國民": "Washington Nationals",
    "國民": "Washington Nationals",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]


@dataclass
class MarketOdd:
    market: str  # moneyline | run_line | total | odd_even
    selection: str  # home | away | over | under | odd | even | team name
    price: int  # American odds
    line: Optional[float] = None  # run_line/total line


@dataclass
class GameOdds:
    game_date: str
    home_team: str
    away_team: str
    game_time: Optional[str]
    markets: List[MarketOdd]
    source: str = "taiwan_sports_lottery"


# -------------------------
# JBot API helpers
# -------------------------

def load_team_name_map() -> Dict[str, str]:
    team_map = dict(DEFAULT_TEAM_NAME_MAP)
    if not TEAM_NAME_MAP_PATH:
        return team_map
    try:
        with open(TEAM_NAME_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            team_map.update({str(k).strip(): str(v).strip() for k, v in data.items()})
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to load TEAM_NAME_MAP_PATH: %s", exc)
    return team_map


def normalize_team_name(name: Optional[str], team_map: Optional[Dict[str, str]] = None) -> str:
    if name is None:
        return ""
    if team_map is None:
        team_map = load_team_name_map()
    raw = str(name).strip()
    mapped = team_map.get(raw, raw)
    mapped = mapped.replace("　", " ")
    mapped = re.sub(r"\s+", " ", mapped)
    return mapped.strip()


def fetch_jbot_odds(
    target_date: date,
    sport: str = "MLB",
    mode: str = "close",
    token: Optional[str] = None,
    url: str = JBOT_API_URL,
    timeout: int = 20,
) -> Dict:
    """Fetch Taiwan odds via JBot API.

    Requires X-JBot-Token in headers.
    """
    token = token or os.getenv("JBOT_TOKEN") or os.getenv("SPORTSBOT_TOKEN") or os.getenv("X_JBOT_TOKEN")
    if not token:
        raise RuntimeError("Missing JBot token. Set JBOT_TOKEN/SPORTSBOT_TOKEN/X_JBOT_TOKEN.")

    params = {"sport": sport, "date": target_date.isoformat(), "mode": mode}
    headers = {"X-JBot-Token": token}
    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    status = payload.get("status")
    if status != "OK":
        raise RuntimeError(f"JBot API error: {status}")
    return payload


def parse_jbot_odds(payload: Dict, target_date: date) -> List[GameOdds]:
    team_map = load_team_name_map()
    games: List[GameOdds] = []
    for item in payload.get("data", []) or []:
        away = normalize_team_name(item.get("away", ""), team_map)
        home = normalize_team_name(item.get("home", ""), team_map)
        game_time = item.get("time")

        odds_list = item.get("odds") or []
        if not odds_list:
            continue
        # prefer last odds entry (close). API returns list even when mode=close
        latest = odds_list[-1]
        normal = latest.get("normal") or {}
        away_dec = normal.get("a")
        home_dec = normal.get("h")

        away_price = parse_decimal_or_american(str(away_dec)) if away_dec is not None else None
        home_price = parse_decimal_or_american(str(home_dec)) if home_dec is not None else None

        markets: List[MarketOdd] = []
        if away_price is not None:
            markets.append(MarketOdd("moneyline", "away", away_price))
        if home_price is not None:
            markets.append(MarketOdd("moneyline", "home", home_price))

        if markets:
            games.append(
                GameOdds(
                    game_date=target_date.isoformat(),
                    home_team=home,
                    away_team=away,
                    game_time=game_time,
                    markets=markets,
                    source="jbot",
                )
            )
    return games


def _convert_game_datetime_tz(
    game_date: Optional[str],
    game_time: Optional[str],
    from_tz: str = "Asia/Taipei",
    to_tz: str = "US/Eastern",
) -> Tuple[Optional[str], Optional[str]]:
    """Convert game_date/game_time between timezones.

    Expects game_date=YYYY-MM-DD and game_time=HH:MM. If game_time is missing,
    returns input date unchanged (no reliable conversion).
    """
    if not game_date:
        return None, None
    if not game_time:
        return game_date, None
    try:
        dt = datetime.fromisoformat(f"{game_date}T{game_time}:00").replace(
            tzinfo=ZoneInfo(from_tz)
        )
        dt_out = dt.astimezone(ZoneInfo(to_tz))
        return dt_out.date().isoformat(), dt_out.strftime("%H:%M")
    except Exception:
        return game_date, game_time


def load_manual_odds_json(path: str) -> List[GameOdds]:
    """Load manual odds JSON in GameOdds schema.

    Expected format (list):
    [
      {
        "game_date": "YYYY-MM-DD",
        "home_team": "...",
        "away_team": "...",
        "game_time": "HH:MM",
        "markets": [{"market":"moneyline","selection":"home","price":-120,"line":null}, ...],
        "source": "manual"
      }
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    games: List[GameOdds] = []
    team_map = load_team_name_map()
    for g in payload or []:
        markets = [MarketOdd(**m) for m in g.get("markets", [])]
        home = normalize_team_name(g.get("home_team"), team_map)
        away = normalize_team_name(g.get("away_team"), team_map)
        source = g.get("source") or "manual"
        game_date = g.get("game_date")
        game_time = g.get("game_time")

        # Taiwan Sports Lottery odds are in Asia/Taipei time by default.
        # Convert to US/Eastern to align with MLB schedule dates.
        if source in {"taiwan_sports_lottery", "jbot", "sportslottery"}:
            game_date, game_time = _convert_game_datetime_tz(
                game_date, game_time, from_tz="Asia/Taipei", to_tz="US/Eastern"
            )

        games.append(
            GameOdds(
                game_date=game_date,
                home_team=home,
                away_team=away,
                game_time=game_time,
                markets=markets,
                source=source,
            )
        )
    return games


# -------------------------
# Fetching (Playwright)
# -------------------------

def fetch_page_html(url: str, timeout_ms: int = 30000) -> str:
    """Fetch fully-rendered HTML using Playwright.

    If Playwright is unavailable, raise a clear error.
    """
    if sync_playwright is None:
        raise RuntimeError("Playwright not installed. Run: pip install playwright && playwright install")

    ua = random.choice(USER_AGENTS)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent=ua, viewport={"width": 1280, "height": 720})
        page = context.new_page()

        # Basic anti-bot: block heavy resources to speed up and reduce load
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in {"image", "font", "media"}
            else route.continue_(),
        )

        page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        # Small jitter delay to allow dynamic rendering to finish
        time.sleep(random.uniform(0.8, 1.6))
        html = page.content()

        context.close()
        browser.close()
        return html


# -------------------------
# Parsing helpers
# -------------------------

def parse_decimal_or_american(odd_str: str) -> Optional[int]:
    """Convert odds string to American odds.

    Supports:
    - American format: +120 / -110
    - Decimal format: 1.85 / 2.05
    """
    if not odd_str:
        return None
    s = odd_str.strip()
    # American
    if re.match(r"^[+-]\d+$", s):
        return int(s)
    # Decimal
    try:
        dec = float(s)
        if dec <= 1:
            return None
        if dec >= 2:
            return int(round((dec - 1) * 100))
        return int(round(-100 / (dec - 1)))
    except Exception:
        return None


def parse_line(line_str: str) -> Optional[float]:
    if not line_str:
        return None
    s = re.sub(r"[^0-9.+-]", "", line_str)
    try:
        return float(s)
    except Exception:
        return None


def pd_pu_to_decimal(pd: Optional[str], pu: Optional[str]) -> Optional[float]:
    """Taiwan Sports Lottery odds format: decimal = 1 + pu/pd."""
    if pd is None or pu is None:
        return None
    try:
        pd_v = float(pd)
        pu_v = float(pu)
        if pd_v == 0:
            return None
        return 1.0 + pu_v / pd_v
    except Exception:
        return None


def decimal_to_american(decimal_odds: Optional[float]) -> Optional[int]:
    if decimal_odds is None or decimal_odds <= 1:
        return None
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))


def _detect_market_name(name: str) -> Optional[str]:
    if not name:
        return None
    n = name.lower()
    if any(k in n for k in ["winner", "money line", "不讓分", "獨贏"]):
        return "moneyline"
    if any(k in n for k in ["handicap", "讓分", "run line", "spread"]):
        return "run_line"
    if any(k in n for k in ["total", "over/under", "over under", "大小", "總分"]):
        return "total"
    if any(k in n for k in ["odd/even", "odd even", "單雙"]):
        return "odd_even"
    return None


def _detect_selection(
    market: str,
    selection_code: Optional[str],
    selection_name: Optional[str],
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
) -> Optional[str]:
    code = str(selection_code or "").strip().upper()
    name = str(selection_name or "").strip().lower()
    if market in {"moneyline", "run_line"}:
        if code == "H":
            return "home"
        if code == "A":
            return "away"
        if home_team and home_team.lower() in name:
            return "home"
        if away_team and away_team.lower() in name:
            return "away"
        return None

    if market == "total":
        if code in {"O", "OVER"} or "over" in name or "大" in name:
            return "over"
        if code in {"U", "UNDER"} or "under" in name or "小" in name:
            return "under"
        return None

    if market == "odd_even":
        if code in {"O", "ODD"} or "odd" in name or "單" in name:
            return "odd"
        if code in {"E", "EVEN"} or "even" in name or "雙" in name:
            return "even"
        return None

    return None


# -------------------------
# Taiwan Sports Lottery API (apidata)
# -------------------------

def fetch_pre_games(sport_id: str = BASEBALL_SPORT_ID, lang: str = "en", timeout: int = 20) -> List[Dict]:
    """Fetch pre-game odds list from Taiwan Sports Lottery apidata."""
    url = f"{APIDATA_BASE}/Pre/{sport_id}-Games.{lang}.json"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _resolve_game_date_time(kt: Optional[str], target_date: date) -> Tuple[Optional[str], Optional[str]]:
    """Resolve game date/time from Taiwan lottery timestamp.

    - kt is typically ISO8601 with timezone, e.g. 2026-03-21T01:05:00+08:00
    - Convert to US/Eastern date for MLB schedule matching.
    - If ET date matches target_date, use ET date/time.
    - Else if local (kt) date matches target_date, use local date/time.
    """
    if not kt:
        return None, None
    try:
        dt = datetime.fromisoformat(kt)
    except Exception:
        # fallback: keep raw date if parse fails
        game_date = kt.split("T")[0]
        if game_date != target_date.isoformat():
            return None, None
        game_time = kt.split("T")[1][:5] if "T" in kt else None
        return game_date, game_time

    if dt.tzinfo is not None:
        try:
            dt_et = dt.astimezone(ZoneInfo("US/Eastern"))
        except Exception:
            dt_et = dt
        et_date = dt_et.date()
        if et_date == target_date:
            return et_date.isoformat(), dt_et.strftime("%H:%M")
        return None, None

    # naive datetime
    game_date = dt.date().isoformat()
    if game_date != target_date.isoformat():
        return None, None
    return game_date, dt.strftime("%H:%M")


def parse_pre_games(
    items: List[Dict],
    target_date: date,
    tournament_keywords: Optional[List[str]] = None,
) -> List[GameOdds]:
    """Parse games from apidata format into GameOdds."""
    if tournament_keywords is None:
        tournament_keywords = ["MLB", "Major League", "美國職棒"]

    team_map = load_team_name_map()
    results: List[GameOdds] = []
    for item in items:
        kt = item.get("kt")
        if not kt:
            continue
        game_date, game_time = _resolve_game_date_time(kt, target_date)
        if not game_date:
            continue

        tn = str(item.get("tn") or "").strip()
        if tournament_keywords:
            if not any(k in tn for k in tournament_keywords):
                continue

        away = normalize_team_name(item.get("an") or "", team_map)
        home = normalize_team_name(item.get("hn") or "", team_map)

        markets: List[MarketOdd] = []
        for m in item.get("ms", []) or []:
            m_name = str(m.get("name") or "")
            market = _detect_market_name(m_name)
            if not market:
                continue

            for c in m.get("cs", []) or []:
                selection = _detect_selection(
                    market,
                    c.get("v"),
                    c.get("name"),
                    home_team=home,
                    away_team=away,
                )
                if not selection:
                    continue

                dec = pd_pu_to_decimal(c.get("pd"), c.get("pu"))
                price = decimal_to_american(dec)
                if price is None:
                    continue

                line = parse_line(c.get("hv")) if c.get("hv") is not None else None
                markets.append(MarketOdd(market, selection, price, line))

        if markets:
            results.append(
                GameOdds(
                    game_date=game_date,
                    home_team=home,
                    away_team=away,
                    game_time=game_time,
                    markets=markets,
                    source="taiwan_sports_lottery",
                )
            )

    return results


# -------------------------
# Main parser (TODO: fill selectors)
# -------------------------

def parse_mlb_odds(html: str, target_date: date) -> List[GameOdds]:
    """Parse MLB odds from HTML.

    IMPORTANT: Replace the CSS selectors and DOM traversal below to match
    Taiwan Sports Lottery MLB page structure.
    """
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup not installed. Run: pip install beautifulsoup4 lxml")

    soup = BeautifulSoup(html, "lxml")

    results: List[GameOdds] = []

    # TODO: Update selectors based on real site DOM
    # Example pseudo-structure:
    # for game_el in soup.select(".mlb-game"):
    #     away = game_el.select_one(".team.away .name").get_text(strip=True)
    #     home = game_el.select_one(".team.home .name").get_text(strip=True)
    #     game_time = game_el.select_one(".game-time").get_text(strip=True)
    #
    #     # Moneyline (獨贏)
    #     away_ml = parse_decimal_or_american(game_el.select_one(".moneyline .away .odd").text)
    #     home_ml = parse_decimal_or_american(game_el.select_one(".moneyline .home .odd").text)
    #
    #     # Run line (讓分盤)
    #     away_sp_line = parse_line(game_el.select_one(".spread .away .line").text)
    #     away_sp_price = parse_decimal_or_american(game_el.select_one(".spread .away .odd").text)
    #     home_sp_line = parse_line(game_el.select_one(".spread .home .line").text)
    #     home_sp_price = parse_decimal_or_american(game_el.select_one(".spread .home .odd").text)
    #
    #     # Total (大小分)
    #     total_line = parse_line(game_el.select_one(".total .line").text)
    #     over_price = parse_decimal_or_american(game_el.select_one(".total .over .odd").text)
    #     under_price = parse_decimal_or_american(game_el.select_one(".total .under .odd").text)
    #
    #     markets = [
    #         MarketOdd("moneyline", "away", away_ml),
    #         MarketOdd("moneyline", "home", home_ml),
    #         MarketOdd("run_line", "away", away_sp_price, away_sp_line),
    #         MarketOdd("run_line", "home", home_sp_price, home_sp_line),
    #         MarketOdd("total", "over", over_price, total_line),
    #         MarketOdd("total", "under", under_price, total_line),
    #     ]
    #
    #     results.append(GameOdds(
    #         game_date=target_date.isoformat(),
    #         away_team=away,
    #         home_team=home,
    #         game_time=game_time,
    #         markets=[m for m in markets if m.price is not None],
    #     ))

    return results


# -------------------------
# Output formatting for DB
# -------------------------

def format_for_db(game_odds: List[GameOdds], game_id_lookup: Dict[Tuple[str, str, str], str]) -> List[Dict]:
    """Convert GameOdds into DB rows for odds table.

    game_id_lookup key: (game_date, away_team, home_team) -> game_id (UUID)

    Returns rows compatible with insert_odds in etl_daily.py
    """
    rows: List[Dict] = []
    now_ts = datetime.utcnow().isoformat()
    team_map = load_team_name_map()
    for g in game_odds:
        key = (g.game_date, g.away_team, g.home_team)
        game_id = game_id_lookup.get(key)
        if not game_id:
            norm_away = normalize_team_name(g.away_team, team_map)
            norm_home = normalize_team_name(g.home_team, team_map)
            norm_key = (g.game_date, norm_away, norm_home)
            game_id = game_id_lookup.get(norm_key)
        if not game_id:
            # skip if game_id unknown; caller can decide to insert games first
            continue
        for m in g.markets:
            rows.append({
                "game_id": game_id,
                "sportsbook": g.source,
                "market": m.market,
                "selection": m.selection,
                "price": int(m.price),
                "line": m.line,
                "retrieved_at": now_ts,
            })
    return rows


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--out", help="Output JSON file")
    p.add_argument("--url", default=MLB_URL, help="Target MLB odds URL")
    p.add_argument("--source", choices=["api", "html"], default="api", help="Data source")
    p.add_argument("--lang", default="en", help="Language for apidata (en/zh)")
    p.add_argument("--sport-id", default=BASEBALL_SPORT_ID, help="Sport ID for apidata")
    p.add_argument("--tournament-keyword", action="append", help="Filter tournament names (repeatable)")
    p.add_argument("--all-baseball", action="store_true", help="Do not filter tournament names")
    return p.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else date.today()

    if args.source == "api":
        items = fetch_pre_games(args.sport_id, args.lang)
        if args.all_baseball:
            keywords = []
        else:
            keywords = args.tournament_keyword
        games = parse_pre_games(items, target_date, tournament_keywords=keywords)
    else:
        html = fetch_page_html(args.url)
        games = parse_mlb_odds(html, target_date)

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

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
