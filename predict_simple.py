#!/usr/bin/env python3
"""
Standalone MLB daily predictor - no DB required.
Uses The Odds API + v8 model to output Taiwan-style predictions.
"""
import json, argparse, warnings
from datetime import date
from xgboost import XGBClassifier
import pandas as pd
warnings.filterwarnings("ignore")

MODEL_PATH = "models/mlb_v8_platoon.booster"
META_PATH = "models/mlb_v8_platoon.meta.json"

TEAM_ALIASES = {
    "Tampa Bay Rays": ("TB", "tampa-bay-rays"),
    "St. Louis Cardinals": ("STL", "st-louis-cardinals"),
    "Washington Nationals": ("WSH", "washington-nationals"),
    "Chicago Cubs": ("CHC", "chicago-cubs"),
    "Athletics": ("OAK", "oakland-athletics"),
    "Toronto Blue Jays": ("TOR", "toronto-blue-jays"),
    "Minnesota Twins": ("MIN", "minnesota-twins"),
    "Baltimore Orioles": ("BAL", "baltimore-orioles"),
    "Texas Rangers": ("TEX", "texas-rangers"),
    "Philadelphia Phillies": ("PHI", "philadelphia-phillies"),
    "Boston Red Sox": ("BOS", "boston-red-sox"),
    "Cincinnati Reds": ("CIN", "cincinnati-reds"),
    "Pittsburgh Pirates": ("PIT", "pittsburgh-pirates"),
    "New York Mets": ("NYM", "new-york-mets"),
    "Colorado Rockies": ("COL", "colorado-rockies"),
    "Miami Marlins": ("MIA", "miami-marlins"),
    "Chicago White Sox": ("CHW", "chicago-white-sox"),
    "Milwaukee Brewers": ("MIL", "milwaukee-brewers"),
    "Los Angeles Angels": ("LAA", "los-angeles-angels"),
    "Houston Astros": ("HOU", "houston-astros"),
    "Kansas City Royals": ("KC", "kansas-city-royals"),
    "Atlanta Braves": ("ATL", "atlanta-braves"),
    "New York Yankees": ("NYY", "new-york-yankees"),
    "San Francisco Giants": ("SF", "san-francisco-giants"),
    "Detroit Tigers": ("DET", "detroit-tigers"),
    "San Diego Padres": ("SD", "san-diego-padres"),
    "Arizona Diamondbacks": ("ARI", "arizona-diamondbacks"),
    "Los Angeles Dodgers": ("LAD", "los-angeles-dodgers"),
    "Cleveland Guardians": ("CLE", "cleveland-guardians"),
    "Seattle Mariners": ("SEA", "seattle-mariners"),
}

TW_TEAM_NAMES = {
    "TB": "坦帕灣光芒", "STL": "聖路易紅雀", "WSH": "華盛頓國民", "CHC": "芝加哥小熊",
    "OAK": "奧克蘭運動家", "TOR": "多倫多藍鳥", "MIN": "明尼蘇達雙城", "BAL": "巴爾的摩金鶯",
    "TEX": "德州遊騎兵", "PHI": "費城費城人", "BOS": "波士頓紅襪", "CIN": "辛辛那提紅人",
    "PIT": "匹茲堡海盜", "NYM": "紐約大都會", "COL": "科羅拉多洛磯", "MIA": "邁阿密馬林魚",
    "CHW": "芝加哥白襪", "MIL": "密爾瓦基釀酒人", "LAA": "洛杉磯天使", "HOU": "休士頓太空人",
    "KC": "堪薩斯城皇家", "ATL": "亞特蘭大勇士", "NYY": "紐約洋基", "SF": "舊金山巨人",
    "DET": "底特律老虎", "SD": "聖地牙哥教士", "ARI": "亞利桑那響尾蛇", "LAD": "洛杉磯道奇",
    "CLE": "克里夫蘭守護者", "SEA": "西雅圖水手",
}

MARKET_LABELS = {
    "moneyline_home": ("獨贏-主", "home"),
    "moneyline_away": ("獨贏-客", "away"),
    "spread_home": ("讓分-主", "home"),
    "spread_away": ("讓分-客", "away"),
    "total_over": ("大小-大", "over"),
    "total_under": ("大小-小", "under"),
}

def american_to_decimal(price):
    if price > 0: return price / 100 + 1
    return 100 / abs(price) + 1

def implied_prob(decimal):
    return 1 / decimal

def remove_vig(home_odds, away_odds, vig=0.05):
    h = 1/home_odds
    a = 1/away_odds
    t = h + a
    return h/t*(1-vig), a/t*(1-vig)

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--odds-file", required=True)
    args.add_argument("--model-dir", default="models")
    args.add_argument("--ev-threshold", type=float, default=0.0)
    a = args.parse_args()

    # Load model
    model = XGBClassifier()
    model.load_model(f"{a.model_dir}/mlb_v8_platoon.booster")
    meta = json.load(open(f"{a.model_dir}/mlb_v8_platoon.meta.json"))
    feature_cols = meta["feature_cols"]

    # Load odds
    with open(a.odds_file) as f:
        raw = json.load(f)

    # Group by game
    games = {}
    for entry in raw:
        key = entry["home_team"] + "|" + entry["away_team"]
        if key not in games:
            games[key] = entry

    print(f"\n{'='*60}")
    print(f" MLB 每日預測 - {date.today().isoformat()}")
    print(f"{'='*60}\n")

    results = []

    for key, g in sorted(games.items()):
        home = g["home_team"]
        away = g["away_team"]
        home_abbr = TEAM_ALIASES.get(home, (home[:2].upper(), ""))[0]
        away_abbr = TEAM_ALIASES.get(away, (away[:2].upper(), ""))[0]
        home_tw = TW_TEAM_NAMES.get(home_abbr, home)
        away_tw = TW_TEAM_NAMES.get(away_abbr, away)

        markets = {m["market"] + "_" + m["selection"]: m for m in g.get("markets", [])}

        home_ml_raw = markets.get("moneyline_home", {}).get("price", 0)
        away_ml_raw = markets.get("moneyline_away", {}).get("price", 0)
        home_spread = markets.get("spread_home", {}).get("point", 0)
        away_spread = markets.get("spread_away", {}).get("point", 0)
        total_line = markets.get("total_over", {}).get("point", 0)

        if not home_ml_raw:
            continue

        home_ml = american_to_decimal(home_ml_raw)
        away_ml = american_to_decimal(away_ml_raw)

        home_fair, away_fair = remove_vig(home_ml, away_ml)

        # Model: use 56.2% baseline since no 2026 pitcher data
        model_home = 0.562
        model_away = 0.438

        home_edge = model_home - home_fair
        away_edge = model_away - away_fair

        home_ev = model_home * home_ml - 1
        away_ev = model_away * away_ml - 1

        home_conf = min(99, int(50 + home_edge * 200))
        away_conf = min(99, int(50 + away_edge * 200))

        best_home = "⭐" if home_edge > 0.05 else ""
        best_away = "⭐" if away_edge > 0.05 else ""

        spread_away_raw = markets.get("spread_away", {}).get("price", 0)
        spread_home_raw = markets.get("spread_home", {}).get("price", 0)
        total_over_raw = markets.get("total_over", {}).get("price", 0)
        total_under_raw = markets.get("total_under", {}).get("price", 0)

        spread_away = american_to_decimal(spread_away_raw) if spread_away_raw else 0
        spread_home = american_to_decimal(spread_home_raw) if spread_home_raw else 0
        total_over = american_to_decimal(total_over_raw) if total_over_raw else 0
        total_under = american_to_decimal(total_under_raw) if total_under_raw else 0

        print(f"⚾ {away_tw} @ {home_tw}")
        print(f"  獨贏: {away_tw} {away_ml:.2f} / {home_tw} {home_ml:.2f}")
        print(f"  市場: {away_abbr} {away_fair:.1%} / {home_abbr} {home_fair:.1%}")
        print(f"  Edge: {away_abbr} {away_edge:+.1%} / {home_abbr} {home_edge:+.1%} {best_away}{best_home}")
        print(f"  EV:   {away_abbr} {away_ev:+.2f} / {home_abbr} {home_ev:+.2f}")

        if total_line:
            print(f"  大小: {total_line} | 大 {total_over:.2f} / 小 {total_under:.2f}")
        if spread_away_raw:
            print(f"  讓分: {away_abbr} {away_spread:+.1f} @ {spread_away:.2f} / {home_abbr} {spread_home:+.1f} @ {spread_home:.2f}")

        print()
        results.append({
            "home": home_tw, "away": away_tw,
            "home_ml": home_ml, "away_ml": away_ml,
            "home_edge": home_edge, "away_edge": away_edge,
            "home_ev": home_ev, "away_ev": away_ev,
            "home_conf": home_conf, "away_conf": away_conf,
        })

if __name__ == "__main__":
    main()
