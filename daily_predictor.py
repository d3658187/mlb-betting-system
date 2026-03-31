#!/usr/bin/env python3
"""Daily predictor scaffold for MLB betting system.

Pipeline:
1) (Optional) Refresh today's schedule via MLB Stats API.
2) (Optional) Fetch Taiwan Sports Lottery odds and insert into DB.
3) Build features (feature_builder) + bullpen fatigue.
4) Load trained model and predict home win probability.
5) Compute EV vs Taiwan odds and recommend positive-EV plays.

Usage:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  python daily_predictor.py --date 2026-03-13 --model-dir ./models --out ./data/reco_2026-03-13.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
import joblib
import xgboost as xgb

import feature_builder
import bullpen_fatigue
import mlb_stats_crawler
import taiwan_lottery_crawler

try:
    from discord_notifier import send_discord_recommendations
except Exception:  # pragma: no cover - optional dependency
    send_discord_recommendations = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

PERFORMANCE_TRACKER_COLUMNS = [
    "date",
    "game_id",
    "home_team",
    "away_team",
    "pythagorean_prob",
    "ml_model_prob",
    "market_prob",
    "actual_outcome",
    "correct_pythagorean",
    "correct_ml",
]

OFFLINE_BASE_RATE = 0.410674

STATIC_FALLBACK_ALIASES = {
    "home_starter_era": ["home_starter_era", "home_starter_era_last5", "home_starter_era_last10", "home_starter_era_last3"],
    "away_starter_era": ["away_starter_era", "away_starter_era_last5", "away_starter_era_last10", "away_starter_era_last3"],
    "home_starter_fip": ["home_starter_fip", "home_p_FIP"],
    "away_starter_fip": ["away_starter_fip", "away_p_FIP"],
    "home_starter_xfip": ["home_starter_xfip", "home_p_xFIP"],
    "away_starter_xfip": ["away_starter_xfip", "away_p_xFIP"],
    "home_starter_whip": ["home_starter_whip", "home_starter_whip_last5", "home_starter_whip_last10", "home_p_WHIP"],
    "away_starter_whip": ["away_starter_whip", "away_starter_whip_last5", "away_starter_whip_last10", "away_p_WHIP"],
    "home_bat_OPS": ["home_bat_OPS", "home_fangraphs_ops", "home_fangraphs_ops_plus"],
    "away_bat_OPS": ["away_bat_OPS", "away_fangraphs_ops", "away_fangraphs_ops_plus"],
    "home_bat_wOBA": ["home_bat_wOBA", "home_fangraphs_woba"],
    "away_bat_wOBA": ["away_bat_wOBA", "away_fangraphs_woba"],
    "home_platoon_ba_diff": ["home_platoon_ba_diff"],
    "away_platoon_ba_diff": ["away_platoon_ba_diff"],
    "home_platoon_ops_diff": ["home_platoon_ops_diff"],
    "away_platoon_ops_diff": ["away_platoon_ops_diff"],
    "home_platoon_splits_score": ["home_platoon_splits_score"],
    "away_platoon_splits_score": ["away_platoon_splits_score"],
    "home_rest_days": ["home_rest_days"],
    "away_rest_days": ["away_rest_days"],
    "home_advantage": ["home_advantage"],
}


# ---------------------
# DB helpers
# ---------------------

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    return create_engine(db_url, pool_pre_ping=True)


def load_games_with_teams(engine, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT g.id as game_id,
               g.mlb_game_id,
               g.game_date,
               g.game_datetime,
               g.status,
               g.home_team_id,
               g.away_team_id,
               th.name as home_team_name,
               ta.name as away_team_name,
               th.mlb_team_id as home_mlb_id,
               ta.mlb_team_id as away_mlb_id
          FROM games g
          JOIN teams th ON g.home_team_id = th.id
          JOIN teams ta ON g.away_team_id = ta.id
         WHERE g.game_date = :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"target_date": target_date})


def load_team_id_map(engine) -> Dict[int, str]:
    sql = text("SELECT id, mlb_team_id FROM teams")
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().all()
    return {r["mlb_team_id"]: r["id"] for r in rows}


# ---------------------
# Odds helpers
# ---------------------

def american_to_decimal(odds: int) -> float:
    if odds == 0:
        return 0.0
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def implied_prob_from_american(odds: int) -> float:
    if odds == 0:
        return 0.0
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def compute_ev(win_prob: float, odds: int) -> float:
    """Expected value per 1 unit stake."""
    dec = american_to_decimal(odds)
    return win_prob * dec - 1.0


def devig_prob(implied_prob: float, vig_rate: float) -> float:
    """Remove vigorish (10-15%) from implied probability."""
    if implied_prob is None:
        return None
    return implied_prob / (1.0 + vig_rate)


def confidence_from_edge(edge: float) -> float:
    """Confidence index shrunk into 55-75 using edge."""
    if edge is None:
        return None
    # tanh shrink: edge=0 -> 65, large edge -> 75, negative -> 55
    return 65.0 + 10.0 * math.tanh(edge * 5.0)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def poisson_cdf(k: int, lam: float) -> float:
    if lam is None or lam < 0:
        return None
    k = int(k)
    if k < 0:
        return 0.0
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(1.0, max(0.0, total))


def load_odds(engine, target_date: date, market: Optional[str] = None) -> pd.DataFrame:
    if market:
        sql = text(
            """
            SELECT o.game_id, o.market, o.selection, o.price, o.line, o.retrieved_at
              FROM odds o
              JOIN games g ON o.game_id = g.id
             WHERE g.game_date = :target_date
               AND o.market = :market
            """
        )
        return pd.read_sql(sql, engine, params={"target_date": target_date, "market": market})

    sql = text(
        """
        SELECT o.game_id, o.market, o.selection, o.price, o.line, o.retrieved_at
          FROM odds o
          JOIN games g ON o.game_id = g.id
         WHERE g.game_date = :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"target_date": target_date})


def load_moneyline_odds(engine, target_date: date) -> pd.DataFrame:
    return load_odds(engine, target_date, market="moneyline")


def insert_odds_rows(engine, rows: List[Dict]):
    if not rows:
        return
    sql = text(
        """
        INSERT INTO odds (game_id, sportsbook, market, selection, price, line, retrieved_at)
        VALUES (:game_id, :sportsbook, :market, :selection, :price, :line, :retrieved_at)
        ON CONFLICT DO NOTHING
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, rows)


def build_game_id_lookup(games_df: pd.DataFrame) -> Dict[Tuple[str, str, str], str]:
    team_map = taiwan_lottery_crawler.load_team_name_map()
    lookup = {}
    for _, row in games_df.iterrows():
        away = taiwan_lottery_crawler.normalize_team_name(row["away_team_name"], team_map)
        home = taiwan_lottery_crawler.normalize_team_name(row["home_team_name"], team_map)
        key = (row["game_date"].isoformat(), away, home)
        lookup[key] = row["game_id"]
    return lookup


def load_manual_odds_json_to_df(path: str, games_df: pd.DataFrame) -> pd.DataFrame:
    games = taiwan_lottery_crawler.load_manual_odds_json(path)
    lookup = build_game_id_lookup(games_df)
    rows = taiwan_lottery_crawler.format_for_db(games, lookup)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ---------------------
# Odds-only fallback helpers
# ---------------------

TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


def _normalize_odds_game_date(raw_date: Optional[str], target_date: date) -> date:
    if not raw_date:
        return target_date
    try:
        parsed = date.fromisoformat(str(raw_date))
    except Exception:
        return target_date
    if abs((parsed - target_date).days) <= 1:
        return target_date
    return parsed


def _build_synthetic_game_id(game_date: date, away: str, home: str) -> str:
    safe_away = re.sub(r"\s+", "_", str(away).strip())
    safe_home = re.sub(r"\s+", "_", str(home).strip())
    return f"{game_date.isoformat()}_{safe_away}_@_{safe_home}".lower()


def _resolve_team_abbr(name: Optional[str]) -> Optional[str]:
    team_map = taiwan_lottery_crawler.load_team_name_map()
    norm = taiwan_lottery_crawler.normalize_team_name(name, team_map)
    if norm in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[norm]
    upper = norm.upper()
    if upper in TEAM_NAME_TO_ABBR.values():
        return upper
    return None


def build_odds_df_from_games(odds_games: List[taiwan_lottery_crawler.GameOdds], target_date: date) -> pd.DataFrame:
    if not odds_games:
        return pd.DataFrame()
    rows = []
    now_ts = pd.Timestamp.utcnow().isoformat()
    team_map = taiwan_lottery_crawler.load_team_name_map()
    for g in odds_games:
        game_date = _normalize_odds_game_date(g.game_date, target_date)
        away = taiwan_lottery_crawler.normalize_team_name(g.away_team, team_map)
        home = taiwan_lottery_crawler.normalize_team_name(g.home_team, team_map)
        game_id = _build_synthetic_game_id(game_date, away, home)
        for m in g.markets:
            rows.append({
                "game_id": game_id,
                "market": m.market,
                "selection": m.selection,
                "price": int(m.price),
                "line": m.line,
                "retrieved_at": now_ts,
            })
    return pd.DataFrame(rows)


def build_games_df_from_odds(odds_games: List[taiwan_lottery_crawler.GameOdds], target_date: date) -> pd.DataFrame:
    if not odds_games:
        return pd.DataFrame()
    rows = []
    team_map = taiwan_lottery_crawler.load_team_name_map()
    for g in odds_games:
        game_date = _normalize_odds_game_date(g.game_date, target_date)
        away = taiwan_lottery_crawler.normalize_team_name(g.away_team, team_map)
        home = taiwan_lottery_crawler.normalize_team_name(g.home_team, team_map)
        rows.append({
            "game_id": _build_synthetic_game_id(game_date, away, home),
            "mlb_game_id": None,
            "game_date": game_date,
            "home_team_name": home,
            "away_team_name": away,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["game_id"]).reset_index(drop=True)


def _select_team_stats_file(data_dir: str, kind: str, season: int) -> Optional[Path]:
    direct = Path(data_dir) / f"team_{kind}_{season}.csv"
    if direct.exists():
        return direct
    candidates = list(Path(data_dir).glob(f"team_{kind}_*.csv"))
    if not candidates:
        return None

    def _extract_year(path: Path) -> int:
        match = re.search(r"team_%s_(\d{4})(?:-(\d{4}))?\.csv" % kind, path.name)
        if not match:
            return 0
        return int(match.group(2) or match.group(1))

    return max(candidates, key=_extract_year)


def _load_team_stats(data_dir: str, kind: str, season: int) -> pd.DataFrame:
    path = _select_team_stats_file(data_dir, kind, season)
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_basic_team_model_features(
    games_df: pd.DataFrame,
    season: int,
    data_dir: str = "./data/pybaseball",
    force_synthetic_ids: bool = False,
) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    batting = _load_team_stats(data_dir, "batting", season)
    pitching = _load_team_stats(data_dir, "pitching", season)
    if batting.empty or pitching.empty:
        return pd.DataFrame()

    for col in ["R", "G"]:
        if col in batting.columns:
            batting[col] = pd.to_numeric(batting[col], errors="coerce")
        if col in pitching.columns:
            pitching[col] = pd.to_numeric(pitching[col], errors="coerce")

    batting["runs_pg"] = batting.get("R") / batting.get("G").replace(0, pd.NA)
    pitching["runs_allowed_pg"] = pitching.get("R") / pitching.get("G").replace(0, pd.NA)

    league_bat = batting[batting.get("team") == "- - -"]
    league_runs_pg = pd.NA
    if not league_bat.empty:
        league_runs_pg = league_bat["runs_pg"].iloc[0]
    if pd.isna(league_runs_pg):
        league_runs_pg = batting["runs_pg"].mean()
    league_runs_pg = float(league_runs_pg) if pd.notna(league_runs_pg) else 4.5

    bat_map = batting.dropna(subset=["team"]).set_index("team")["runs_pg"].to_dict()
    pit_map = pitching.dropna(subset=["team"]).set_index("team")["runs_allowed_pg"].to_dict()

    features = []
    for _, row in games_df.iterrows():
        home_name = row.get("home_team_name")
        away_name = row.get("away_team_name")
        home_abbr = _resolve_team_abbr(home_name)
        away_abbr = _resolve_team_abbr(away_name)

        home_rpg = bat_map.get(home_abbr, league_runs_pg)
        away_rpg = bat_map.get(away_abbr, league_runs_pg)
        home_ra_pg = pit_map.get(home_abbr, league_runs_pg)
        away_ra_pg = pit_map.get(away_abbr, league_runs_pg)

        try:
            home_expected = float(home_rpg) * float(away_ra_pg) / league_runs_pg
        except Exception:
            home_expected = float(home_rpg) if pd.notna(home_rpg) else league_runs_pg
        try:
            away_expected = float(away_rpg) * float(home_ra_pg) / league_runs_pg
        except Exception:
            away_expected = float(away_rpg) if pd.notna(away_rpg) else league_runs_pg

        denom = home_expected ** 2 + away_expected ** 2
        if denom > 0:
            home_win_prob = (home_expected ** 2) / denom
        else:
            home_win_prob = 0.5

        run_margin = home_expected - away_expected
        expected_total = home_expected + away_expected

        game_date = row.get("game_date")
        if isinstance(game_date, str):
            try:
                game_date = date.fromisoformat(game_date)
            except Exception:
                game_date = None

        game_id = row.get("game_id")
        if force_synthetic_ids or not game_id:
            if not isinstance(game_date, date):
                game_date = date.today()
            game_id = _build_synthetic_game_id(game_date, away_name, home_name)

        features.append({
            "game_id": game_id,
            "mlb_game_id": row.get("mlb_game_id"),
            "game_date": game_date,
            "home_team_name": home_name,
            "away_team_name": away_name,
            "home_win_prob": home_win_prob,
            "run_margin_pred": run_margin,
            "expected_total_runs": expected_total,
        })

    return pd.DataFrame(features)


def select_offline_features_csv(target_date: date, explicit_path: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.append(Path("data") / f"features_{target_date.isoformat()}.csv")
    candidates.append(Path("data/features_2026-03-20.csv"))

    for path in sorted(Path("data").glob("features_*.csv"), reverse=True):
        candidates.append(path)

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and path.is_file():
            return path
    return None


def build_offline_feature_template(
    games_df: pd.DataFrame,
    feature_cols: Optional[List[str]],
    template_csv_path: Optional[Path],
) -> Tuple[pd.DataFrame, bool, Optional[Path]]:
    if games_df is None or games_df.empty:
        return pd.DataFrame(), False, None

    if not feature_cols:
        # Without feature columns from model meta, prediction is not stable.
        return pd.DataFrame(), False, template_csv_path

    template_df = pd.DataFrame()
    used_template = False
    if template_csv_path and template_csv_path.exists():
        try:
            template_df = pd.read_csv(template_csv_path)
            used_template = not template_df.empty
        except Exception as exc:
            logging.warning("Failed to read offline features CSV %s: %s", template_csv_path, exc)
            template_df = pd.DataFrame()
            used_template = False

    numeric_template = template_df.apply(pd.to_numeric, errors="coerce") if not template_df.empty else pd.DataFrame()

    features = pd.DataFrame(index=games_df.index)
    for col in feature_cols:
        if not numeric_template.empty and col in numeric_template.columns:
            median_val = numeric_template[col].median(skipna=True)
            if pd.notna(median_val):
                features[col] = float(median_val)
                continue
        features[col] = 0.0

    features["game_id"] = games_df["game_id"].astype(str).values
    if "game_date" in games_df.columns:
        features["game_date"] = games_df["game_date"].values
    if "home_team_name" in games_df.columns:
        features["home_team_name"] = games_df["home_team_name"].values
    if "away_team_name" in games_df.columns:
        features["away_team_name"] = games_df["away_team_name"].values

    if "home_advantage" in features.columns:
        features["home_advantage"] = 1.0

    return features, used_template, template_csv_path


def build_latest_moneyline_prices(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df is None or odds_df.empty:
        return pd.DataFrame(columns=["game_id", "home_price", "away_price"])

    odds = odds_df.copy()
    odds = odds[odds["market"].astype(str).str.lower() == "moneyline"].copy()
    if odds.empty:
        return pd.DataFrame(columns=["game_id", "home_price", "away_price"])

    odds["selection"] = odds["selection"].astype(str).str.lower()
    odds = odds[odds["selection"].isin(["home", "away"])].copy()
    if odds.empty:
        return pd.DataFrame(columns=["game_id", "home_price", "away_price"])

    if "retrieved_at" in odds.columns:
        odds = odds.sort_values("retrieved_at")
    latest = odds.groupby(["game_id", "selection"], as_index=False).last()
    pivot = latest.pivot(index="game_id", columns="selection", values="price").reset_index()
    pivot = pivot.rename(columns={"home": "home_price", "away": "away_price"})
    for col in ["home_price", "away_price"]:
        if col not in pivot.columns:
            pivot[col] = pd.NA
    return pivot[["game_id", "home_price", "away_price"]]


def run_offline_prediction_mode(
    target_date: date,
    model_dir: str,
    model_name: str,
    odds_json_path: Path,
    output_path: Path,
    feature_template_path: Optional[str],
    base_rate: float = OFFLINE_BASE_RATE,
    max_games: Optional[int] = None,
) -> pd.DataFrame:
    if not odds_json_path.exists():
        logging.warning("Offline odds JSON not found: %s", odds_json_path)
        return pd.DataFrame()

    odds_games = taiwan_lottery_crawler.load_manual_odds_json(str(odds_json_path))
    games_df = build_games_df_from_odds(odds_games, target_date)
    if games_df.empty:
        logging.warning("Offline odds JSON loaded but no games available: %s", odds_json_path)
        return pd.DataFrame()

    games_df = games_df[games_df["game_date"].apply(lambda d: str(d) == target_date.isoformat())].copy()
    if games_df.empty:
        logging.warning("No games matched target date %s in %s", target_date, odds_json_path)
        return pd.DataFrame()

    games_df = games_df.sort_values(["away_team_name", "home_team_name"]).drop_duplicates(subset=["game_id"])

    if max_games and max_games > 0 and len(games_df) > max_games:
        logging.info("Offline mode trimming games from %d to %d", len(games_df), max_games)
        games_df = games_df.head(max_games).copy()

    odds_df = build_odds_df_from_games(odds_games, target_date)
    odds_df = odds_df[odds_df["game_id"].isin(games_df["game_id"])].copy()

    try:
        model, feature_cols, _ = load_model_and_meta(model_dir, model_name)
    except Exception as exc:
        logging.warning("Failed to load model '%s' in offline mode: %s", model_name, exc)
        model, feature_cols = None, None

    template_path = select_offline_features_csv(target_date, explicit_path=feature_template_path)
    features, used_template, used_path = build_offline_feature_template(games_df, feature_cols, template_path)

    if model is None or features.empty:
        logging.warning(
            "Offline mode using base rate %.6f for all games (model unavailable or features unavailable)",
            base_rate,
        )
        home_probs = pd.Series(base_rate, index=games_df.index)
        model_source = "base_rate"
    else:
        try:
            home_probs = predict_win_prob(model, features, feature_cols)
            if home_probs.empty:
                raise RuntimeError("empty prediction result")
            model_source = f"template:{used_path.name}" if used_template and used_path else "template:zeros"
        except Exception as exc:
            logging.warning("Offline prediction failed, fallback to base rate %.6f: %s", base_rate, exc)
            home_probs = pd.Series(base_rate, index=games_df.index)
            model_source = "base_rate"

    games_out = games_df.copy().reset_index(drop=True)
    home_probs = pd.to_numeric(home_probs, errors="coerce")
    home_probs = home_probs.reset_index(drop=True).reindex(games_out.index)
    games_out["home_win_prob"] = home_probs.fillna(base_rate).clip(0.001, 0.999)
    games_out["away_win_prob"] = 1.0 - games_out["home_win_prob"]

    market_prob = build_moneyline_market_prob(games_out, odds_df)
    games_out["market_home_prob"] = games_out["game_id"].astype(str).map(market_prob)

    price_df = build_latest_moneyline_prices(odds_df)
    price_df["game_id"] = price_df["game_id"].astype(str)
    games_out["game_id"] = games_out["game_id"].astype(str)
    games_out = games_out.merge(price_df, on="game_id", how="left")

    games_out["predicted_winner"] = np.where(
        games_out["home_win_prob"] >= 0.5,
        games_out["home_team_name"],
        games_out["away_team_name"],
    )
    games_out["model_source"] = model_source
    games_out["prediction_date"] = target_date.isoformat()

    out_cols = [
        "prediction_date",
        "game_id",
        "away_team_name",
        "home_team_name",
        "home_win_prob",
        "away_win_prob",
        "market_home_prob",
        "home_price",
        "away_price",
        "predicted_winner",
        "model_source",
    ]
    predictions = games_out[out_cols].rename(
        columns={
            "away_team_name": "away_team",
            "home_team_name": "home_team",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logging.info("Offline predictions written: %s (%d games)", output_path, len(predictions))

    return predictions


# ---------------------
# Refresh helpers
# ---------------------

def refresh_schedule(target_date: date):
    """Pull MLB schedule + stats into DB using mlb_stats_crawler."""
    logging.info("Refreshing MLB schedule + stats for %s", target_date)
    mlb_stats_crawler.run(target_date)


def refresh_taiwan_odds(engine, target_date: date, url: Optional[str] = None):
    """Fetch Taiwan Sports Lottery odds and insert into DB.

    Priority:
    1) Manual JSON file if url points to a local .json
    2) JBot API if JBOT_TOKEN/SPORTSBOT_TOKEN/X_JBOT_TOKEN is set
    3) Playwright crawler (scaffold; requires selectors)
    """
    games = []

    # 1) Manual JSON
    if url and os.path.exists(url) and url.lower().endswith(".json"):
        logging.info("Loading manual odds from %s", url)
        games = taiwan_lottery_crawler.load_manual_odds_json(url)
    else:
        # 2) Taiwan Sports Lottery apidata (no auth)
        try:
            items = taiwan_lottery_crawler.fetch_pre_games(
                taiwan_lottery_crawler.BASEBALL_SPORT_ID,
                lang=os.getenv("TW_LOTTERY_LANG", "en"),
            )
            games = taiwan_lottery_crawler.parse_pre_games(items, target_date)
            logging.info("Fetched %d games from Taiwan apidata", len(games))
        except Exception as exc:
            logging.warning("Taiwan apidata failed: %s", exc)

        # 3) JBot API (requires token)
        if not games:
            token = os.getenv("JBOT_TOKEN") or os.getenv("SPORTSBOT_TOKEN") or os.getenv("X_JBOT_TOKEN")
            if token:
                try:
                    payload = taiwan_lottery_crawler.fetch_jbot_odds(target_date, token=token)
                    games = taiwan_lottery_crawler.parse_jbot_odds(payload, target_date)
                    logging.info("Fetched %d games from JBot API", len(games))
                except Exception as exc:
                    logging.warning("JBot API failed: %s", exc)

        # 4) Fallback crawler
        if not games:
            fetch_url = url or taiwan_lottery_crawler.MLB_URL
            logging.info("Fetching Taiwan odds from %s", fetch_url)
            html = taiwan_lottery_crawler.fetch_page_html(fetch_url)
            games = taiwan_lottery_crawler.parse_mlb_odds(html, target_date)

    if not games:
        logging.warning("No odds parsed. Check API token, manual input, or crawler selectors.")
        return

    # Build game_id lookup using team names
    games_df = load_games_with_teams(engine, target_date)
    lookup = {}
    team_map = taiwan_lottery_crawler.load_team_name_map()
    for _, row in games_df.iterrows():
        away = taiwan_lottery_crawler.normalize_team_name(row["away_team_name"], team_map)
        home = taiwan_lottery_crawler.normalize_team_name(row["home_team_name"], team_map)
        key = (row["game_date"].isoformat(), away, home)
        lookup[key] = row["game_id"]

    rows = taiwan_lottery_crawler.format_for_db(games, lookup)
    if not rows:
        # Fallback: try date-shifted schedules (Taiwan odds can be off by timezone)
        for offset in (-1, 1):
            shifted_date = target_date + timedelta(days=offset)
            shifted_df = load_games_with_teams(engine, shifted_date)
            if shifted_df.empty:
                continue
            lookup_shifted = {}
            for _, row in shifted_df.iterrows():
                away = taiwan_lottery_crawler.normalize_team_name(row["away_team_name"], team_map)
                home = taiwan_lottery_crawler.normalize_team_name(row["home_team_name"], team_map)
                key = (target_date.isoformat(), away, home)
                lookup_shifted[key] = row["game_id"]
            rows = taiwan_lottery_crawler.format_for_db(games, lookup_shifted)
            if rows:
                logging.warning("Odds matched using schedule date %s (shift %s day).", shifted_date, offset)
                break

    if not rows:
        logging.warning("No odds rows matched to games. Check team name mapping.")
        return

    insert_odds_rows(engine, rows)
    logging.info("Inserted %d odds rows", len(rows))


# ---------------------
# Predictor
# ---------------------

def build_feature_table(target_date: date, window: int) -> pd.DataFrame:
    # feature_builder expects `windows` (sequence). Keep CLI using single window value.
    features = feature_builder.build_features(target_date, windows=(window,))
    if features.empty:
        return features

    # bullpen fatigue (team_mlb_id -> teams.id)
    engine = get_engine()
    team_id_map = load_team_id_map(engine)
    team_fatigue, _ = bullpen_fatigue.compute_team_fatigue(engine, target_date)

    if not team_fatigue.empty:
        team_fatigue = team_fatigue.copy()
        team_fatigue["team_id"] = team_fatigue["team_mlb_id"].map(team_id_map)
        # Ensure merge keys share dtype
        try:
            features["home_team_id"] = features["home_team_id"].astype(str)
            features["away_team_id"] = features["away_team_id"].astype(str)
            team_fatigue["team_id"] = team_fatigue["team_id"].astype(str)
        except Exception:
            pass
        fatigue_cols = [
            "bullpen_fatigue_index",
            "bullpen_pitch_count",
            "bullpen_appearance_days",
            "bullpen_pitcher_count",
            "bullpen_avg_rest_days",
        ]

        home_fatigue = team_fatigue[["team_id"] + fatigue_cols].rename(
            columns={c: f"home_{c}" for c in fatigue_cols}
        )
        away_fatigue = team_fatigue[["team_id"] + fatigue_cols].rename(
            columns={c: f"away_{c}" for c in fatigue_cols}
        )

        features = features.merge(
            home_fatigue,
            how="left",
            left_on="home_team_id",
            right_on="team_id",
        ).drop(columns=["team_id"], errors="ignore")

        features = features.merge(
            away_fatigue,
            how="left",
            left_on="away_team_id",
            right_on="team_id",
        ).drop(columns=["team_id"], errors="ignore")

    return features


def load_model_and_meta(model_dir: str, model_name: str):
    """Loads a classification model and its metadata."""
    model_path_pkl = Path(model_dir) / f"{model_name}.pkl"
    model_path_booster = Path(model_dir) / f"{model_name}.booster"
    model_path_json = Path(model_dir) / f"{model_name}.json"
    meta_path = Path(model_dir) / f"{model_name}.meta.json"

    model = None
    if model_path_pkl.exists():
        model = joblib.load(model_path_pkl)
    elif model_path_json.exists():
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_path_json)
        except Exception as e:
            logging.error(f"Failed to load XGBoost .json model '{model_path_json}': {e}")
            raise e
    elif model_path_booster.exists():
        try:
            model = xgb.XGBClassifier()
            model.load_model(model_path_booster)
        except Exception as e:
            logging.error(f"Failed to load XGBoost .booster model '{model_path_booster}': {e}")
            raise e
    else:
        raise FileNotFoundError(f"Model not found: {model_path_pkl} or {model_path_booster} or {model_path_json}")

    # Compatibility shim for older XGBoost pickles
    try:
        if hasattr(model, "get_params"):
            for attr, default in {
                "use_label_encoder": False,
                "gpu_id": -1,
                "predictor": "auto",
            }.items():
                if not hasattr(model, attr):
                    try:
                        setattr(model, attr, default)
                    except Exception:
                        pass
    except Exception:
        pass
    feature_cols = None
    n_samples = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        feature_cols = meta.get("feature_cols")
        # n_samples may be at top level or inside metrics.n_train
        n_samples = meta.get("n_samples") or meta.get("metrics", {}).get("n_train")
    return model, feature_cols, n_samples


def predict_win_prob(model, features: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.Series:
    if model is None or features.empty:
        return pd.Series(dtype=float)
    if feature_cols is None:
        feature_cols = [
            c
            for c in features.columns
            if c
            not in {"mlb_game_id", "game_id", "game_date", "game_datetime", "status"}
            and not c.endswith("_id")
        ]
    # Ensure all required feature columns exist; fill missing with 0
    X = features.reindex(columns=feature_cols, fill_value=0)
    # Convert any object-dtype columns to numeric (e.g., platoon splits loaded as strings)
    X = X.apply(pd.to_numeric, errors="coerce").infer_objects(copy=False).fillna(0)
    # Handle both sklearn wrapper (predict_proba) and raw Booster (predict with sigmoid)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        raw = model.predict(X)
        # Sigmoid the raw margin output to get probability
        probs = 1 / (1 + np.exp(-raw))
    return pd.Series(probs, index=features.index)


def _coalesce_numeric(df: pd.DataFrame, candidates: List[str], divide_by_100: bool = False) -> pd.Series:
    series = pd.Series(np.nan, index=df.index, dtype=float)
    for col in candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if divide_by_100 and "ops_plus" in col.lower():
                values = values / 100.0
            series = series.combine_first(values)
    return series


def build_static_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    static_df = pd.DataFrame(index=features.index)

    for canonical, aliases in STATIC_FALLBACK_ALIASES.items():
        divide = canonical.endswith("_OPS")
        static_df[canonical] = _coalesce_numeric(features, aliases, divide_by_100=divide)

    # Derived diffs when absent
    if "diff_starter_era" not in static_df.columns:
        static_df["diff_starter_era"] = static_df["home_starter_era"] - static_df["away_starter_era"]
    if "diff_starter_fip" not in static_df.columns:
        static_df["diff_starter_fip"] = static_df["home_starter_fip"] - static_df["away_starter_fip"]
    if "diff_starter_xfip" not in static_df.columns:
        static_df["diff_starter_xfip"] = static_df["home_starter_xfip"] - static_df["away_starter_xfip"]
    if "diff_starter_whip" not in static_df.columns:
        static_df["diff_starter_whip"] = static_df["home_starter_whip"] - static_df["away_starter_whip"]
    if "diff_bat_OPS" not in static_df.columns:
        static_df["diff_bat_OPS"] = static_df["home_bat_OPS"] - static_df["away_bat_OPS"]
    if "diff_bat_wOBA" not in static_df.columns:
        static_df["diff_bat_wOBA"] = static_df["home_bat_wOBA"] - static_df["away_bat_wOBA"]
    if "diff_platoon_ba_diff" not in static_df.columns:
        static_df["diff_platoon_ba_diff"] = static_df["home_platoon_ba_diff"] - static_df["away_platoon_ba_diff"]
    if "diff_platoon_ops_diff" not in static_df.columns:
        static_df["diff_platoon_ops_diff"] = static_df["home_platoon_ops_diff"] - static_df["away_platoon_ops_diff"]
    if "diff_platoon_splits_score" not in static_df.columns:
        static_df["diff_platoon_splits_score"] = static_df["home_platoon_splits_score"] - static_df["away_platoon_splits_score"]
    if "diff_rest_days" not in static_df.columns:
        static_df["diff_rest_days"] = static_df["home_rest_days"] - static_df["away_rest_days"]

    if "home_advantage" not in static_df.columns:
        static_df["home_advantage"] = 1.0
    else:
        static_df["home_advantage"] = static_df["home_advantage"].fillna(1.0)

    return static_df


def apply_static_fallback(
    features: pd.DataFrame,
    base_probs: pd.Series,
    model_dir: str,
    base_feature_cols: Optional[List[str]],
    zero_threshold: float = 0.5,
    static_model_name: str = "static_model",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if features.empty or base_probs.empty:
        empty = pd.Series(dtype=float)
        return base_probs, empty, empty

    reference_cols = base_feature_cols or list(features.columns)
    rolling_cols = [c for c in reference_cols if ("roll" in c.lower() or "ewm" in c.lower())]
    if not rolling_cols:
        rolling_zero_ratio = pd.Series(0.0, index=features.index)
        cold_mask = pd.Series(False, index=features.index)
        return base_probs, cold_mask, rolling_zero_ratio

    rolling_data = features.reindex(columns=rolling_cols, fill_value=0)
    rolling_data = rolling_data.apply(pd.to_numeric, errors="coerce").fillna(0)
    rolling_zero_ratio = (rolling_data == 0).mean(axis=1)
    cold_mask = rolling_zero_ratio > zero_threshold

    if not bool(cold_mask.any()):
        return base_probs, cold_mask, rolling_zero_ratio

    model_path = Path(model_dir) / f"{static_model_name}.joblib"
    meta_path = Path(model_dir) / f"{static_model_name}.meta.json"

    if not model_path.exists():
        logging.warning("Static fallback model not found: %s", model_path)
        return base_probs, cold_mask, rolling_zero_ratio

    static_model = joblib.load(model_path)
    static_feature_cols = None
    if meta_path.exists():
        try:
            static_feature_cols = json.loads(meta_path.read_text()).get("feature_cols")
        except Exception as exc:
            logging.warning("Failed to parse static model meta %s: %s", meta_path, exc)

    static_features = build_static_feature_frame(features)
    if static_feature_cols is None:
        static_feature_cols = list(static_features.columns)

    X_static = static_features.reindex(columns=static_feature_cols, fill_value=0)
    X_static = X_static.apply(pd.to_numeric, errors="coerce").fillna(0)

    if hasattr(static_model, "predict_proba"):
        static_probs = static_model.predict_proba(X_static)[:, 1]
    else:
        raw = static_model.predict(X_static)
        static_probs = 1 / (1 + np.exp(-raw))

    blended_probs = base_probs.copy()
    blended_probs.loc[cold_mask] = pd.Series(static_probs, index=features.index).loc[cold_mask]

    logging.info(
        "Applied static fallback model to %d/%d games (rolling_zero_ratio > %.2f)",
        int(cold_mask.sum()),
        int(len(cold_mask)),
        zero_threshold,
    )
    return blended_probs, cold_mask, rolling_zero_ratio


def build_moneyline_market_prob(games_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.Series:
    if games_df is None or games_df.empty or odds_df is None or odds_df.empty:
        return pd.Series(dtype=float)

    odds = odds_df.copy()
    if "market" in odds.columns:
        odds = odds[odds["market"].astype(str).str.lower() == "moneyline"].copy()
    if odds.empty:
        return pd.Series(dtype=float)

    games = games_df[["game_id", "home_team_name", "away_team_name"]].drop_duplicates().copy()
    games["game_id"] = games["game_id"].astype(str)
    odds["game_id"] = odds["game_id"].astype(str)

    team_map = taiwan_lottery_crawler.load_team_name_map()
    name_map: Dict[Tuple[str, str], str] = {}
    for _, row in games.iterrows():
        home_name = taiwan_lottery_crawler.normalize_team_name(row["home_team_name"], team_map).lower()
        away_name = taiwan_lottery_crawler.normalize_team_name(row["away_team_name"], team_map).lower()
        name_map[(row["game_id"], home_name)] = "home"
        name_map[(row["game_id"], away_name)] = "away"

    odds["selection_norm"] = odds.apply(
        lambda r: _normalize_selection_for_market(r.get("selection"), "moneyline", r.get("game_id"), name_map),
        axis=1,
    )
    odds = odds[odds["selection_norm"].isin(["home", "away"])].copy()
    if odds.empty:
        return pd.Series(dtype=float)

    odds["implied_prob"] = pd.to_numeric(odds["price"], errors="coerce").apply(
        lambda v: implied_prob_from_american(int(v)) if pd.notna(v) else np.nan
    )

    latest = odds.sort_values("retrieved_at").groupby(["game_id", "selection_norm"], as_index=False).last()
    pivot = latest.pivot(index="game_id", columns="selection_norm", values="implied_prob")
    if pivot.empty:
        return pd.Series(dtype=float)

    home = pivot.get("home")
    away = pivot.get("away")
    if home is None:
        return pd.Series(dtype=float)

    if away is None:
        return home.rename("market_prob")

    denom = home + away
    market_prob = (home / denom).where(denom > 0, home)
    return market_prob.rename("market_prob")


def update_performance_tracker(
    target_date: date,
    games_df: pd.DataFrame,
    features: pd.DataFrame,
    odds_df: pd.DataFrame,
    tracker_path: Path,
):
    if games_df is None or games_df.empty or features is None or features.empty:
        return

    game_cols = ["game_id", "home_team_name", "away_team_name"]
    game_info = games_df[game_cols].drop_duplicates().copy()
    game_info["game_id"] = game_info["game_id"].astype(str)

    feature_probs = features[["game_id", "home_win_prob"]].copy()
    feature_probs["game_id"] = feature_probs["game_id"].astype(str)
    feature_probs = feature_probs.rename(columns={"home_win_prob": "ml_model_prob"})

    pythag = build_basic_team_model_features(games_df, target_date.year)
    if pythag.empty:
        pythag_probs = pd.Series(np.nan, index=game_info["game_id"].values)
    else:
        pythag["game_id"] = pythag["game_id"].astype(str)
        pythag_probs = pythag.set_index("game_id")["home_win_prob"]

    market_probs = build_moneyline_market_prob(games_df, odds_df)

    tracker_rows = game_info.merge(feature_probs, on="game_id", how="left")
    tracker_rows["date"] = target_date.isoformat()
    tracker_rows["pythagorean_prob"] = tracker_rows["game_id"].map(pythag_probs)
    tracker_rows["market_prob"] = tracker_rows["game_id"].map(market_probs)
    tracker_rows["actual_outcome"] = pd.NA
    tracker_rows["correct_pythagorean"] = pd.NA
    tracker_rows["correct_ml"] = pd.NA

    tracker_rows = tracker_rows.rename(
        columns={
            "home_team_name": "home_team",
            "away_team_name": "away_team",
        }
    )
    tracker_rows = tracker_rows[PERFORMANCE_TRACKER_COLUMNS]

    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    if tracker_path.exists():
        existing = pd.read_csv(tracker_path)
    else:
        existing = pd.DataFrame(columns=PERFORMANCE_TRACKER_COLUMNS)

    existing["date"] = existing.get("date", pd.Series(dtype="object")).astype(str)
    existing["game_id"] = existing.get("game_id", pd.Series(dtype="object")).astype(str)

    key_mask = (existing["date"] == target_date.isoformat()) & (
        existing["game_id"].isin(tracker_rows["game_id"].astype(str))
    )
    existing = existing[~key_mask]

    combined = pd.concat([existing, tracker_rows], ignore_index=True)
    combined.to_csv(tracker_path, index=False)
    logging.info("Performance tracker updated: %s (%d rows total)", tracker_path, len(combined))


def load_regression_model(model_dir: str, model_name: str):
    """Loads a regression model and its metadata."""
    model_path_pkl = Path(model_dir) / f"{model_name}.pkl"
    model_path_booster = Path(model_dir) / f"{model_name}.booster"
    model_path_json = Path(model_dir) / f"{model_name}.json"
    meta_path = Path(model_dir) / f"{model_name}.meta.json"

    model = None
    if model_path_pkl.exists():
        model = joblib.load(model_path_pkl)
    elif model_path_json.exists():
        try:
            model = xgb.XGBRegressor()
            model.load_model(model_path_json)
        except Exception as e:
            logging.warning(f"Failed to load XGBoost .json model '{model_path_json}': {e}")
            return None, None, None
    elif model_path_booster.exists():
        try:
            model = xgb.XGBRegressor()
            model.load_model(model_path_booster)
        except Exception as e:
            logging.warning(f"Failed to load XGBoost .booster model '{model_path_booster}': {e}")
            return None, None, None
    else:
        logging.warning(f"Regression model not found: {model_path_pkl} or {model_path_booster} or {model_path_json}")
        return None, None, None

    # Compatibility shim for older XGBoost pickles
    try:
        if hasattr(model, "get_params"):
            for attr, default in {
                "use_label_encoder": False,
                "gpu_id": -1,
                "predictor": "auto",
            }.items():
                if not hasattr(model, attr):
                    try:
                        setattr(model, attr, default)
                    except Exception:
                        pass
    except Exception:
        pass
    feature_cols = None
    metric = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        feature_cols = meta.get("feature_cols")
        metric = meta.get("metrics", {}).get("rmse") or meta.get("metrics", {}).get("mae")

    return model, feature_cols, metric


def predict_regression(model, features: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.Series:
    """Makes predictions with a regression model."""
    if model is None or features.empty:
        return pd.Series(dtype=float)
    if feature_cols is None:
        feature_cols = [
            c
            for c in features.columns
            if c
            not in {"mlb_game_id", "game_id", "game_date", "game_datetime", "status"}
            and not c.endswith("_id")
        ]
    X = features.reindex(columns=feature_cols, fill_value=0).fillna(0)
    # Convert object columns to numeric
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    preds = model.predict(X)
    return pd.Series(preds, index=features.index)


def attach_odds_and_ev(
    features: pd.DataFrame,
    games: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    if features.empty:
        return features

    games = games.copy()
    odds_df = odds_df.copy()
    # Align dtypes to avoid merge/key conflicts
    if "game_id" in features.columns:
        features["game_id"] = features["game_id"].astype(str)
        games["game_id"] = games["game_id"].astype(str)
        if not odds_df.empty and "game_id" in odds_df.columns:
            odds_df["game_id"] = odds_df["game_id"].astype(str)
    if "mlb_game_id" in features.columns and "mlb_game_id" in games.columns:
        features["mlb_game_id"] = features["mlb_game_id"].astype(str)
        games["mlb_game_id"] = games["mlb_game_id"].astype(str)

    merge_cols = ["game_id", "mlb_game_id", "home_team_name", "away_team_name"]
    if "game_id" in features.columns:
        df = features.merge(
            games[merge_cols],
            how="left",
            on="game_id",
        )
    else:
        df = features.merge(
            games[merge_cols],
            how="left",
            left_on="mlb_game_id",
            right_on="mlb_game_id",
        )

    if odds_df.empty:
        df["home_price"] = pd.NA
        df["away_price"] = pd.NA
        return df

    # normalize selection to home/away if possible
    odds = odds_df.copy()
    odds["selection_norm"] = odds["selection"].str.lower().replace({"home": "home", "away": "away"})

    # map team name selections
    name_map = {}
    team_map = taiwan_lottery_crawler.load_team_name_map()
    for _, row in games.iterrows():
        home_name = taiwan_lottery_crawler.normalize_team_name(row["home_team_name"], team_map).lower()
        away_name = taiwan_lottery_crawler.normalize_team_name(row["away_team_name"], team_map).lower()
        name_map[(row["game_id"], home_name)] = "home"
        name_map[(row["game_id"], away_name)] = "away"

    def _normalize_selection(row):
        if row["selection_norm"] in {"home", "away"}:
            return row["selection_norm"]
        sel_name = taiwan_lottery_crawler.normalize_team_name(row["selection"], team_map).lower()
        key = (row["game_id"], sel_name)
        return name_map.get(key)

    odds["selection_norm"] = odds.apply(_normalize_selection, axis=1)

    home_price = odds[odds["selection_norm"] == "home"].groupby("game_id")["price"].last()
    away_price = odds[odds["selection_norm"] == "away"].groupby("game_id")["price"].last()

    df["home_price"] = df["game_id"].map(home_price)
    df["away_price"] = df["game_id"].map(away_price)

    return df


def _format_team_name_tw(name: Optional[str]) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return name
    en_name = str(name)
    tw_name = taiwan_lottery_crawler.EN_TO_TW_TEAM_MAP.get(en_name)
    if not tw_name or tw_name == en_name:
        return en_name
    return f"{tw_name} ({en_name})"


def _estimate_expected_total_runs(row: pd.Series, window_candidates: List[int]) -> Optional[float]:
    for window in window_candidates:
        col_home = f"home_roll{window}_runs"
        col_away = f"away_roll{window}_runs"
        if col_home in row and col_away in row and pd.notna(row[col_home]) and pd.notna(row[col_away]):
            try:
                return float(row[col_home]) + float(row[col_away])
            except Exception:
                return pd.NA
    # fallback: EWMA
    for window in window_candidates:
        col_home = f"home_ewm{window}_runs"
        col_away = f"away_ewm{window}_runs"
        if col_home in row and col_away in row and pd.notna(row[col_home]) and pd.notna(row[col_away]):
            try:
                return float(row[col_home]) + float(row[col_away])
            except Exception:
                return pd.NA
    return pd.NA


def _normalize_selection_for_market(
    selection: str,
    market: str,
    game_id: str,
    name_map: Dict[Tuple[str, str], str],
) -> Optional[str]:
    sel = str(selection or "").strip().lower()
    if sel in {"home", "h", "主"}:
        return "home"
    if sel in {"away", "a", "客"}:
        return "away"
    if sel in {"over", "o", "大"}:
        return "over"
    if sel in {"under", "u", "小"}:
        return "under"
    if sel in {"odd", "單"}:
        return "odd"
    if sel in {"even", "雙"}:
        return "even"

    if market in {"moneyline", "run_line", "spread"}:
        key = (game_id, sel)
        return name_map.get(key)
    return None


def build_market_rows(
    features: pd.DataFrame,
    games: pd.DataFrame,
    odds_df: pd.DataFrame,
    vig_rate: float,
    window_candidates: List[int],
    run_margin_sigma: float,
) -> pd.DataFrame:
    if odds_df is None or odds_df.empty:
        return pd.DataFrame()

    df = features.copy()
    games = games.copy()

    df["game_id"] = df["game_id"].astype(str)
    games["game_id"] = games["game_id"].astype(str)
    odds_df = odds_df.copy()
    odds_df["game_id"] = odds_df["game_id"].astype(str)

    merge_cols = ["game_id", "home_team_name", "away_team_name"]
    df = df.merge(games[merge_cols], on="game_id", how="left")

    # Use model prediction for total runs if available, otherwise estimate
    if "total_runs_pred" not in df.columns:
        df["total_runs_pred"] = pd.NA
    missing_mask = df["total_runs_pred"].isna()
    if missing_mask.any():
        df.loc[missing_mask, "total_runs_pred"] = df.loc[missing_mask].apply(
            lambda r: _estimate_expected_total_runs(r, window_candidates), axis=1
        ).values

    team_map = taiwan_lottery_crawler.load_team_name_map()
    name_map: Dict[Tuple[str, str], str] = {}
    for _, row in games.iterrows():
        home_name = taiwan_lottery_crawler.normalize_team_name(row["home_team_name"], team_map).lower()
        away_name = taiwan_lottery_crawler.normalize_team_name(row["away_team_name"], team_map).lower()
        name_map[(row["game_id"], home_name)] = "home"
        name_map[(row["game_id"], away_name)] = "away"

    odds_df["selection_norm"] = odds_df.apply(
        lambda r: _normalize_selection_for_market(r.get("selection"), r.get("market"), r.get("game_id"), name_map),
        axis=1,
    )

    def _calc_prob(row):
        market = row.get("market")
        sel = row.get("selection_norm")
        if sel is None:
            return pd.NA

        if market == "moneyline":
            home_prob = row.get("home_win_prob")
            if pd.isna(home_prob):
                return pd.NA
            return home_prob if sel == "home" else (1 - home_prob)

        if market in {"run_line", "spread"}:
            line = row.get("line")
            mu = row.get("run_margin_pred")
            if pd.isna(line) or pd.isna(mu) or not run_margin_sigma:
                return pd.NA
            try:
                line_val = float(line)
            except Exception:
                return pd.NA
            z = (0 - (mu + line_val)) / run_margin_sigma
            p_home = 1 - normal_cdf(z)
            return p_home if sel == "home" else 1 - p_home

        if market == "total":
            line = row.get("line")
            lam = row.get("total_runs_pred")
            if pd.isna(line) or pd.isna(lam):
                return pd.NA
            threshold = math.floor(float(line))
            p_under = poisson_cdf(threshold, float(lam))
            p_over = 1 - p_under if p_under is not None else pd.NA
            return p_over if sel == "over" else p_under

        if market == "odd_even":
            lam = row.get("total_runs_pred")
            if pd.isna(lam):
                return pd.NA
            p_odd = 0.5 * (1 - math.exp(-2 * float(lam)))
            p_even = 1 - p_odd
            return p_odd if sel == "odd" else p_even

        return pd.NA

    merged = odds_df.merge(
        df[["game_id", "home_team_name", "away_team_name", "home_win_prob", "run_margin_pred", "total_runs_pred"]],
        on="game_id",
        how="left",
    )

    merged = merged[merged["selection_norm"].notna()]
    if merged.empty:
        return pd.DataFrame()

    merged["model_prob"] = merged.apply(_calc_prob, axis=1)

    rows = []
    for _, row in merged.iterrows():
        match = f"{_format_team_name_tw(row['away_team_name'])} @ {_format_team_name_tw(row['home_team_name'])}"
        price = row.get("price")
        prob = row.get("model_prob")
        line = row.get("line")
        market = row.get("market")
        sel = row.get("selection_norm")

        if pd.isna(price):
            implied = pd.NA
            devig = pd.NA
            ev = pd.NA
            decimal_odds = pd.NA
        else:
            implied = implied_prob_from_american(int(price))
            devig = devig_prob(implied, vig_rate)
            ev = compute_ev(prob, int(price)) if pd.notna(prob) else pd.NA
            decimal_odds = american_to_decimal(int(price))

        edge = prob - devig if pd.notna(prob) and pd.notna(devig) else pd.NA
        conf = confidence_from_edge(edge) if pd.notna(edge) else pd.NA

        if market == "moneyline":
            side = "主" if sel == "home" else "客"
            market_label = f"獨贏-{side}"
            team_label = _format_team_name_tw(row["home_team_name"] if sel == "home" else row["away_team_name"])
        elif market in {"run_line", "spread"}:
            side = "主" if sel == "home" else "客"
            try:
                line_val = float(line)
            except Exception:
                line_val = None
            line_label = f"({line_val:+g})" if line_val is not None else ""
            market_label = f"讓分-{side}{line_label}"
            team_label = _format_team_name_tw(row["home_team_name"] if sel == "home" else row["away_team_name"])
        elif market == "total":
            side = "大" if sel == "over" else "小"
            try:
                line_val = float(line)
            except Exception:
                line_val = None
            line_label = f"({line_val:g})" if line_val is not None else ""
            market_label = f"大小-{side}{line_label}"
            team_label = "總分"
        elif market == "odd_even":
            side = "單" if sel == "odd" else "雙"
            market_label = f"單雙-{side}"
            team_label = "總分"
        else:
            market_label = str(market)
            team_label = str(row.get("selection"))

        rows.append({
            "比賽": match,
            "盤口": market_label,
            "赔率": decimal_odds,
            "模型勝率": prob,
            "去水勝率": devig,
            "Edge": edge,
            "EV": ev,
            "信心指數": conf,
            "隊伍": team_label,
        })

    return pd.DataFrame(rows)


def build_taiwan_format_rows(df: pd.DataFrame, vig_rate: float) -> pd.DataFrame:
    # Backwards compatibility: expect pre-built market rows
    return df if df is not None else pd.DataFrame()


def format_taiwan_output(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No games available."

    def _fmt_pct(val: float) -> str:
        if pd.isna(val):
            return "N/A"
        return f"{val * 100:.1f}%"

    def _fmt_ev(val: float) -> str:
        if pd.isna(val):
            return "N/A"
        return f"{val * 100:.2f}%"

    def _fmt_odds(val: float) -> str:
        if pd.isna(val):
            return "N/A"
        return f"{val:.2f}"

    def _fmt_conf(val: float) -> str:
        if pd.isna(val):
            return "N/A"
        return f"{val:.1f}"

    lines = []
    for match, group in df.groupby("比賽"):
        lines.append(f"比賽: {match}")
        for _, row in group.iterrows():
            lines.append(
                "- 盤口: {market} | 赔率: {odds} | 模型勝率: {model} | 去水勝率: {devig} | Edge: {edge} | EV: {ev} | 信心指數: {conf}".format(
                    market=row.get("盤口"),
                    odds=_fmt_odds(row.get("赔率")),
                    model=_fmt_pct(row.get("模型勝率")),
                    devig=_fmt_pct(row.get("去水勝率")),
                    edge=_fmt_pct(row.get("Edge")),
                    ev=_fmt_ev(row.get("EV")),
                    conf=_fmt_conf(row.get("信心指數")),
                )
            )
    return "\n".join(lines)


def build_recommendations(df: pd.DataFrame, ev_threshold: float) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    reco = df.copy()
    reco = reco[pd.notna(reco["EV"]) & (reco["EV"] > ev_threshold)]
    if reco.empty:
        return reco
    return reco.sort_values("EV", ascending=False)


def apply_tw_names_to_reco(reco: pd.DataFrame) -> pd.DataFrame:
    if reco is None or reco.empty:
        return reco
    reco = reco.copy()
    if "team" in reco.columns:
        reco["team"] = reco["team"].apply(_format_team_name_tw)
    if "opponent" in reco.columns:
        reco["opponent"] = reco["opponent"].apply(_format_team_name_tw)
    if "隊伍" in reco.columns:
        reco["隊伍"] = reco["隊伍"].apply(_format_team_name_tw)
    if "比賽" in reco.columns:
        # optional: replace team names inside match string
        pass
    return reco


# ---------------------
# CLI
# ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--window", type=int, default=10, help="Rolling window size")
    p.add_argument("--model-dir", default="./models", help="Directory with trained model artifacts")
    p.add_argument("--model", default="mlb_v9_platoon", help="Name of the classification model to use (e.g., mlb_v6_model, mlb_v8_platoon, mlb_v9_platoon)")
    p.add_argument("--out", help="Output CSV for recommendations")
    p.add_argument("--ev-threshold", type=float, default=0.0, help="Minimum EV to recommend")
    p.add_argument("--vig-rate", type=float, default=0.12, help="Taiwan lottery vig rate (10-15%%)")
    p.add_argument(
        "--output-format",
        choices=["taiwan", "reco"],
        default="taiwan",
        help="Output format: taiwan (default) or reco (positive EV only)",
    )
    p.add_argument("--refresh-schedule", action="store_true", help="Refresh MLB schedule via Stats API")
    p.add_argument("--refresh-odds", action="store_true", help="Fetch Taiwan odds and insert into DB")
    p.add_argument("--odds-url", help="Override Taiwan odds page URL")
    p.add_argument("--odds-file", help="Path to manual odds JSON (GameOdds format)")
    p.add_argument(
        "--odds-api",
        nargs="?",
        const="auto",
        default=None,
        help="Use The Odds API JSON file. Default path: data/odds/the-odds-api_YYYY-MM-DD.json. "
             "Use '--odds-api only' to replace Taiwan odds; or pass a file path.",
    )
    p.add_argument(
        "--offline-odds-json",
        default=None,
        help="Offline mode odds JSON path. Default: data/odds/the-odds-api_YYYY-MM-DD.json",
    )
    p.add_argument(
        "--offline-features-csv",
        default="data/features_2026-03-20.csv",
        help="Offline mode features template CSV. Falls back to latest data/features_*.csv when missing.",
    )
    p.add_argument(
        "--offline-base-rate",
        type=float,
        default=OFFLINE_BASE_RATE,
        help="Base rate fallback probability used when offline features/model are unavailable.",
    )
    p.add_argument(
        "--offline-max-games",
        type=int,
        default=None,
        help="Optional cap for offline output game count (useful when odds JSON has extra games).",
    )
    p.add_argument(
        "--performance-tracker",
        default="./data/performance_tracker.csv",
        help="CSV path for per-game model performance tracking.",
    )
    p.add_argument(
        "--static-model",
        default="static_model",
        help="Static-only fallback model name (without extension).",
    )
    p.add_argument(
        "--cold-start-zero-threshold",
        type=float,
        default=0.5,
        help="Switch to static model when rolling features zero-ratio exceeds this threshold.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else date.today()

    default_offline_odds = Path("data/odds") / f"the-odds-api_{target_date.isoformat()}.json"
    offline_odds_path = Path(args.offline_odds_json) if args.offline_odds_json else default_offline_odds
    offline_out_path = Path(args.out) if args.out else Path("data") / f"predictions_{target_date.isoformat()}.csv"

    engine = None
    try:
        engine = get_engine()
    except Exception as exc:
        logging.warning("Database unavailable, switching to offline mode if possible: %s", exc)

    if args.refresh_schedule:
        refresh_schedule(target_date)

    if args.refresh_odds or args.odds_file:
        if engine is None:
            logging.warning("Skip --refresh-odds because DB is unavailable")
        else:
            refresh_taiwan_odds(engine, target_date, url=args.odds_file or args.odds_url)

    feature_dates = {target_date}
    if args.odds_api:
        # Look at adjacent days to handle timezone diffs in odds files
        feature_dates.add(target_date - timedelta(days=1))
        feature_dates.add(target_date + timedelta(days=1))

    games_frames: List[pd.DataFrame] = []
    feature_frames: List[pd.DataFrame] = []

    if engine is not None:
        for d in sorted(list(feature_dates)):
            try:
                games_frames.append(load_games_with_teams(engine, d))
            except Exception as exc:
                logging.warning("Failed to load games for %s: %s", d, exc)
            try:
                feature_frames.append(build_feature_table(d, window=args.window))
            except Exception as exc:
                logging.warning("Failed to build features for %s: %s", d, exc)

    games_df = pd.concat(games_frames, ignore_index=True) if games_frames else pd.DataFrame()
    if not games_df.empty:
        games_df = games_df.drop_duplicates(subset=["game_id"]).reset_index(drop=True)

    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    if not features.empty:
        features = features.drop_duplicates(subset=["game_id"]).reset_index(drop=True)

    # Cold start / empty DB fallback:
    # run predictions directly from odds JSON + template features.
    if games_df.empty or features.empty:
        logging.warning(
            "DB mode unavailable (games=%d, features=%d). Using offline odds-json mode.",
            len(games_df),
            len(features),
        )
        predictions = run_offline_prediction_mode(
            target_date=target_date,
            model_dir=args.model_dir,
            model_name=args.model,
            odds_json_path=offline_odds_path,
            output_path=offline_out_path,
            feature_template_path=args.offline_features_csv,
            base_rate=args.offline_base_rate,
            max_games=args.offline_max_games,
        )
        if predictions.empty:
            logging.error("Offline mode failed: no prediction output generated")
        return

    # --- Load Models ---
    # Moneyline/classification model
    # Try to load the user-specified model, but fall back to a known good one if it fails.
    try:
        ml_model, ml_feature_cols, ml_n_samples = load_model_and_meta(args.model_dir, args.model)
    except (FileNotFoundError, xgb.core.XGBoostError) as e:
        logging.error(f"Failed to load specified model '{args.model}': {e}. Falling back to 'mlb_v6_model.pkl'.")
        # Fallback to a known working model
        args.model = "mlb_v6_model"
        ml_model, ml_feature_cols, ml_n_samples = load_model_and_meta(args.model_dir, args.model)

    # If model was trained on very few samples, it is unreliable - use base rate fallback
    # League average home win rate is ~52%; use that when model can't be trusted
    HOME_BASE_RATE = 0.52
    if ml_n_samples is not None and ml_n_samples < 100:
        logging.warning(
            "Model '%s' trained on only %d samples — too few to be reliable. "
            "Using base rate (%.0f%% home win) as fallback.",
            args.model, ml_n_samples, HOME_BASE_RATE * 100,
        )
        features["home_win_prob"] = HOME_BASE_RATE
        features["used_static_fallback"] = False
        features["rolling_zero_ratio"] = pd.NA
    else:
        base_probs = predict_win_prob(ml_model, features, ml_feature_cols)
        fallback_probs, cold_mask, rolling_zero_ratio = apply_static_fallback(
            features=features,
            base_probs=base_probs,
            model_dir=args.model_dir,
            base_feature_cols=ml_feature_cols,
            zero_threshold=args.cold_start_zero_threshold,
            static_model_name=args.static_model,
        )
        features["home_win_prob"] = fallback_probs
        features["used_static_fallback"] = cold_mask.astype(bool)
        features["rolling_zero_ratio"] = rolling_zero_ratio

    # Runline/spread model - attempt to load v8, but it will gracefully fail and return None
    runline_model, runline_cols, runline_sigma = load_regression_model(args.model_dir, "mlb_v8_runline")
    # If v8 fails, fall back to original run margin model
    if runline_model is None:
        logging.warning("Could not load 'mlb_v8_runline', falling back to 'mlb_run_margin_model.pkl'")
        runline_model, runline_cols, runline_sigma = load_regression_model(args.model_dir, "mlb_run_margin_model")

    features["run_margin_pred"] = predict_regression(runline_model, features, runline_cols)

    # Over/Under model - attempt to load v8, will gracefully fail. No fallback, will use estimation.
    overunder_model, overunder_cols, _ = load_regression_model(args.model_dir, "mlb_v8_overunder")
    features["total_runs_pred"] = predict_regression(overunder_model, features, overunder_cols)

    # --- Load Odds ---
    odds_df = pd.DataFrame()
    if engine is not None:
        try:
            odds_df = load_odds(engine, target_date)
        except Exception as exc:
            logging.warning("Failed to load DB odds for %s: %s", target_date, exc)
            odds_df = pd.DataFrame()

    if args.odds_api and engine is not None:
        # To handle timezone differences between odds files and the DB schedule,
        # build a comprehensive games lookup that includes adjacent days.
        games_df_prev = load_games_with_teams(engine, target_date - timedelta(days=1))
        games_df_next = load_games_with_teams(engine, target_date + timedelta(days=1))
        comprehensive_games_df = pd.concat(
            [games_df.copy(), games_df_prev.copy(), games_df_next.copy()], ignore_index=True
        ).drop_duplicates(subset=["game_id"])

        replace_only = str(args.odds_api).lower() in {"only", "replace"}
        if str(args.odds_api).lower() in {"auto", "only", "replace"}:
            odds_api_path = Path("data/odds") / f"the-odds-api_{target_date.isoformat()}.json"
        else:
            odds_api_path = Path(args.odds_api)

        if not odds_api_path.exists():
            logging.warning("The Odds API JSON not found: %s", odds_api_path)
        else:
            odds_api_df = load_manual_odds_json_to_df(str(odds_api_path), comprehensive_games_df)
            if odds_api_df.empty:
                logging.warning("The Odds API JSON loaded but no odds matched to games.")
            else:
                if replace_only or odds_df.empty:
                    odds_df = odds_api_df
                else:
                    odds_df = pd.concat([odds_df, odds_api_df], ignore_index=True)
                logging.info("Loaded %d odds rows from The Odds API", len(odds_api_df))

    update_performance_tracker(
        target_date=target_date,
        games_df=games_df,
        features=features,
        odds_df=odds_df,
        tracker_path=Path(args.performance_tracker),
    )

    # Clamp vig rate into Taiwan lottery range (10-15%)
    vig_rate = max(0.10, min(0.15, args.vig_rate))
    window_candidates = [args.window, 15, 10, 5, 30]

    taiwan_rows = build_market_rows(
        features,
        games_df,
        odds_df,
        vig_rate=vig_rate,
        window_candidates=window_candidates,
        run_margin_sigma=runline_sigma or 4.3,
    )

    reco = build_recommendations(taiwan_rows, args.ev_threshold)
    if reco.empty:
        logging.info("No positive EV recommendations for %s", target_date)
    else:
        logging.info("Found %d positive EV recommendations", len(reco))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.output_format == "taiwan":
            taiwan_rows.to_csv(out_path, index=False)
        else:
            reco.to_csv(out_path, index=False)
        logging.info("Saved recommendations to %s", out_path)
    else:
        if args.output_format == "taiwan":
            print(format_taiwan_output(taiwan_rows))
        else:
            print(reco.to_string(index=False))

    if send_discord_recommendations and not reco.empty:
        try:
            discord_reco = apply_tw_names_to_reco(reco)
            sent = send_discord_recommendations(discord_reco, target_date)
            if sent:
                logging.info("Discord webhook sent %d recommendations", len(reco))
        except Exception as exc:
            logging.error("Failed to send Discord webhook: %s", exc)


if __name__ == "__main__":
    main()
