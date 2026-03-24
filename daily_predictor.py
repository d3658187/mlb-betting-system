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
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
import joblib

import feature_builder
import bullpen_fatigue
import mlb_stats_crawler
import taiwan_lottery_crawler

try:
    from discord_notifier import send_discord_recommendations
except Exception:  # pragma: no cover - optional dependency
    send_discord_recommendations = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


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


def load_model_and_meta(model_dir: str):
    model_path = Path(model_dir) / "mlb_2025_model.pkl"
    meta_path = Path(model_dir) / "mlb_2025_model.meta.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
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
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        feature_cols = meta.get("feature_cols")
    return model, feature_cols


def predict_win_prob(model, features: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.Series:
    if features.empty:
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
    X = features.reindex(columns=feature_cols, fill_value=0).fillna(0)
    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, index=features.index)


def load_run_margin_model(model_dir: str):
    model_path = Path(model_dir) / "mlb_run_margin_model.pkl"
    meta_path = Path(model_dir) / "mlb_run_margin_model.meta.json"
    if not model_path.exists():
        return None, None, None
    model = joblib.load(model_path)
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
    rmse = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        feature_cols = meta.get("feature_cols")
        rmse = meta.get("metrics", {}).get("rmse")
    if not rmse:
        rmse = 4.3
    return model, feature_cols, rmse


def predict_run_margin(model, features: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.Series:
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

    # Precompute expected total runs
    df["expected_total_runs"] = df.apply(
        lambda r: _estimate_expected_total_runs(r, window_candidates), axis=1
    )

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
            lam = row.get("expected_total_runs")
            if pd.isna(line) or pd.isna(lam):
                return pd.NA
            threshold = math.floor(float(line))
            p_under = poisson_cdf(threshold, float(lam))
            p_over = 1 - p_under if p_under is not None else pd.NA
            return p_over if sel == "over" else p_under

        if market == "odd_even":
            lam = row.get("expected_total_runs")
            if pd.isna(lam):
                return pd.NA
            p_odd = 0.5 * (1 - math.exp(-2 * float(lam)))
            p_even = 1 - p_odd
            return p_odd if sel == "odd" else p_even

        return pd.NA

    merged = odds_df.merge(
        df[["game_id", "home_team_name", "away_team_name", "home_win_prob", "run_margin_pred", "expected_total_runs"]],
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
    return p.parse_args()


def main():
    args = parse_args()
    target_date = date.fromisoformat(args.date) if args.date else date.today()

    engine = get_engine()

    if args.refresh_schedule:
        refresh_schedule(target_date)

    if args.refresh_odds or args.odds_file:
        refresh_taiwan_odds(engine, target_date, url=args.odds_file or args.odds_url)

    games_df = load_games_with_teams(engine, target_date)
    if games_df.empty:
        logging.warning("No games found for %s", target_date)
        return

    features = build_feature_table(target_date, window=args.window)
    if features.empty:
        logging.warning("No features built for %s", target_date)
        return

    model, feature_cols = load_model_and_meta(args.model_dir)
    features["home_win_prob"] = predict_win_prob(model, features, feature_cols)

    run_margin_model, run_margin_cols, run_margin_sigma = load_run_margin_model(args.model_dir)
    features["run_margin_pred"] = predict_run_margin(run_margin_model, features, run_margin_cols)

    odds_df = load_odds(engine, target_date)

    if args.odds_api:
        replace_only = str(args.odds_api).lower() in {"only", "replace"}
        if str(args.odds_api).lower() in {"auto", "only", "replace"}:
            odds_api_path = Path("data/odds") / f"the-odds-api_{target_date.isoformat()}.json"
        else:
            odds_api_path = Path(args.odds_api)

        if not odds_api_path.exists():
            logging.warning("The Odds API JSON not found: %s", odds_api_path)
        else:
            odds_api_df = load_manual_odds_json_to_df(str(odds_api_path), games_df)
            if odds_api_df.empty:
                logging.warning("The Odds API JSON loaded but no odds matched to games.")
            else:
                if replace_only or odds_df.empty:
                    odds_df = odds_api_df
                else:
                    odds_df = pd.concat([odds_df, odds_api_df], ignore_index=True)
                logging.info("Loaded %d odds rows from The Odds API", len(odds_api_df))

    # Clamp vig rate into Taiwan lottery range (10-15%)
    vig_rate = max(0.10, min(0.15, args.vig_rate))
    window_candidates = [args.window, 15, 10, 5, 30]

    taiwan_rows = build_market_rows(
        features,
        games_df,
        odds_df,
        vig_rate=vig_rate,
        window_candidates=window_candidates,
        run_margin_sigma=run_margin_sigma or 4.3,
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
