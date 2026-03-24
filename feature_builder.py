#!/usr/bin/env python3
"""Feature engineering scaffold for MLB spread/margin modeling.

Usage:
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  python feature_builder.py --date 2026-03-13 --out ./features_2026-03-13.csv

Notes:
- This is a framework. Adjust SQL fields to match your actual schema.
- Expected raw tables (example): games, stats_batting, stats_pitching, game_results
- Adds multi-scale rolling (5/15/30) and EWMA features by default.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, List, Tuple

import numpy as np

import pandas as pd
from sqlalchemy import create_engine, text

try:
    from pybaseball import statcast_batter_exitvelo_barrels, statcast_pitcher
except Exception:  # pragma: no cover
    statcast_batter_exitvelo_barrels = None
    statcast_pitcher = None

try:
    import taiwan_lottery_crawler
except Exception:  # pragma: no cover
    taiwan_lottery_crawler = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------
# DB helpers
# ---------------------

def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    return create_engine(db_url, pool_pre_ping=True)


def table_exists(engine, table: str) -> bool:
    sql = text(
        """
        SELECT EXISTS (
          SELECT 1 FROM information_schema.tables
          WHERE table_schema = 'public' AND table_name = :table
        )
        """
    )
    with engine.begin() as conn:
        return bool(conn.execute(sql, {"table": table}).scalar())


def _find_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def load_team_abbrev_map(engine) -> pd.DataFrame:
    try:
        df = pd.read_sql("SELECT mlb_team_id, abbreviation FROM teams", engine)
    except Exception:
        return pd.DataFrame(columns=["mlb_team_id", "abbreviation"])
    return df.dropna(subset=["mlb_team_id", "abbreviation"])


# ---------------------
# Loaders
# ---------------------

def load_games(engine, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT g.id AS game_id,
               g.mlb_game_id,
               g.game_date,
               g.game_datetime,
               th.mlb_team_id AS home_team_id,
               ta.mlb_team_id AS away_team_id,
               g.status
          FROM games g
          JOIN teams th ON th.id = g.home_team_id
          JOIN teams ta ON ta.id = g.away_team_id
         WHERE g.game_date = :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"target_date": target_date})


def load_batting(engine, since_date: date, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT g.game_date,
               b.game_id,
               b.team_mlb_id,
               b.at_bats,
               b.hits,
               b.runs,
               b.rbi,
               b.walks,
               b.strikeouts,
               b.avg,
               b.obp,
               b.slg,
               b.ops
          FROM stats_batting b
          JOIN games g ON g.id = b.game_id
         WHERE g.game_date >= :since_date
           AND g.game_date < :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": since_date, "target_date": target_date})


def load_pitching(engine, since_date: date, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT g.game_date,
               p.game_id,
               p.pitcher_mlb_id,
               p.team_mlb_id,
               p.innings_pitched,
               p.runs,
               p.earned_runs,
               p.walks,
               p.strikeouts,
               p.era,
               p.whip
          FROM stats_pitching p
          JOIN games g ON g.id = p.game_id
         WHERE g.game_date >= :since_date
           AND g.game_date < :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": since_date, "target_date": target_date})


def load_results(engine, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT g.mlb_game_id,
               r.home_score,
               r.away_score,
               r.home_win,
               r.total_points
          FROM game_results r
          JOIN games g ON g.id = r.game_id
         WHERE g.game_date = :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"target_date": target_date})


def load_moneyline_odds(engine, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT o.game_id,
               o.selection,
               o.price,
               o.sportsbook,
               o.retrieved_at,
               th.name AS home_team_name,
               ta.name AS away_team_name
          FROM odds o
          JOIN games g ON g.id = o.game_id
          JOIN teams th ON g.home_team_id = th.id
          JOIN teams ta ON g.away_team_id = ta.id
         WHERE g.game_date = :target_date
           AND o.market = 'moneyline'
        """
    )
    return pd.read_sql(sql, engine, params={"target_date": target_date})


def load_starting_pitchers(engine, since_date: date, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT g.game_date,
               s.game_id,
               s.team_mlb_id,
               s.is_home,
               s.pitcher_mlb_id
          FROM starting_pitchers s
          JOIN games g ON g.mlb_game_id = s.game_id
         WHERE g.game_date >= :since_date
           AND g.game_date <= :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"since_date": since_date, "target_date": target_date})


def load_fangraphs_team_batting(engine, season: int) -> pd.DataFrame:
    sql = text(
        """
        SELECT team_id, team, wrc_plus, woba, xwoba, ops_plus
          FROM fangraphs_team_batting
         WHERE season = :season
        """
    )
    return pd.read_sql(sql, engine, params={"season": season})


def load_fangraphs_pitchers(engine, season: int) -> pd.DataFrame:
    sql = text(
        """
        SELECT player_id, fip, xfip, k_per_9, bb_per_9
          FROM fangraphs_pitchers
         WHERE season = :season
        """
    )
    return pd.read_sql(sql, engine, params={"season": season})


def load_platoon_splits_csv(data_dir: str, season: int) -> pd.DataFrame:
    path = Path(data_dir) / f"platoon_splits_{season}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()
    if "mlbam_id" in df.columns:
        df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    return df


def load_bullpen_fatigue(engine, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT team_mlb_id,
               bullpen_fatigue_index,
               bullpen_pitch_count,
               bullpen_appearance_days,
               bullpen_pitcher_count,
               bullpen_avg_rest_days
          FROM bullpen_fatigue
         WHERE game_date = :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"target_date": target_date})


def load_game_weather(engine, target_date: date) -> pd.DataFrame:
    sql = text(
        """
        SELECT w.mlb_game_id,
               w.temperature_c AS weather_temperature_c,
               w.relative_humidity AS weather_relative_humidity,
               w.wind_speed AS weather_wind_speed,
               w.wind_direction AS weather_wind_direction
          FROM game_weather w
          JOIN games g ON g.mlb_game_id = w.mlb_game_id
         WHERE g.game_date = :target_date
        """
    )
    return pd.read_sql(sql, engine, params={"target_date": target_date})


def _run_fangraphs_crawler(season: int, mode: str = "both") -> None:
    script_path = Path(__file__).with_name("fangraphs_crawler.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--season",
        str(season),
        "--mode",
        mode,
        "--store-db",
        "--init-db",
    ]
    logging.info("Triggering pybaseball loader (mode=%s, season=%s)", mode, season)
    try:
        subprocess.run(cmd, check=False, env=os.environ.copy())
    except Exception as exc:
        logging.warning("pybaseball loader failed to run: %s", exc)


def _apply_fangraphs_team_features(features: pd.DataFrame, fg_teams: pd.DataFrame, engine) -> Tuple[pd.DataFrame, bool]:
    if fg_teams.empty:
        return features, False

    metrics = {
        "wrc_plus": "fangraphs_wrc_plus",
        "woba": "fangraphs_woba",
        "xwoba": "fangraphs_xwoba",
        "ops_plus": "fangraphs_ops_plus",
    }

    team_info = pd.read_sql(
        text("SELECT mlb_team_id, abbreviation, name FROM teams"), engine
    )
    mlb_to_abbr = team_info.set_index("mlb_team_id")["abbreviation"].to_dict()
    mlb_to_name = team_info.set_index("mlb_team_id")["name"].to_dict()

    missing_any = False
    for fg_col, feat_suffix in metrics.items():
        if fg_col not in fg_teams.columns:
            continue
        fg_map = fg_teams.dropna(subset=["team_id"]).set_index("team_id")[fg_col].to_dict()
        home_col = f"home_{feat_suffix}"
        away_col = f"away_{feat_suffix}"
        features[home_col] = features["home_team_id"].map(fg_map)
        features[away_col] = features["away_team_id"].map(fg_map)

        missing_mask = features[[home_col, away_col]].isna().any(axis=1)
        if missing_mask.any():
            logging.info("%s missing for %d games before fallback mapping", fg_col, int(missing_mask.sum()))

        if missing_mask.any() and "team" in fg_teams.columns:
            fg_by_abbr = (
                fg_teams.dropna(subset=["team"])
                .set_index(fg_teams["team"].str.upper())[fg_col]
                .to_dict()
            )
            fg_by_name = (
                fg_teams.dropna(subset=["team"])
                .set_index(fg_teams["team"].str.lower())[fg_col]
                .to_dict()
            )

            def _fallback(team_id):
                abbr = mlb_to_abbr.get(team_id)
                if abbr and abbr.upper() in fg_by_abbr:
                    return fg_by_abbr.get(abbr.upper())
                name = mlb_to_name.get(team_id)
                if name and name.lower() in fg_by_name:
                    return fg_by_name.get(name.lower())
                return pd.NA

            home_missing = features[home_col].isna()
            if home_missing.any():
                features.loc[home_missing, home_col] = (
                    features.loc[home_missing, "home_team_id"].map(_fallback)
                )
            away_missing = features[away_col].isna()
            if away_missing.any():
                features.loc[away_missing, away_col] = (
                    features.loc[away_missing, "away_team_id"].map(_fallback)
                )

        missing_after_mask = features[[home_col, away_col]].isna().any(axis=1)
        if missing_after_mask.any():
            logging.warning("%s still missing for %d games after fallback mapping", fg_col, int(missing_after_mask.sum()))
            missing_any = True

    return features, missing_any


def _apply_fangraphs_pitcher_features(features: pd.DataFrame, fg_pitchers: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    if fg_pitchers.empty:
        return features, False

    fg_pitch_map = fg_pitchers.set_index("player_id")[["fip", "xfip", "k_per_9", "bb_per_9"]].to_dict("index")

    missing = False
    if "home_pitcher_mlb_id" in features.columns:
        features["home_starter_fip"] = features["home_pitcher_mlb_id"].map(
            lambda pid: fg_pitch_map.get(pid, {}).get("fip")
        )
        features["home_starter_xfip"] = features["home_pitcher_mlb_id"].map(
            lambda pid: fg_pitch_map.get(pid, {}).get("xfip")
        )
        features["home_starter_k_per_9"] = features["home_pitcher_mlb_id"].map(
            lambda pid: fg_pitch_map.get(pid, {}).get("k_per_9")
        )
        features["home_starter_bb_per_9"] = features["home_pitcher_mlb_id"].map(
            lambda pid: fg_pitch_map.get(pid, {}).get("bb_per_9")
        )
        missing_home = (
            features["home_pitcher_mlb_id"].notna() & features["home_starter_fip"].isna()
        )
        missing |= missing_home.any()
        if missing_home.any():
            logging.info("Starter FIP missing for %d home pitchers", int(missing_home.sum()))

    if "away_pitcher_mlb_id" in features.columns:
        features["away_starter_fip"] = features["away_pitcher_mlb_id"].map(
            lambda pid: fg_pitch_map.get(pid, {}).get("fip")
        )
        features["away_starter_xfip"] = features["away_pitcher_mlb_id"].map(
            lambda pid: fg_pitch_map.get(pid, {}).get("xfip")
        )
        features["away_starter_k_per_9"] = features["away_pitcher_mlb_id"].map(
            lambda pid: fg_pitch_map.get(pid, {}).get("k_per_9")
        )
        features["away_starter_bb_per_9"] = features["away_pitcher_mlb_id"].map(
            lambda pid: fg_pitch_map.get(pid, {}).get("bb_per_9")
        )
        missing_away = (
            features["away_pitcher_mlb_id"].notna() & features["away_starter_fip"].isna()
        )
        missing |= missing_away.any()
        if missing_away.any():
            logging.info("Starter FIP missing for %d away pitchers", int(missing_away.sum()))

    return features, bool(missing)


def _apply_platoon_splits_features(features: pd.DataFrame, platoon_df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    if platoon_df.empty:
        return features, False

    required_cols = [
        "platoon_ba_diff",
        "platoon_ops_diff",
        "platoon_k_rate_lhb",
        "platoon_k_rate_rhb",
        "platoon_splits_score",
    ]
    if "mlbam_id" not in platoon_df.columns:
        return features, False

    platoon_df = platoon_df.copy()
    platoon_df["mlbam_id"] = pd.to_numeric(platoon_df["mlbam_id"], errors="coerce")

    avail_cols = [c for c in required_cols if c in platoon_df.columns]
    if not avail_cols:
        return features, False

    for col in avail_cols:
        platoon_df[col] = pd.to_numeric(platoon_df[col], errors="coerce")

    pmap = platoon_df.set_index("mlbam_id")[avail_cols].to_dict("index")

    missing = False
    if "home_pitcher_mlb_id" in features.columns:
        for col in avail_cols:
            features[f"home_{col}"] = features["home_pitcher_mlb_id"].map(
                lambda pid, c=col: pmap.get(pid, {}).get(c)
            )
        if "home_platoon_ba_diff" in features.columns:
            missing_home = (
                features["home_pitcher_mlb_id"].notna() & features["home_platoon_ba_diff"].isna()
            )
            missing |= missing_home.any()
            if missing_home.any():
                logging.info("Platoon splits missing for %d home pitchers", int(missing_home.sum()))

    if "away_pitcher_mlb_id" in features.columns:
        for col in avail_cols:
            features[f"away_{col}"] = features["away_pitcher_mlb_id"].map(
                lambda pid, c=col: pmap.get(pid, {}).get(c)
            )
        if "away_platoon_ba_diff" in features.columns:
            missing_away = (
                features["away_pitcher_mlb_id"].notna() & features["away_platoon_ba_diff"].isna()
            )
            missing |= missing_away.any()
            if missing_away.any():
                logging.info("Platoon splits missing for %d away pitchers", int(missing_away.sum()))

    return features, bool(missing)


def attach_fangraphs_features(features: pd.DataFrame, engine, season: int) -> pd.DataFrame:
    """B-layer pybaseball補值：只有在缺少 FIP / wRC+ 時才觸發 loader。"""
    # Teams (wRC+)
    triggered = False
    if not table_exists(engine, "fangraphs_team_batting"):
        _run_fangraphs_crawler(season, mode="teams")
        triggered = True
    fg_teams = load_fangraphs_team_batting(engine, season) if table_exists(engine, "fangraphs_team_batting") else pd.DataFrame()
    if fg_teams.empty and not triggered:
        _run_fangraphs_crawler(season, mode="teams")
        triggered = True
        fg_teams = load_fangraphs_team_batting(engine, season) if table_exists(engine, "fangraphs_team_batting") else pd.DataFrame()

    if not fg_teams.empty:
        features, missing = _apply_fangraphs_team_features(features, fg_teams, engine)
        if missing and not triggered:
            _run_fangraphs_crawler(season, mode="teams")
            fg_teams = load_fangraphs_team_batting(engine, season)
            features, _ = _apply_fangraphs_team_features(features, fg_teams, engine)

    # Pitchers (FIP/xFIP)
    triggered = False
    if not table_exists(engine, "fangraphs_pitchers"):
        _run_fangraphs_crawler(season, mode="pitchers")
        triggered = True
    fg_pitchers = load_fangraphs_pitchers(engine, season) if table_exists(engine, "fangraphs_pitchers") else pd.DataFrame()
    if fg_pitchers.empty and not triggered:
        _run_fangraphs_crawler(season, mode="pitchers")
        triggered = True
        fg_pitchers = load_fangraphs_pitchers(engine, season) if table_exists(engine, "fangraphs_pitchers") else pd.DataFrame()

    if not fg_pitchers.empty:
        features, missing = _apply_fangraphs_pitcher_features(features, fg_pitchers)
        if missing and not triggered:
            _run_fangraphs_crawler(season, mode="pitchers")
            fg_pitchers = load_fangraphs_pitchers(engine, season)
            features, _ = _apply_fangraphs_pitcher_features(features, fg_pitchers)

    return features


# ---------------------
# Feature builders
# ---------------------


def _innings_to_float(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return pd.NA
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            if "." in val:
                whole, frac = val.split(".", 1)
                frac_int = int(frac)
                if frac_int in (1, 2):
                    return int(whole) + frac_int / 3
            return float(val)
        except ValueError:
            return pd.NA
    return pd.NA


def _roll_team_stats_multi(
    team_game_df: pd.DataFrame,
    windows: Sequence[int],
    ewm_spans: Sequence[int],
) -> pd.DataFrame:
    """Compute multi-scale rolling mean + EWMA, excluding current game."""
    team_game_df = team_game_df.sort_values(["team_mlb_id", "game_date"])
    numeric_cols = [
        c
        for c in team_game_df.columns
        if c not in {"team_mlb_id", "game_date", "game_id"} and not c.endswith("_id")
    ]

    base = team_game_df[["team_mlb_id", "game_date"]].reset_index(drop=True)
    grouped = team_game_df.groupby("team_mlb_id", group_keys=False)
    features = [base]

    for window in windows:
        rolled = (
            grouped[numeric_cols]
            .rolling(window=window, min_periods=1)
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )
        rolled = rolled.add_prefix(f"roll{window}_")
        features.append(rolled.reset_index(drop=True))

    for span in ewm_spans:
        ewm = (
            grouped[numeric_cols]
            .apply(lambda df: df.ewm(span=span, adjust=False).mean().shift(1))
            .reset_index(level=0, drop=True)
        )
        ewm = ewm.add_prefix(f"ewm{span}_")
        features.append(ewm.reset_index(drop=True))

    return pd.concat(features, axis=1)


def build_team_batting_features(
    batting_df: pd.DataFrame,
    windows: Sequence[int],
    ewm_spans: Sequence[int],
) -> pd.DataFrame:
    numeric_cols = [
        "at_bats",
        "hits",
        "runs",
        "rbi",
        "walks",
        "strikeouts",
        "avg",
        "obp",
        "slg",
        "ops",
        "batting_avg",
    ]

    if batting_df.empty:
        cols = ["team_mlb_id", "game_date"]
        for window in windows:
            cols += [f"roll{window}_{c}" for c in numeric_cols]
        for span in ewm_spans:
            cols += [f"ewm{span}_{c}" for c in numeric_cols]
        return pd.DataFrame(columns=cols)

    batting_df = batting_df.copy()
    base_numeric = [c for c in numeric_cols if c != "batting_avg"]
    for col in base_numeric:
        if col in batting_df.columns:
            batting_df[col] = pd.to_numeric(batting_df[col], errors="coerce")

    team_game = (
        batting_df
        .groupby(["team_mlb_id", "game_date"], as_index=False)
        .agg(
            at_bats=("at_bats", "sum"),
            hits=("hits", "sum"),
            runs=("runs", "sum"),
            rbi=("rbi", "sum"),
            walks=("walks", "sum"),
            strikeouts=("strikeouts", "sum"),
            avg=("avg", "mean"),
            obp=("obp", "mean"),
            slg=("slg", "mean"),
            ops=("ops", "mean"),
        )
    )
    team_game["batting_avg"] = team_game["hits"] / team_game["at_bats"].replace(0, pd.NA)
    team_game["batting_avg"] = pd.to_numeric(team_game["batting_avg"], errors="coerce")

    return _roll_team_stats_multi(team_game, windows=windows, ewm_spans=ewm_spans)


def build_team_pitching_features(
    pitching_df: pd.DataFrame,
    windows: Sequence[int],
    ewm_spans: Sequence[int],
) -> pd.DataFrame:
    numeric_cols = [
        "innings_pitched",
        "runs_allowed",
        "earned_runs",
        "walks_allowed",
        "strikeouts",
        "era",
        "whip",
        "era_calc",
    ]

    if pitching_df.empty:
        cols = ["team_mlb_id", "game_date"]
        for window in windows:
            cols += [f"pitch_roll{window}_{c}" for c in numeric_cols]
        for span in ewm_spans:
            cols += [f"pitch_ewm{span}_{c}" for c in numeric_cols]
        return pd.DataFrame(columns=cols)

    pitching_df = pitching_df.copy()
    base_numeric = ["innings_pitched", "runs", "earned_runs", "walks", "strikeouts", "era", "whip"]
    for col in base_numeric:
        if col in pitching_df.columns:
            pitching_df[col] = pd.to_numeric(pitching_df[col], errors="coerce")

    team_game = (
        pitching_df
        .groupby(["team_mlb_id", "game_date"], as_index=False)
        .agg(
            innings_pitched=("innings_pitched", "sum"),
            runs_allowed=("runs", "sum"),
            earned_runs=("earned_runs", "sum"),
            walks_allowed=("walks", "sum"),
            strikeouts=("strikeouts", "sum"),
            era=("era", "mean"),
            whip=("whip", "mean"),
        )
    )
    team_game["era_calc"] = 9.0 * team_game["earned_runs"] / team_game["innings_pitched"].replace(0, pd.NA)
    team_game["era_calc"] = pd.to_numeric(team_game["era_calc"], errors="coerce")

    features = _roll_team_stats_multi(team_game, windows=windows, ewm_spans=ewm_spans)
    rename_cols = {
        c: f"pitch_{c}"
        for c in features.columns
        if c not in {"team_mlb_id", "game_date"}
    }
    return features.rename(columns=rename_cols)


def build_starter_rolling_features(starters_df: pd.DataFrame, pitching_df: pd.DataFrame) -> pd.DataFrame:
    if starters_df.empty or pitching_df.empty:
        return pd.DataFrame()

    starters = starters_df.copy()
    starters["game_date"] = pd.to_datetime(starters["game_date"])

    pitching = pitching_df.copy()
    for col in ["strikeouts", "walks", "earned_runs", "innings_pitched", "whip"]:
        if col in pitching.columns:
            pitching[col] = pd.to_numeric(pitching[col], errors="coerce")
    pitching["innings_pitched"] = pitching["innings_pitched"].apply(_innings_to_float)

    starter_game = starters.merge(
        pitching[["game_id", "pitcher_mlb_id", "strikeouts", "walks", "earned_runs", "innings_pitched", "whip"]],
        on=["game_id", "pitcher_mlb_id"],
        how="left",
    )

    starter_game = starter_game.sort_values(["pitcher_mlb_id", "game_date"])
    group = starter_game.groupby("pitcher_mlb_id", group_keys=False)

    starter_game["k_last3"] = (
        group["strikeouts"].rolling(window=3, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
    )
    starter_game["bb_last3"] = (
        group["walks"].rolling(window=3, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
    )
    starter_game["ip_last3"] = (
        group["innings_pitched"].rolling(window=3, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
    )
    starter_game["er_last3"] = (
        group["earned_runs"].rolling(window=3, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
    )

    starter_game["starter_kbb_last3"] = starter_game["k_last3"] / starter_game["bb_last3"].replace(0, pd.NA)
    starter_game["starter_era_last3"] = 9.0 * starter_game["er_last3"] / starter_game["ip_last3"].replace(0, pd.NA)

    starter_game["k_rate"] = starter_game["strikeouts"] / (
        starter_game["innings_pitched"] * 3
    ).replace(0, pd.NA)

    for window in (5, 10):
        k_roll = group["strikeouts"].rolling(window=window, min_periods=1).sum().shift(1)
        ip_roll = group["innings_pitched"].rolling(window=window, min_periods=1).sum().shift(1)
        er_roll = group["earned_runs"].rolling(window=window, min_periods=1).sum().shift(1)
        whip_roll = group["whip"].rolling(window=window, min_periods=1).mean().shift(1)
        k_rate_roll = group["k_rate"].rolling(window=window, min_periods=1).mean().shift(1)

        starter_game[f"starter_era_last{window}"] = 9.0 * er_roll / ip_roll.replace(0, pd.NA)
        starter_game[f"starter_whip_last{window}"] = whip_roll
        starter_game[f"starter_k_rate_last{window}"] = k_rate_roll

    prev_date = group["game_date"].shift(1)
    starter_game["starter_rest_days"] = (starter_game["game_date"] - prev_date).dt.days

    return starter_game[[
        "game_id",
        "game_date",
        "team_mlb_id",
        "is_home",
        "pitcher_mlb_id",
        "starter_kbb_last3",
        "starter_era_last3",
        "starter_era_last5",
        "starter_era_last10",
        "starter_k_rate_last5",
        "starter_k_rate_last10",
        "starter_whip_last5",
        "starter_whip_last10",
        "starter_rest_days",
    ]]


def build_bullpen_prev_innings(pitching_df: pd.DataFrame, starters_df: pd.DataFrame) -> pd.DataFrame:
    if pitching_df.empty:
        return pd.DataFrame()

    pitching = pitching_df.copy()
    pitching["innings_pitched"] = pitching["innings_pitched"].apply(_innings_to_float)

    team_game = (
        pitching
        .groupby(["team_mlb_id", "game_date", "game_id"], as_index=False)
        .agg(total_ip=("innings_pitched", "sum"))
    )

    if starters_df.empty:
        bullpen = team_game.copy()
        bullpen["bullpen_ip"] = bullpen["total_ip"]
    else:
        starter_ip = (
            starters_df.merge(
                pitching[["game_id", "pitcher_mlb_id", "innings_pitched"]],
                on=["game_id", "pitcher_mlb_id"],
                how="left",
            )
            .groupby(["team_mlb_id", "game_date", "game_id"], as_index=False)
            .agg(starter_ip=("innings_pitched", "sum"))
        )
        bullpen = team_game.merge(starter_ip, on=["team_mlb_id", "game_date", "game_id"], how="left")
        bullpen["bullpen_ip"] = bullpen["total_ip"] - bullpen["starter_ip"].fillna(0)

    bullpen = bullpen.sort_values(["team_mlb_id", "game_date", "game_id"])
    bullpen["bullpen_ip_prev"] = bullpen.groupby("team_mlb_id")["bullpen_ip"].shift(1)

    return bullpen[["team_mlb_id", "game_date", "game_id", "bullpen_ip_prev"]]


def fetch_statcast_batter_team_metrics(season: int) -> pd.DataFrame:
    if statcast_batter_exitvelo_barrels is None:
        return pd.DataFrame()
    try:
        df = statcast_batter_exitvelo_barrels(season)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    cols = df.columns
    team_col = _find_col(cols, ["team", "team_name", "team_abbreviation", "team_name_abbrev"])
    ev_col = _find_col(cols, ["avg_hit_speed", "avg_exit_velocity", "avg_hit_speed_2024", "avg_hit_speed_2023"])
    la_col = _find_col(cols, ["avg_launch_angle", "avg_launch_angle_2024", "avg_launch_angle_2023"])
    barrel_col = _find_col(cols, ["barrel_batted_rate", "barrel_pct", "barrel_percent", "brl_percent"])

    if not team_col:
        return pd.DataFrame()

    metric_cols = {
        "team_ev": ev_col,
        "team_la": la_col,
        "team_barrel_rate": barrel_col,
    }
    use_cols = [team_col] + [c for c in metric_cols.values() if c]
    df = df[use_cols].copy()

    for metric, col in metric_cols.items():
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    agg_map = {col: "mean" for col in metric_cols.values() if col}
    team_df = df.groupby(team_col, as_index=False).agg(agg_map)
    rename_map = {team_col: "team_abbrev"}
    for metric, col in metric_cols.items():
        if col and col in team_df.columns:
            rename_map[col] = metric
    team_df = team_df.rename(columns=rename_map)
    return team_df


def fetch_pitcher_statcast_metrics(pitcher_ids: Sequence[int], start_dt: date, end_dt: date) -> pd.DataFrame:
    if statcast_pitcher is None:
        return pd.DataFrame()

    rows: List[Dict] = []
    for pid in sorted({int(p) for p in pitcher_ids if pd.notna(p)}):
        try:
            data = statcast_pitcher(start_dt=start_dt.isoformat(), end_dt=end_dt.isoformat(), player_id=pid)
        except Exception:
            continue
        if data is None or data.empty:
            continue

        data = data.copy()
        desc = data["description"] if "description" in data.columns else pd.Series([], dtype="object")
        events = data["events"] if "events" in data.columns else pd.Series([], dtype="object")
        stand = data["stand"] if "stand" in data.columns else pd.Series([], dtype="object")

        pitches = len(data)
        swstr = desc.isin(["swinging_strike", "swinging_strike_blocked"]).sum() if len(desc) else 0
        swstr_pct = swstr / pitches if pitches else pd.NA

        strikeouts = events.isin(["strikeout", "strikeout_double_play"]).sum() if len(events) else 0
        walks = events.isin(["walk", "intent_walk"]).sum() if len(events) else 0
        pa = events.notna().sum() if len(events) else 0

        k_rate = strikeouts / pa if pa else pd.NA
        bb_rate = walks / pa if pa else pd.NA
        kbb_ratio = strikeouts / walks if walks else pd.NA

        woba_vs_l = pd.NA
        woba_vs_r = pd.NA
        if "woba_value" in data.columns and "woba_denom" in data.columns:
            for side, colname in [("L", "woba_vs_l"), ("R", "woba_vs_r")]:
                side_df = data[stand == side]
                if not side_df.empty:
                    denom = side_df["woba_denom"].sum()
                    val = side_df["woba_value"].sum()
                    if denom:
                        if side == "L":
                            woba_vs_l = val / denom
                        else:
                            woba_vs_r = val / denom

        rows.append(
            {
                "pitcher_mlb_id": pid,
                "pitcher_k_rate": k_rate,
                "pitcher_bb_rate": bb_rate,
                "pitcher_kbb_ratio": kbb_ratio,
                "pitcher_swstr_pct": swstr_pct,
                "pitcher_woba_vs_l": woba_vs_l,
                "pitcher_woba_vs_r": woba_vs_r,
            }
        )

    return pd.DataFrame(rows)


def build_home_away_win_pct_diff(engine, target_date: date) -> Dict:
    sql = text(
        """
        SELECT g.game_date, g.home_team_id, g.away_team_id, r.home_win
          FROM games g
          JOIN game_results r ON g.mlb_game_id = r.game_id
         WHERE g.game_date < :target_date
        """
    )
    df = pd.read_sql(sql, engine, params={"target_date": target_date})
    if df.empty:
        return {}

    df = df.dropna(subset=["home_win"]).copy()
    df["home_win"] = df["home_win"].astype(int)

    home = df[["home_team_id", "home_win"]].rename(columns={"home_team_id": "team_id", "home_win": "win"})
    away = df[["away_team_id", "home_win"]].rename(columns={"away_team_id": "team_id"})
    away["win"] = 1 - away["home_win"]
    away = away[["team_id", "win"]]

    home_stats = home.groupby("team_id")["win"].agg(["sum", "count"]).rename(columns={"sum": "home_wins", "count": "home_games"})
    away_stats = away.groupby("team_id")["win"].agg(["sum", "count"]).rename(columns={"sum": "away_wins", "count": "away_games"})

    stats = home_stats.join(away_stats, how="outer").fillna(0)
    stats["home_win_pct"] = stats["home_wins"] / stats["home_games"].replace(0, pd.NA)
    stats["away_win_pct"] = stats["away_wins"] / stats["away_games"].replace(0, pd.NA)
    stats["win_pct_diff"] = stats["home_win_pct"].fillna(0) - stats["away_win_pct"].fillna(0)

    return stats["win_pct_diff"].to_dict()


def compute_rest_days(team_game_dates: pd.DataFrame, target_date: date) -> Dict[int, Optional[int]]:
    if team_game_dates.empty:
        return {}
    last_dates = (
        team_game_dates[team_game_dates["game_date"] < target_date]
        .groupby("team_mlb_id")["game_date"].max()
    )
    return {team_id: (target_date - last_date).days for team_id, last_date in last_dates.items()}


def build_features(
    target_date: date,
    windows: Sequence[int] = (5, 15, 30),
    ewm_spans: Sequence[int] = (5, 15, 30),
) -> pd.DataFrame:
    engine = get_engine()
    since_date = target_date - timedelta(days=200)

    games = load_games(engine, target_date)
    batting = load_batting(engine, since_date, target_date)
    pitching = load_pitching(engine, since_date, target_date)
    if not batting.empty and "game_date" in batting.columns:
        batting["game_date"] = pd.to_datetime(batting["game_date"]).dt.date
    if not pitching.empty and "game_date" in pitching.columns:
        pitching["game_date"] = pd.to_datetime(pitching["game_date"]).dt.date
    try:
        starters = load_starting_pitchers(engine, since_date, target_date)
    except Exception:
        logging.warning("starting_pitchers table not ready; skipping starter features.")
        starters = pd.DataFrame()

    batting_feat = build_team_batting_features(batting, windows=windows, ewm_spans=ewm_spans)
    pitching_feat = build_team_pitching_features(pitching, windows=windows, ewm_spans=ewm_spans)
    if not batting_feat.empty and "game_date" in batting_feat.columns:
        batting_feat["game_date"] = pd.to_datetime(batting_feat["game_date"]).dt.date
    if not pitching_feat.empty and "game_date" in pitching_feat.columns:
        pitching_feat["game_date"] = pd.to_datetime(pitching_feat["game_date"]).dt.date

    # Use latest available rolling stats per team (pre-game features)
    batting_latest = batting_feat.sort_values("game_date").groupby("team_mlb_id", as_index=False).tail(1)
    batting_latest = batting_latest.drop(columns=["game_date"], errors="ignore")
    pitching_latest = pitching_feat.sort_values("game_date").groupby("team_mlb_id", as_index=False).tail(1)
    pitching_latest = pitching_latest.drop(columns=["game_date"], errors="ignore")

    starter_metrics = build_starter_rolling_features(starters, pitching)
    bullpen_prev = build_bullpen_prev_innings(pitching, starters)

    try:
        win_pct_diff = build_home_away_win_pct_diff(engine, target_date)
    except Exception:
        logging.warning("win pct diff feature unavailable; skipping.")
        win_pct_diff = {}

    team_dates = pd.concat(
        [
            batting[["team_mlb_id", "game_date"]] if not batting.empty else pd.DataFrame(columns=["team_mlb_id", "game_date"]),
            pitching[["team_mlb_id", "game_date"]] if not pitching.empty else pd.DataFrame(columns=["team_mlb_id", "game_date"]),
        ],
        ignore_index=True,
    )
    rest_days = compute_rest_days(team_dates, target_date)

    # Merge features into games
    features = games.copy()
    # Normalize mlb_game_id dtype to avoid merge errors when NULLs exist
    if "mlb_game_id" in features.columns:
        features["mlb_game_id"] = pd.to_numeric(features["mlb_game_id"], errors="coerce")
    features["home_advantage"] = 1

    home_batting = (
        batting_latest.add_prefix("home_")
        .rename(columns={"home_team_mlb_id": "home_team_id"})
    )
    away_batting = (
        batting_latest.add_prefix("away_")
        .rename(columns={"away_team_mlb_id": "away_team_id"})
    )

    home_pitching = (
        pitching_latest.add_prefix("home_")
        .rename(columns={"home_team_mlb_id": "home_team_id"})
    )
    away_pitching = (
        pitching_latest.add_prefix("away_")
        .rename(columns={"away_team_mlb_id": "away_team_id"})
    )

    features = features.merge(
        home_batting,
        how="left",
        on=["home_team_id"],
    )
    features = features.merge(
        away_batting,
        how="left",
        on=["away_team_id"],
    )
    features = features.merge(
        home_pitching,
        how="left",
        on=["home_team_id"],
    )
    features = features.merge(
        away_pitching,
        how="left",
        on=["away_team_id"],
    )

    if not bullpen_prev.empty:
        home_bullpen = bullpen_prev.rename(
            columns={
                "team_mlb_id": "home_team_id",
                "bullpen_ip_prev": "home_bullpen_ip_prev",
                "game_id": "mlb_game_id",
            }
        )[["home_team_id", "mlb_game_id", "home_bullpen_ip_prev"]]
        away_bullpen = bullpen_prev.rename(
            columns={
                "team_mlb_id": "away_team_id",
                "bullpen_ip_prev": "away_bullpen_ip_prev",
                "game_id": "mlb_game_id",
            }
        )[["away_team_id", "mlb_game_id", "away_bullpen_ip_prev"]]
        home_bullpen["mlb_game_id"] = pd.to_numeric(home_bullpen["mlb_game_id"], errors="coerce")
        away_bullpen["mlb_game_id"] = pd.to_numeric(away_bullpen["mlb_game_id"], errors="coerce")
        features = features.merge(
            home_bullpen,
            how="left",
            on=["home_team_id", "mlb_game_id"],
        )
        features = features.merge(
            away_bullpen,
            how="left",
            on=["away_team_id", "mlb_game_id"],
        )

    if not starter_metrics.empty:
        starter_today = starter_metrics[starter_metrics["game_date"] == pd.to_datetime(target_date)]
        home_starters = starter_today[starter_today["is_home"]].rename(
            columns={
                "starter_kbb_last3": "home_starter_kbb_last3",
                "starter_era_last3": "home_starter_era_last3",
                "starter_era_last5": "home_starter_era_last5",
                "starter_era_last10": "home_starter_era_last10",
                "starter_k_rate_last5": "home_starter_k_rate_last5",
                "starter_k_rate_last10": "home_starter_k_rate_last10",
                "starter_whip_last5": "home_starter_whip_last5",
                "starter_whip_last10": "home_starter_whip_last10",
                "starter_rest_days": "home_starter_rest_days",
                "pitcher_mlb_id": "home_pitcher_mlb_id",
                "game_id": "mlb_game_id",
            }
        )[[
            "mlb_game_id",
            "home_pitcher_mlb_id",
            "home_starter_kbb_last3",
            "home_starter_era_last3",
            "home_starter_era_last5",
            "home_starter_era_last10",
            "home_starter_k_rate_last5",
            "home_starter_k_rate_last10",
            "home_starter_whip_last5",
            "home_starter_whip_last10",
            "home_starter_rest_days",
        ]]
        away_starters = starter_today[~starter_today["is_home"]].rename(
            columns={
                "starter_kbb_last3": "away_starter_kbb_last3",
                "starter_era_last3": "away_starter_era_last3",
                "starter_era_last5": "away_starter_era_last5",
                "starter_era_last10": "away_starter_era_last10",
                "starter_k_rate_last5": "away_starter_k_rate_last5",
                "starter_k_rate_last10": "away_starter_k_rate_last10",
                "starter_whip_last5": "away_starter_whip_last5",
                "starter_whip_last10": "away_starter_whip_last10",
                "starter_rest_days": "away_starter_rest_days",
                "pitcher_mlb_id": "away_pitcher_mlb_id",
                "game_id": "mlb_game_id",
            }
        )[[
            "mlb_game_id",
            "away_pitcher_mlb_id",
            "away_starter_kbb_last3",
            "away_starter_era_last3",
            "away_starter_era_last5",
            "away_starter_era_last10",
            "away_starter_k_rate_last5",
            "away_starter_k_rate_last10",
            "away_starter_whip_last5",
            "away_starter_whip_last10",
            "away_starter_rest_days",
        ]]
        home_starters["mlb_game_id"] = pd.to_numeric(home_starters["mlb_game_id"], errors="coerce")
        away_starters["mlb_game_id"] = pd.to_numeric(away_starters["mlb_game_id"], errors="coerce")
        features = features.merge(
            home_starters,
            how="left",
            on=["mlb_game_id"],
        )
        features = features.merge(
            away_starters,
            how="left",
            on=["mlb_game_id"],
        )

    features["home_rest_days"] = features["home_team_id"].map(rest_days)
    features["away_rest_days"] = features["away_team_id"].map(rest_days)
    features["home_win_pct_diff"] = features["home_team_id"].map(win_pct_diff)
    features["away_win_pct_diff"] = features["away_team_id"].map(win_pct_diff)

    season = target_date.year

    # Statcast batter EV/LA/Barrel% (team-level)
    try:
        team_abbrev = load_team_abbrev_map(engine)
        batter_team = fetch_statcast_batter_team_metrics(season)
        if not team_abbrev.empty and not batter_team.empty:
            abbrev_map = dict(zip(team_abbrev["mlb_team_id"], team_abbrev["abbreviation"]))
            features["home_team_abbrev"] = features["home_team_id"].map(abbrev_map)
            features["away_team_abbrev"] = features["away_team_id"].map(abbrev_map)
            batter_map = batter_team.set_index("team_abbrev")
            for metric in ["team_ev", "team_la", "team_barrel_rate"]:
                if metric in batter_map.columns:
                    metric_dict = batter_map[metric].to_dict()
                    features[f"home_{metric}"] = features["home_team_abbrev"].map(metric_dict)
                    features[f"away_{metric}"] = features["away_team_abbrev"].map(metric_dict)
    except Exception:
        logging.warning("Statcast batter data unavailable; skipping.")

    # Statcast pitcher K/BB%, SwStr%, vs L/R matchup
    try:
        if not starter_metrics.empty:
            starter_today = starter_metrics[starter_metrics["game_date"] == pd.to_datetime(target_date)]
            pitcher_ids = list(starter_today["pitcher_mlb_id"].dropna().unique())
            if pitcher_ids:
                start_dt = target_date - timedelta(days=45)
                pitcher_metrics = fetch_pitcher_statcast_metrics(pitcher_ids, start_dt, target_date)
                if not pitcher_metrics.empty:
                    pmap = pitcher_metrics.set_index("pitcher_mlb_id")
                    for metric in [
                        "pitcher_k_rate",
                        "pitcher_bb_rate",
                        "pitcher_kbb_ratio",
                        "pitcher_swstr_pct",
                        "pitcher_woba_vs_l",
                        "pitcher_woba_vs_r",
                    ]:
                        if metric in pmap.columns:
                            metric_dict = pmap[metric].to_dict()
                            features[f"home_{metric}"] = features["home_pitcher_mlb_id"].map(metric_dict)
                            features[f"away_{metric}"] = features["away_pitcher_mlb_id"].map(metric_dict)
    except Exception:
        logging.warning("Statcast pitcher data unavailable; skipping.")

    # Platoon splits (pybaseball / Baseball Reference)
    try:
        platoon_dir = Path(__file__).resolve().parent / "data" / "pybaseball"
        platoon_df = load_platoon_splits_csv(str(platoon_dir), season)
        if not platoon_df.empty:
            features, _ = _apply_platoon_splits_features(features, platoon_df)
    except Exception:
        logging.warning("Platoon splits data unavailable; skipping.")

    # pybaseball integration (B-layer on-demand補值)
    try:
        features = attach_fangraphs_features(features, engine, season)
    except Exception:
        logging.warning("pybaseball data unavailable; skipping.")

    # Taiwan Sports Lottery moneyline odds (for model features)
    try:
        odds = load_moneyline_odds(engine, target_date)
        if not odds.empty:
            odds = odds.copy()
            if "sportsbook" in odds.columns:
                preferred = odds[odds["sportsbook"].isin({"taiwan_sports_lottery", "taiwan_lottery", "sportslottery", "jbot"})]
                if not preferred.empty:
                    odds = preferred

            if taiwan_lottery_crawler is not None:
                team_map = taiwan_lottery_crawler.load_team_name_map()
            else:
                team_map = {}

            def _normalize_selection(row):
                sel = str(row.get("selection") or "").strip().lower()
                if sel in {"home", "h", "主"}:
                    return "home"
                if sel in {"away", "a", "客"}:
                    return "away"
                if taiwan_lottery_crawler is None:
                    return None
                home = taiwan_lottery_crawler.normalize_team_name(row.get("home_team_name"), team_map).lower()
                away = taiwan_lottery_crawler.normalize_team_name(row.get("away_team_name"), team_map).lower()
                sel_name = taiwan_lottery_crawler.normalize_team_name(row.get("selection"), team_map).lower()
                if sel_name == home:
                    return "home"
                if sel_name == away:
                    return "away"
                return None

            odds["selection_norm"] = odds.apply(_normalize_selection, axis=1)
            odds = odds.dropna(subset=["selection_norm"])
            odds = odds.sort_values("retrieved_at")
            home_price = odds[odds["selection_norm"] == "home"].groupby("game_id")["price"].last()
            away_price = odds[odds["selection_norm"] == "away"].groupby("game_id")["price"].last()

            features["home_price"] = features["game_id"].map(home_price)
            features["away_price"] = features["game_id"].map(away_price)
            features["tw_home_ml_odds"] = features["home_price"]
            features["tw_away_ml_odds"] = features["away_price"]
    except Exception:
        logging.warning("Taiwan odds data unavailable; skipping.")

    # Bullpen fatigue integration
    try:
        team_fatigue = pd.DataFrame()
        if table_exists(engine, "bullpen_fatigue"):
            team_fatigue = load_bullpen_fatigue(engine, target_date)
        if team_fatigue.empty:
            from bullpen_fatigue import compute_team_fatigue

            team_fatigue, _ = compute_team_fatigue(engine, target_date, window_days=5)
        if not team_fatigue.empty:
            fatigue_map = team_fatigue.set_index("team_mlb_id")["bullpen_fatigue_index"].to_dict()
            features["home_bullpen_fatigue_index"] = features["home_team_id"].map(fatigue_map)
            features["away_bullpen_fatigue_index"] = features["away_team_id"].map(fatigue_map)
    except Exception:
        logging.warning("Bullpen fatigue data unavailable; skipping.")

    # Weather integration
    try:
        weather = pd.DataFrame()
        if table_exists(engine, "game_weather"):
            weather = load_game_weather(engine, target_date)
        if not weather.empty:
            for col in [
                "weather_temperature_c",
                "weather_relative_humidity",
                "weather_wind_speed",
                "weather_wind_direction",
            ]:
                if col in weather.columns:
                    weather[col] = pd.to_numeric(weather[col], errors="coerce")
            features = features.merge(weather, how="left", on=["mlb_game_id"])
            if "weather_wind_direction" in features.columns:
                rad = np.deg2rad(features["weather_wind_direction"])
                features["weather_wind_dir_sin"] = np.sin(rad)
                features["weather_wind_dir_cos"] = np.cos(rad)
    except Exception:
        logging.warning("Weather data unavailable; skipping.")

    # Add labels if results exist
    try:
        results = load_results(engine, target_date)
        if not results.empty:
            features = features.merge(results, how="left", on="mlb_game_id")
    except Exception:
        logging.warning("Results table not ready; skipping labels.")

    if "home_score" in features.columns and "away_score" in features.columns:
        features["home_score"] = pd.to_numeric(features["home_score"], errors="coerce")
        features["away_score"] = pd.to_numeric(features["away_score"], errors="coerce")
        features["run_margin"] = features["home_score"] - features["away_score"]

    if "run_margin" in features.columns:
        if "home_win" not in features.columns:
            features["home_win"] = pd.NA
        home_win_mask = features["home_win"].isna() & features["run_margin"].notna()
        if home_win_mask.any():
            features.loc[home_win_mask, "home_win"] = (features.loc[home_win_mask, "run_margin"] > 0).astype("int")

    # Placeholder: Taiwan lottery & international spread lines + diff
    if "tw_home_spread" not in features.columns:
        features["tw_home_spread"] = pd.NA
    if "intl_home_spread" not in features.columns:
        features["intl_home_spread"] = pd.NA
    features["tw_intl_spread_diff"] = features["tw_home_spread"] - features["intl_home_spread"]

    if "run_margin" in features.columns:
        if "home_spread_line" in features.columns:
            spread_line = features["home_spread_line"]
        elif "tw_home_spread" in features.columns:
            spread_line = features["tw_home_spread"]
        else:
            spread_line = None

        if spread_line is not None:
            cover = pd.Series(np.nan, index=features.index, dtype="float")
            valid_mask = spread_line.notna() & features["run_margin"].notna()
            if valid_mask.any():
                cover.loc[valid_mask] = (
                    features.loc[valid_mask, "run_margin"] + spread_line[valid_mask] > 0
                ).astype("float")
            features["cover_spread"] = cover
        else:
            features["cover_spread"] = pd.NA

    # Derive model target (prefer cover_spread, fallback to home_win)
    if "target" not in features.columns:
        target = pd.Series(pd.NA, index=features.index)
        if "cover_spread" in features.columns:
            target = features["cover_spread"].copy()
        if "home_win" in features.columns:
            target = target.where(target.notna(), features["home_win"])
        features["target"] = target

    # Ensure optional/label columns exist for schema consistency
    optional_cols = [
        "home_score",
        "away_score",
        "home_win",
        "total_points",
        "run_margin",
        "cover_spread",
        "target",
        "home_fangraphs_wrc_plus",
        "away_fangraphs_wrc_plus",
        "home_fangraphs_woba",
        "away_fangraphs_woba",
        "home_fangraphs_xwoba",
        "away_fangraphs_xwoba",
        "home_fangraphs_ops_plus",
        "away_fangraphs_ops_plus",
        "home_starter_fip",
        "home_starter_xfip",
        "home_starter_k_per_9",
        "home_starter_bb_per_9",
        "away_starter_fip",
        "away_starter_xfip",
        "away_starter_k_per_9",
        "away_starter_bb_per_9",
        "home_bullpen_fatigue_index",
        "away_bullpen_fatigue_index",
        "home_bullpen_ip_prev",
        "away_bullpen_ip_prev",
        "home_pitcher_mlb_id",
        "away_pitcher_mlb_id",
        "home_starter_kbb_last3",
        "home_starter_era_last3",
        "home_starter_era_last5",
        "home_starter_era_last10",
        "home_starter_k_rate_last5",
        "home_starter_k_rate_last10",
        "home_starter_whip_last5",
        "home_starter_whip_last10",
        "home_starter_rest_days",
        "away_starter_kbb_last3",
        "away_starter_era_last3",
        "away_starter_era_last5",
        "away_starter_era_last10",
        "away_starter_k_rate_last5",
        "away_starter_k_rate_last10",
        "away_starter_whip_last5",
        "away_starter_whip_last10",
        "away_starter_rest_days",
        "home_team_ev",
        "away_team_ev",
        "home_team_la",
        "away_team_la",
        "home_team_barrel_rate",
        "away_team_barrel_rate",
        "home_pitcher_k_rate",
        "away_pitcher_k_rate",
        "home_pitcher_bb_rate",
        "away_pitcher_bb_rate",
        "home_pitcher_kbb_ratio",
        "away_pitcher_kbb_ratio",
        "home_pitcher_swstr_pct",
        "away_pitcher_swstr_pct",
        "home_pitcher_woba_vs_l",
        "home_pitcher_woba_vs_r",
        "away_pitcher_woba_vs_l",
        "away_pitcher_woba_vs_r",
        "home_platoon_ba_diff",
        "away_platoon_ba_diff",
        "home_platoon_ops_diff",
        "away_platoon_ops_diff",
        "home_platoon_k_rate_lhb",
        "away_platoon_k_rate_lhb",
        "home_platoon_k_rate_rhb",
        "away_platoon_k_rate_rhb",
        "home_platoon_splits_score",
        "away_platoon_splits_score",
        "home_price",
        "away_price",
        "tw_home_ml_odds",
        "tw_away_ml_odds",
        "weather_temperature_c",
        "weather_relative_humidity",
        "weather_wind_speed",
        "weather_wind_direction",
        "weather_wind_dir_sin",
        "weather_wind_dir_cos",
    ]
    for col in optional_cols:
        if col not in features.columns:
            features[col] = pd.NA

    return features


def _get_table_columns(engine, table_name: str) -> List[str]:
    sql = text(
        """
        SELECT column_name
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = :table_name
        """
    )
    rows = pd.read_sql(sql, engine, params={"table_name": table_name})
    return rows["column_name"].tolist()


def write_features_to_db(engine, df: pd.DataFrame, table_name: str = "model_features"):
    if df.empty:
        logging.warning("No features to write.")
        return
    table_cols = _get_table_columns(engine, table_name)
    if table_cols:
        df = df[[c for c in df.columns if c in table_cols]]
    df.to_sql(table_name, engine, if_exists="append", index=False)
    logging.info("Inserted %d rows into %s", len(df), table_name)


# ---------------------
# Historical (CSV / pybaseball)
# ---------------------

def load_pybaseball_games(data_dir: str, seasons: Sequence[int]) -> pd.DataFrame:
    """Load schedule from starting_pitchers CSVs and attach results if available."""
    schedule_paths = []
    for season in seasons:
        path = Path(data_dir) / f"starting_pitchers_{season}.csv"
        if path.exists():
            schedule_paths.append(path)
    if not schedule_paths:
        merged = Path(data_dir) / "starting_pitchers_2023_2025.csv"
        if merged.exists():
            schedule_paths.append(merged)
    if not schedule_paths:
        raise FileNotFoundError(f"No starting_pitchers CSV found in {data_dir}")

    schedule_frames = [pd.read_csv(p) for p in schedule_paths]
    games = pd.concat(schedule_frames, ignore_index=True)
    if "game_date" in games.columns:
        games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    for col in ["home_team", "away_team"]:
        if col in games.columns:
            games[col] = games[col].astype(str)
    for col in ["home_pitcher_mlbam", "away_pitcher_mlbam"]:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce")
    if "season" in games.columns:
        games["season"] = pd.to_numeric(games.get("season"), errors="coerce")

    # Attach results (home/away score + win) from games CSVs when present
    result_paths = []
    for season in seasons:
        path = Path(data_dir) / f"games_{season}.csv"
        if path.exists():
            result_paths.append(path)
    if not result_paths:
        merged = Path(data_dir) / "games_2023_2025.csv"
        if merged.exists():
            result_paths.append(merged)

    if result_paths:
        result_frames = [pd.read_csv(p) for p in result_paths]
        results = pd.concat(result_frames, ignore_index=True)
        if "game_date" in results.columns:
            results["game_date"] = pd.to_datetime(results["game_date"]).dt.date
        for col in ["home_team", "away_team"]:
            if col in results.columns:
                results[col] = results[col].astype(str)
        if "season" in results.columns:
            results["season"] = pd.to_numeric(results.get("season"), errors="coerce")

        join_keys = ["game_date", "home_team", "away_team"]
        if "season" in games.columns and "season" in results.columns:
            join_keys.append("season")
        result_cols = [c for c in ["home_score", "away_score", "home_win"] if c in results.columns]
        if result_cols:
            results = results[join_keys + result_cols].drop_duplicates()
            games = games.merge(results, how="left", on=join_keys)

    # Ensure label columns exist (may be NA if no results available)
    for col in ["home_score", "away_score", "home_win"]:
        if col not in games.columns:
            games[col] = pd.NA

    return games


def load_pybaseball_team_stats(data_dir: str, seasons: Sequence[int], kind: str) -> pd.DataFrame:
    paths = []
    for season in seasons:
        path = Path(data_dir) / f"team_{kind}_{season}.csv"
        if path.exists():
            paths.append(path)
    if not paths:
        merged = Path(data_dir) / f"team_{kind}_2023_2025.csv"
        if merged.exists():
            paths.append(merged)
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "team" in df.columns:
        df["team"] = df["team"].astype(str)
    return df




def load_pybaseball_pitcher_stats(data_dir: str, seasons: Sequence[int]) -> pd.DataFrame:
    paths = []
    for season in seasons:
        path = Path(data_dir) / f"pitcher_stats_{season}.csv"
        if path.exists():
            paths.append(path)
    if not paths:
        merged = Path(data_dir) / "pitcher_stats_2023_2025.csv"
        if merged.exists():
            paths.append(merged)
    if not paths:
        return pd.DataFrame()

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "Season" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"Season": "season"})
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "mlbam_id" in df.columns:
        df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce")
    return df


def load_pybaseball_platoon_splits(data_dir: str, seasons: Sequence[int]) -> pd.DataFrame:
    paths = []
    for season in seasons:
        path = Path(data_dir) / f"platoon_splits_{season}.csv"
        if path.exists():
            paths.append(path)
    if not paths:
        return pd.DataFrame()

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    if "mlbam_id" in df.columns:
        df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce")
    return df


def load_pybaseball_starting_pitchers(data_dir: str, seasons: Sequence[int]) -> pd.DataFrame:
    paths = []
    for season in seasons:
        path = Path(data_dir) / f"starting_pitchers_{season}.csv"
        if path.exists():
            paths.append(path)
    if not paths:
        merged = Path(data_dir) / "starting_pitchers_2023_2025.csv"
        if merged.exists():
            paths.append(merged)
    if not paths:
        return pd.DataFrame()

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in ["home_pitcher_mlbam", "away_pitcher_mlbam"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    return df


def _build_team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    games["game_key"] = (
        games["game_date"].astype(str)
        + "_"
        + games["home_team"].astype(str)
        + "_"
        + games["away_team"].astype(str)
    )

    home = games[[
        "game_key",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "season",
    ]].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["runs_scored"] = pd.to_numeric(home["home_score"], errors="coerce")
    home["runs_allowed"] = pd.to_numeric(home["away_score"], errors="coerce")
    home["win"] = pd.to_numeric(home["home_win"], errors="coerce")
    home["is_home"] = True

    away = games[[
        "game_key",
        "game_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "home_win",
        "season",
    ]].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["runs_scored"] = pd.to_numeric(away["away_score"], errors="coerce")
    away["runs_allowed"] = pd.to_numeric(away["home_score"], errors="coerce")
    away["win"] = 1 - pd.to_numeric(away["home_win"], errors="coerce")
    away["is_home"] = False

    team_game = pd.concat([home, away], ignore_index=True)
    team_game["run_diff"] = team_game["runs_scored"] - team_game["runs_allowed"]
    team_game = team_game.sort_values(["team", "game_date", "game_key"]).reset_index(drop=True)
    return team_game


def _rolling_team_features(team_game: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    metrics = ["runs_scored", "runs_allowed", "run_diff", "win"]
    base = team_game[["game_key", "team", "game_date", "is_home"]].copy()
    grouped = team_game.groupby("team", group_keys=False)

    for window in windows:
        for col in metrics:
            rolled_mean = (
                grouped[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
            rolled_sum = (
                grouped[col]
                .rolling(window=window, min_periods=1)
                .sum()
                .shift(1)
                .reset_index(level=0, drop=True)
            )
            base[f"roll{window}_{col}_mean"] = rolled_mean.values
            base[f"roll{window}_{col}_sum"] = rolled_sum.values

    # home/away split rolling (win + runs)
    grouped_ha = team_game.groupby(["team", "is_home"], group_keys=False)
    for window in windows:
        for col in ["runs_scored", "runs_allowed", "win"]:
            rolled = (
                grouped_ha[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .shift(1)
                .reset_index(level=[0, 1], drop=True)
            )
            base[f"ha_roll{window}_{col}_mean"] = rolled.values

    # rest days
    prev_date = grouped["game_date"].shift(1)
    base["rest_days"] = (pd.to_datetime(base["game_date"]) - pd.to_datetime(prev_date)).dt.days

    # bullpen burden proxy: previous game runs allowed + last3 mean
    base["bullpen_prev_runs_allowed"] = grouped["runs_allowed"].shift(1).values
    base["bullpen_ra_last3"] = (
        grouped["runs_allowed"].rolling(window=3, min_periods=1).mean().shift(1).reset_index(level=0, drop=True)
    ).values

    return base


def build_historical_features_from_csv(
    data_dir: str,
    seasons: Sequence[int],
    windows: Sequence[int] = (3, 5, 10, 20, 30),
) -> pd.DataFrame:
    games = load_pybaseball_games(data_dir, seasons)
    games = games.dropna(subset=["home_team", "away_team", "game_date"]).copy()
    games["season"] = pd.to_numeric(games.get("season"), errors="coerce")

    team_game = _build_team_game_log(games)
    team_features = _rolling_team_features(team_game, windows=windows)

    # Merge team-level rolling stats back to game rows
    home_feat = team_features[team_features["is_home"]].copy()
    away_feat = team_features[~team_features["is_home"]].copy()

    home_feat = home_feat.drop(columns=["is_home"], errors="ignore").rename(
        columns={c: f"home_{c}" for c in home_feat.columns if c not in {"game_key", "team", "game_date"}}
    )
    away_feat = away_feat.drop(columns=["is_home"], errors="ignore").rename(
        columns={c: f"away_{c}" for c in away_feat.columns if c not in {"game_key", "team", "game_date"}}
    )

    games = games.copy()
    games["game_key"] = (
        games["game_date"].astype(str)
        + "_"
        + games["home_team"].astype(str)
        + "_"
        + games["away_team"].astype(str)
    )

    games = games.merge(home_feat, how="left", left_on=["game_key", "home_team"], right_on=["game_key", "team"]).drop(columns=["team"], errors="ignore")
    games = games.merge(away_feat, how="left", left_on=["game_key", "away_team"], right_on=["game_key", "team"]).drop(columns=["team"], errors="ignore")

    # Team season stats (batting/pitching)
    team_batting = load_pybaseball_team_stats(data_dir, seasons, kind="batting")
    team_pitching = load_pybaseball_team_stats(data_dir, seasons, kind="pitching")

    if not team_batting.empty:
        home_bat = team_batting.rename(columns={"team": "home_team"}).copy()
        away_bat = team_batting.rename(columns={"team": "away_team"}).copy()
        home_bat = home_bat.rename(
            columns={c: f"home_bat_{c}" for c in home_bat.columns if c not in {"home_team", "season"}}
        )
        away_bat = away_bat.rename(
            columns={c: f"away_bat_{c}" for c in away_bat.columns if c not in {"away_team", "season"}}
        )
        games = games.merge(home_bat, how="left", on=["home_team", "season"])
        games = games.merge(away_bat, how="left", on=["away_team", "season"])

    if not team_pitching.empty:
        home_pit = team_pitching.rename(columns={"team": "home_team"}).copy()
        away_pit = team_pitching.rename(columns={"team": "away_team"}).copy()
        home_pit = home_pit.rename(
            columns={c: f"home_pit_{c}" for c in home_pit.columns if c not in {"home_team", "season"}}
        )
        away_pit = away_pit.rename(
            columns={c: f"away_pit_{c}" for c in away_pit.columns if c not in {"away_team", "season"}}
        )
        games = games.merge(home_pit, how="left", on=["home_team", "season"])
        games = games.merge(away_pit, how="left", on=["away_team", "season"])

    # Starter pitcher season stats (pybaseball)
    if "home_pitcher_mlbam" not in games.columns or "away_pitcher_mlbam" not in games.columns:
        starters = load_pybaseball_starting_pitchers(data_dir, seasons)
        if not starters.empty:
            starters = starters[[
                "game_date",
                "home_team",
                "away_team",
                "home_pitcher_mlbam",
                "away_pitcher_mlbam",
            ]].copy()
            games = games.merge(starters, how="left", on=["game_date", "home_team", "away_team"])

    pitcher_stats = load_pybaseball_pitcher_stats(data_dir, seasons)
    if not pitcher_stats.empty:
        pit = pitcher_stats.copy()
        rename_map = {
            "ERA": "era",
            "FIP": "fip",
            "xFIP": "xfip",
            "SIERA": "siera",
            "WHIP": "whip",
            "K%": "k_pct",
            "BB%": "bb_pct",
            "K-BB%": "kbb_pct",
            "WAR": "war",
            "IP": "ip",
        }
        pit.rename(columns={k: v for k, v in rename_map.items() if k in pit.columns}, inplace=True)
        pit["season"] = pd.to_numeric(pit.get("season"), errors="coerce")
        pit["mlbam_id"] = pd.to_numeric(pit.get("mlbam_id"), errors="coerce")
        stat_cols = [c for c in rename_map.values() if c in pit.columns]
        for col in stat_cols:
            pit[col] = pd.to_numeric(pit[col], errors="coerce")

        if "home_pitcher_mlbam" in games.columns:
            home_pit = pit[["season", "mlbam_id"] + stat_cols].copy()
            home_pit = home_pit.rename(columns={c: f"home_starter_{c}" for c in stat_cols})
            games = games.merge(
                home_pit,
                how="left",
                left_on=["season", "home_pitcher_mlbam"],
                right_on=["season", "mlbam_id"],
            ).drop(columns=["mlbam_id"], errors="ignore")

        if "away_pitcher_mlbam" in games.columns:
            away_pit = pit[["season", "mlbam_id"] + stat_cols].copy()
            away_pit = away_pit.rename(columns={c: f"away_starter_{c}" for c in stat_cols})
            games = games.merge(
                away_pit,
                how="left",
                left_on=["season", "away_pitcher_mlbam"],
                right_on=["season", "mlbam_id"],
            ).drop(columns=["mlbam_id"], errors="ignore")

    platoon = load_pybaseball_platoon_splits(data_dir, seasons)
    if not platoon.empty:
        required_cols = [
            "platoon_ba_diff",
            "platoon_ops_diff",
            "platoon_k_rate_lhb",
            "platoon_k_rate_rhb",
            "platoon_splits_score",
        ]
        avail_cols = [c for c in required_cols if c in platoon.columns]
        if avail_cols:
            for col in avail_cols:
                platoon[col] = pd.to_numeric(platoon[col], errors="coerce")

            if "home_pitcher_mlbam" in games.columns:
                home_platoon = platoon[["season", "mlbam_id"] + avail_cols].copy()
                home_platoon = home_platoon.rename(
                    columns={
                        "mlbam_id": "home_pitcher_mlbam",
                        **{c: f"home_{c}" for c in avail_cols},
                    }
                )
                games = games.merge(home_platoon, how="left", on=["season", "home_pitcher_mlbam"])

            if "away_pitcher_mlbam" in games.columns:
                away_platoon = platoon[["season", "mlbam_id"] + avail_cols].copy()
                away_platoon = away_platoon.rename(
                    columns={
                        "mlbam_id": "away_pitcher_mlbam",
                        **{c: f"away_{c}" for c in avail_cols},
                    }
                )
                games = games.merge(away_platoon, how="left", on=["season", "away_pitcher_mlbam"])

    # Derived diffs
    diff_cols = []
    for col in games.columns:
        if col.startswith("home_") and col.replace("home_", "away_") in games.columns:
            base = col.replace("home_", "")
            diff_col = f"diff_{base}"
            games[diff_col] = pd.to_numeric(games[col], errors="coerce") - pd.to_numeric(games[col.replace("home_", "away_")], errors="coerce")
            diff_cols.append(diff_col)

    # Labels
    games["home_score"] = pd.to_numeric(games.get("home_score"), errors="coerce")
    games["away_score"] = pd.to_numeric(games.get("away_score"), errors="coerce")
    games["run_margin"] = games["home_score"] - games["away_score"]
    games["home_win"] = pd.to_numeric(games.get("home_win"), errors="coerce")

    return games


# ---------------------
# CLI
# ---------------------

def _parse_int_list(raw: Optional[str], fallback: Sequence[int]) -> Sequence[int]:
    if not raw:
        return fallback
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--date", help="YYYY-MM-DD (default: today)")
    p.add_argument("--window", type=int, help="(Deprecated) Single rolling window size")
    p.add_argument("--windows", type=str, help="Comma-separated windows (default: 5,15,30)")
    p.add_argument("--ewm-spans", type=str, help="Comma-separated EWMA spans (default: 5,15,30)")
    p.add_argument("--out", help="Optional CSV output path")
    p.add_argument("--write-db", action="store_true", help="Write features to model_features table")
    p.add_argument("--historical", action="store_true", help="Build historical features from pybaseball CSV")
    p.add_argument("--data-dir", default="./data/pybaseball", help="pybaseball CSV directory")
    p.add_argument("--seasons", default="2023-2025", help="Season range for historical features")
    return p.parse_args()


def main():
    args = parse_args()

    windows = _parse_int_list(args.windows, fallback=(5, 15, 30))
    ewm_spans = _parse_int_list(args.ewm_spans, fallback=(5, 15, 30))
    if args.window:
        windows = (args.window,)

    if args.historical:
        try:
            start_year, end_year = [int(x) for x in args.seasons.split("-")]
            seasons = list(range(start_year, end_year + 1))
        except ValueError:
            seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
        df = build_historical_features_from_csv(args.data_dir, seasons, windows=windows)
        logging.info("Built %d historical feature rows", len(df))
        if args.out:
            df.to_csv(args.out, index=False)
            logging.info("Saved historical features to %s", args.out)
        else:
            default_out = Path(args.data_dir) / "historical_features.csv"
            df.to_csv(default_out, index=False)
            logging.info("Saved historical features to %s", default_out)
        return

    target_date = date.fromisoformat(args.date) if args.date else date.today()
    df = build_features(target_date, windows=windows, ewm_spans=ewm_spans)
    logging.info("Built %d feature rows for %s", len(df), target_date)

    if args.out:
        df.to_csv(args.out, index=False)
        logging.info("Saved features to %s", args.out)

    if args.write_db:
        engine = get_engine()
        write_features_to_db(engine, df)


if __name__ == "__main__":
    main()
