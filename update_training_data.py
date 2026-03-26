#!/usr/bin/env python3
"""Update training data with completed MLB game results.

Workflow:
- Read data/results/*.json
- Map team names -> abbreviations
- Attach starting pitchers from pybaseball CSV
- Update platoon splits (if missing)
- Build v8 feature set (33 features) + labels
- Append new rows to training CSV
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pybaseball import team_ids
except Exception:  # pragma: no cover
    team_ids = None

import feature_builder
import fangraphs_platoon_splits_crawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

V8_FEATURES = [
    "home_pitcher_mlbam",
    "away_pitcher_mlbam",
    "season",
    "home_p_ERA",
    "home_p_WHIP",
    "home_p_K%",
    "home_p_BB%",
    "home_p_K-BB%",
    "home_p_FIP",
    "home_p_xFIP",
    "home_p_SIERA",
    "home_p_WAR",
    "home_p_IP",
    "away_p_ERA",
    "away_p_WHIP",
    "away_p_K%",
    "away_p_BB%",
    "away_p_K-BB%",
    "away_p_FIP",
    "away_p_xFIP",
    "away_p_SIERA",
    "away_p_WAR",
    "away_p_IP",
    "home_platoon_ba_diff",
    "home_platoon_ops_diff",
    "home_platoon_k_rate_lhb",
    "home_platoon_k_rate_rhb",
    "away_platoon_ba_diff",
    "away_platoon_ops_diff",
    "away_platoon_k_rate_lhb",
    "away_platoon_k_rate_rhb",
    "home_platoon_splits_score",
    "away_platoon_splits_score",
]


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
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

TEAM_ALIAS = {
    "Los Angeles Angels of Anaheim": "Los Angeles Angels",
    "Tampa Bay Devil Rays": "Tampa Bay Rays",
    "Cleveland Indians": "Cleveland Guardians",
    "Arizona D-backs": "Arizona Diamondbacks",
    "LA Dodgers": "Los Angeles Dodgers",
    "LA Angels": "Los Angeles Angels",
}


def _normalize_team_name(name: str) -> str:
    if not name:
        return name
    name = name.strip()
    return TEAM_ALIAS.get(name, name)


def load_team_map() -> Dict[str, str]:
    mapping = dict(TEAM_NAME_TO_ABBR)
    if team_ids is None:
        return mapping
    try:
        df = team_ids()
        name_col = None
        abbr_col = None
        for cand in ["name", "teamName", "team_name", "team", "Team"]:
            if cand in df.columns:
                name_col = cand
                break
        for cand in ["abbrev", "abbreviation", "team_abbrev", "teamAbbrev", "Abbrev", "abbrev_1"]:
            if cand in df.columns:
                abbr_col = cand
                break
        if name_col and abbr_col:
            for _, row in df.iterrows():
                name = str(row.get(name_col, "")).strip()
                abbr = str(row.get(abbr_col, "")).strip()
                if name and abbr:
                    mapping[name] = abbr
    except Exception:
        pass
    return mapping


def _parse_results_item(item: Dict) -> Optional[Dict]:
    if not item:
        return None

    # Saved format from fetch_results.py
    if "home_score" in item and "away_score" in item:
        if not item.get("completed", True):
            return None
        return {
            "game_date": item.get("game_date"),
            "home_team": item.get("home_team"),
            "away_team": item.get("away_team"),
            "home_runs": item.get("home_score"),
            "away_runs": item.get("away_score"),
        }

    # Raw Odds API scores format
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
        if score.get("name") == home_team:
            home_score = score.get("score")
        elif score.get("name") == away_team:
            away_score = score.get("score")

    if home_score is None or away_score is None:
        return None

    game_date = None
    for cand in ["commence_time", "last_update"]:
        if item.get(cand):
            game_date = str(item.get(cand))[:10]
            break

    return {
        "game_date": game_date,
        "home_team": home_team,
        "away_team": away_team,
        "home_runs": home_score,
        "away_runs": away_score,
    }


def load_results_files(paths: Sequence[Path]) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            logging.warning("Failed to read %s: %s", path, exc)
            continue

        for item in payload or []:
            parsed = _parse_results_item(item)
            if parsed:
                rows.append(parsed)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["game_date", "home_team", "away_team", "home_runs", "away_runs"])
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
    df = df.dropna(subset=["game_date"])
    df["home_runs"] = pd.to_numeric(df["home_runs"], errors="coerce")
    df["away_runs"] = pd.to_numeric(df["away_runs"], errors="coerce")
    df = df.dropna(subset=["home_runs", "away_runs"])
    df["home_win"] = (df["home_runs"] > df["away_runs"]).astype(int)
    return df


def _resolve_team_abbr(df: pd.DataFrame) -> pd.DataFrame:
    team_map = load_team_map()
    df = df.copy()
    df["home_team"] = df["home_team"].apply(_normalize_team_name).map(team_map)
    df["away_team"] = df["away_team"].apply(_normalize_team_name).map(team_map)
    return df.dropna(subset=["home_team", "away_team"])


def _load_pitcher_stats(py_dir: Path, seasons: Sequence[int]) -> pd.DataFrame:
    paths = []
    for season in seasons:
        p = py_dir / f"pitcher_stats_{season}.csv"
        if p.exists():
            paths.append(p)
    if not paths:
        merged = py_dir / "pitcher_stats_2023_2025.csv"
        if merged.exists():
            paths.append(merged)
    if not paths:
        return pd.DataFrame()

    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "Season" in df.columns and "season" not in df.columns:
        df = df.rename(columns={"Season": "season"})
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce")
    df["mlbam_id"] = pd.to_numeric(df.get("mlbam_id"), errors="coerce")
    return df


def _update_platoon_splits(py_dir: Path, season: int, pitcher_ids: List[int]) -> None:
    if not pitcher_ids:
        return
    out_path = py_dir / f"platoon_splits_{season}.csv"
    existing = set()
    if out_path.exists():
        try:
            df = pd.read_csv(out_path)
            existing = set(pd.to_numeric(df.get("mlbam_id"), errors="coerce").dropna().astype(int).tolist())
        except Exception:
            existing = set()

    missing = sorted({pid for pid in pitcher_ids if pid and pid not in existing})
    if not missing:
        return

    logging.info("Fetching platoon splits for %d pitchers (season %s)", len(missing), season)
    fangraphs_platoon_splits_crawler.crawl_with_checkpoint(season, missing, str(out_path))

    # refresh merged file if present
    merged_path = py_dir / "platoon_splits_merged.csv"
    if merged_path.exists():
        try:
            merged = pd.read_csv(merged_path)
            season_df = pd.read_csv(out_path)
            merged = pd.concat([merged, season_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["season", "mlbam_id"], keep="last")
            merged.to_csv(merged_path, index=False)
        except Exception as exc:
            logging.warning("Failed to update merged platoon splits: %s", exc)


def _attach_pitcher_stats(df: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    if df.empty or pit.empty:
        return df
    df = df.copy()
    pit = pit.copy()

    pit = pit.rename(
        columns={
            "ERA": "ERA",
            "WHIP": "WHIP",
            "K%": "K%",
            "BB%": "BB%",
            "K-BB%": "K-BB%",
            "FIP": "FIP",
            "xFIP": "xFIP",
            "SIERA": "SIERA",
            "WAR": "WAR",
            "IP": "IP",
        }
    )

    keep_cols = ["mlbam_id", "season", "ERA", "WHIP", "K%", "BB%", "K-BB%", "FIP", "xFIP", "SIERA", "WAR", "IP"]
    pit = pit[[c for c in keep_cols if c in pit.columns]].copy()

    pit["mlbam_id"] = pd.to_numeric(pit.get("mlbam_id"), errors="coerce")
    pit["season"] = pd.to_numeric(pit.get("season"), errors="coerce")

    home_pit = pit.rename(
        columns={
            "mlbam_id": "home_pitcher_mlbam",
            "ERA": "home_p_ERA",
            "WHIP": "home_p_WHIP",
            "K%": "home_p_K%",
            "BB%": "home_p_BB%",
            "K-BB%": "home_p_K-BB%",
            "FIP": "home_p_FIP",
            "xFIP": "home_p_xFIP",
            "SIERA": "home_p_SIERA",
            "WAR": "home_p_WAR",
            "IP": "home_p_IP",
        }
    )
    away_pit = pit.rename(
        columns={
            "mlbam_id": "away_pitcher_mlbam",
            "ERA": "away_p_ERA",
            "WHIP": "away_p_WHIP",
            "K%": "away_p_K%",
            "BB%": "away_p_BB%",
            "K-BB%": "away_p_K-BB%",
            "FIP": "away_p_FIP",
            "xFIP": "away_p_xFIP",
            "SIERA": "away_p_SIERA",
            "WAR": "away_p_WAR",
            "IP": "away_p_IP",
        }
    )

    df = df.merge(
        home_pit,
        how="left",
        on=["season", "home_pitcher_mlbam"],
    )
    df = df.merge(
        away_pit,
        how="left",
        on=["season", "away_pitcher_mlbam"],
    )
    return df


def _attach_platoon(df: pd.DataFrame, platoon: pd.DataFrame) -> pd.DataFrame:
    if df.empty or platoon.empty:
        return df
    df = df.copy()
    platoon = platoon.copy()

    platoon["mlbam_id"] = pd.to_numeric(platoon.get("mlbam_id"), errors="coerce")
    platoon["season"] = pd.to_numeric(platoon.get("season"), errors="coerce")

    required_cols = [
        "platoon_ba_diff",
        "platoon_ops_diff",
        "platoon_k_rate_lhb",
        "platoon_k_rate_rhb",
        "platoon_splits_score",
    ]
    avail_cols = [c for c in required_cols if c in platoon.columns]

    # Use latest season per pitcher
    platoon_latest = (
        platoon.dropna(subset=["mlbam_id"])
        .sort_values("season")
        .drop_duplicates(subset=["mlbam_id"], keep="last")
    )

    home_platoon = platoon_latest[["mlbam_id"] + avail_cols].rename(
        columns={"mlbam_id": "home_pitcher_mlbam", **{c: f"home_{c}" for c in avail_cols}}
    )
    away_platoon = platoon_latest[["mlbam_id"] + avail_cols].rename(
        columns={"mlbam_id": "away_pitcher_mlbam", **{c: f"away_{c}" for c in avail_cols}}
    )

    df = df.merge(home_platoon, how="left", on="home_pitcher_mlbam")
    df = df.merge(away_platoon, how="left", on="away_pitcher_mlbam")
    return df


def _fill_missing_numeric(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    exclude = set(exclude or [])
    df = df.copy()
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    for col in numeric_cols:
        series = df[col]
        if series.isna().all():
            df[col] = 0
        else:
            df[col] = series.fillna(series.median())
    return df


def build_training_rows(results_df: pd.DataFrame, py_dir: Path) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    results_df = results_df.copy()
    results_df["season"] = pd.to_datetime(results_df["game_date"]).dt.year

    seasons = sorted(results_df["season"].dropna().unique().astype(int).tolist())

    starters = feature_builder.load_pybaseball_starting_pitchers(str(py_dir), seasons)
    if not starters.empty:
        starters = starters[[
            "game_date",
            "home_team",
            "away_team",
            "home_pitcher_mlbam",
            "away_pitcher_mlbam",
        ]].copy()
        results_df = results_df.merge(
            starters,
            how="left",
            on=["game_date", "home_team", "away_team"],
        )

    # update platoon splits for new pitchers
    pitcher_ids = pd.concat([
        results_df.get("home_pitcher_mlbam"),
        results_df.get("away_pitcher_mlbam"),
    ]).dropna().astype(int).unique().tolist()

    for season in seasons:
        _update_platoon_splits(py_dir, season, pitcher_ids)

    pitcher_stats = _load_pitcher_stats(py_dir, seasons)
    if pitcher_stats.empty:
        logging.warning("Pitcher stats missing for seasons %s", seasons)
    else:
        results_df = _attach_pitcher_stats(results_df, pitcher_stats)

    platoon = feature_builder.load_pybaseball_platoon_splits(str(py_dir), seasons)
    if platoon.empty:
        logging.warning("Platoon splits missing for seasons %s", seasons)
    else:
        results_df = _attach_platoon(results_df, platoon)

    # ensure v8 feature columns exist
    for col in V8_FEATURES:
        if col not in results_df.columns:
            results_df[col] = pd.NA

    results_df = _fill_missing_numeric(results_df, exclude=["home_runs", "away_runs", "home_win"])

    return results_df


def _load_existing_keys(training_csv: Path) -> set:
    if not training_csv.exists():
        return set()
    try:
        df = pd.read_csv(training_csv)
    except Exception:
        return set()
    if df.empty:
        return set()
    keys = set(
        zip(
            pd.to_datetime(df.get("game_date"), errors="coerce").dt.date,
            df.get("home_team"),
            df.get("away_team"),
        )
    )
    return {k for k in keys if all(k)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./data/results")
    p.add_argument("--date", help="Process a single date (YYYY-MM-DD)")
    p.add_argument("--training-csv", default="./data/training_2022_2025_enhanced_v6.csv")
    p.add_argument("--pybaseball-dir", default="./data/pybaseball")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    py_dir = Path(args.pybaseball_dir)
    training_csv = Path(args.training_csv)

    if args.date:
        paths = [results_dir / f"{args.date}.json"]
    else:
        paths = sorted(results_dir.glob("*.json"))

    if not paths:
        logging.warning("No result files found in %s", results_dir)
        return

    results_df = load_results_files(paths)
    if results_df.empty:
        logging.warning("No completed games found in results")
        return

    results_df = _resolve_team_abbr(results_df)
    if results_df.empty:
        logging.warning("No games after team name normalization")
        return

    existing_keys = _load_existing_keys(training_csv)
    results_df["game_key"] = list(zip(results_df["game_date"], results_df["home_team"], results_df["away_team"]))
    results_df = results_df[~results_df["game_key"].isin(existing_keys)].copy()
    if results_df.empty:
        logging.info("No new games to append")
        return

    results_df = results_df.drop(columns=["game_key"], errors="ignore")
    results_df = build_training_rows(results_df, py_dir)

    # order columns
    base_cols = [
        "game_date",
        "home_team",
        "away_team",
        "home_runs",
        "away_runs",
        "home_win",
    ]
    cols = [c for c in base_cols if c in results_df.columns] + V8_FEATURES
    extra = [c for c in results_df.columns if c not in cols]
    out_df = results_df[cols + extra].copy()

    training_csv.parent.mkdir(parents=True, exist_ok=True)
    if training_csv.exists():
        existing_df = pd.read_csv(training_csv)
        combined = pd.concat([existing_df, out_df], ignore_index=True)
    else:
        combined = out_df

    combined.to_csv(training_csv, index=False)
    logging.info("Appended %d games to %s", len(out_df), training_csv)


if __name__ == "__main__":
    main()
