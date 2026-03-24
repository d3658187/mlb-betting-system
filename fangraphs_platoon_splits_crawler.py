#!/usr/bin/env python3
"""Platoon splits crawler via pybaseball (Baseball Reference).

Usage:
  python fangraphs_platoon_splits_crawler.py --season 2025 --pitcher-csv ./data/pybaseball/starting_pitchers_2025.csv \
    --out ./data/pybaseball/platoon_splits_2025.csv

  python fangraphs_platoon_splits_crawler.py --season 2025 --mlbam-ids 477132,592450 --limit 2
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from pybaseball import get_splits, playerid_reverse_lookup
except Exception:  # pragma: no cover
    get_splits = None
    playerid_reverse_lookup = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

RATE_LIMIT_PER_MIN = 20
REQUEST_INTERVAL_SEC = 60.0 / RATE_LIMIT_PER_MIN


# ---------------------
# Helpers
# ---------------------

def _chunked(seq: List[int], size: int = 200) -> List[List[int]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def _to_int(val) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _to_float(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _find_col(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    col_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in col_map:
            return col_map[key]
    return None


def _rate_limit(last_ts: Optional[float]) -> float:
    if last_ts is None:
        return time.time()
    elapsed = time.time() - last_ts
    if elapsed < REQUEST_INTERVAL_SEC:
        time.sleep(REQUEST_INTERVAL_SEC - elapsed)
    return time.time()


def _load_ids_from_csv(path: str) -> List[int]:
    df = pd.read_csv(path)
    if df.empty:
        return []
    candidates = [
        "mlbam_id",
        "pitcher_mlbam",
        "pitcher_mlb_id",
        "home_pitcher_mlbam",
        "away_pitcher_mlbam",
        "home_pitcher_mlb_id",
        "away_pitcher_mlb_id",
    ]
    ids: List[int] = []
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist()
            ids.extend(vals)
    ids = sorted(set(ids))
    return ids


def _map_mlbam_to_bbref(mlbam_ids: List[int]) -> Tuple[Dict[int, str], Dict[int, str]]:
    if playerid_reverse_lookup is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")
    mapping: Dict[int, str] = {}
    name_map: Dict[int, str] = {}
    for chunk in _chunked(mlbam_ids, size=200):
        df = playerid_reverse_lookup(chunk, key_type="mlbam")
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            mlbam_id = _to_int(row.get("key_mlbam"))
            bbref_id = row.get("key_bbref")
            first = str(row.get("name_first") or "").strip()
            last = str(row.get("name_last") or "").strip()
            full = str(row.get("name_full") or "").strip()
            name = full or f"{first} {last}".strip()
            if mlbam_id is None or not bbref_id:
                continue
            mapping[mlbam_id] = str(bbref_id)
            if name:
                name_map[mlbam_id] = name
    return mapping, name_map


def _extract_platoon_features(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    if df is None:
        return None

    if isinstance(df, tuple):
        df = df[0]

    if df is None or df.empty:
        return None

    if isinstance(df.index, pd.MultiIndex) and "Split Type" not in df.columns:
        df = df.reset_index()

    split_type_col = _find_col(df.columns, ["Split Type", "SplitType", "split_type"])
    split_col = _find_col(df.columns, ["Split", "split"])
    if not split_type_col or not split_col:
        return None

    platoon = df[df[split_type_col].astype(str).str.contains("Platoon", case=False, na=False)].copy()
    if platoon.empty:
        return None

    platoon[split_col] = platoon[split_col].astype(str).str.strip()
    platoon = platoon[platoon[split_col].isin(["vs LHB", "vs RHB"])]
    if platoon.empty:
        return None

    def _get_split_row(label: str) -> Optional[pd.Series]:
        rows = platoon[platoon[split_col] == label]
        if rows.empty:
            return None
        return rows.iloc[0]

    def _get_val(row: pd.Series, col_name: str) -> Optional[float]:
        if row is None:
            return None
        col = _find_col(row.index, [col_name])
        if not col:
            return None
        return _to_float(row.get(col))

    lhb = _get_split_row("vs LHB")
    rhb = _get_split_row("vs RHB")

    pa_lhb = _get_val(lhb, "PA")
    so_lhb = _get_val(lhb, "SO")
    ba_lhb = _get_val(lhb, "BA")
    ops_lhb = _get_val(lhb, "OPS")

    pa_rhb = _get_val(rhb, "PA")
    so_rhb = _get_val(rhb, "SO")
    ba_rhb = _get_val(rhb, "BA")
    ops_rhb = _get_val(rhb, "OPS")

    platoon_ba_diff = None if ba_lhb is None or ba_rhb is None else ba_lhb - ba_rhb
    platoon_ops_diff = None if ops_lhb is None or ops_rhb is None else ops_lhb - ops_rhb
    platoon_k_rate_lhb = None if not pa_lhb else (so_lhb / pa_lhb if so_lhb is not None else None)
    platoon_k_rate_rhb = None if not pa_rhb else (so_rhb / pa_rhb if so_rhb is not None else None)

    if platoon_ba_diff is None or platoon_ops_diff is None:
        platoon_splits_score = None
    else:
        platoon_splits_score = abs(platoon_ba_diff) + abs(platoon_ops_diff) * 0.5

    return {
        "pa_vs_lhb": pa_lhb,
        "so_vs_lhb": so_lhb,
        "ba_vs_lhb": ba_lhb,
        "ops_vs_lhb": ops_lhb,
        "pa_vs_rhb": pa_rhb,
        "so_vs_rhb": so_rhb,
        "ba_vs_rhb": ba_rhb,
        "ops_vs_rhb": ops_rhb,
        "platoon_ba_diff": platoon_ba_diff,
        "platoon_ops_diff": platoon_ops_diff,
        "platoon_k_rate_lhb": platoon_k_rate_lhb,
        "platoon_k_rate_rhb": platoon_k_rate_rhb,
        "platoon_splits_score": platoon_splits_score,
    }


def fetch_platoon_splits(
    season: int,
    mlbam_ids: List[int],
    limit: Optional[int] = None,
) -> List[Dict[str, Optional[float]]]:
    if get_splits is None:
        raise RuntimeError("pybaseball is not installed; run pip install pybaseball")

    mlbam_ids = [int(x) for x in mlbam_ids if x is not None]
    if limit:
        mlbam_ids = mlbam_ids[:limit]

    mapping, name_map = _map_mlbam_to_bbref(mlbam_ids)

    rows: List[Dict[str, Optional[float]]] = []
    last_ts: Optional[float] = None
    missing = 0

    for mlbam_id in mlbam_ids:
        bbref_id = mapping.get(mlbam_id)
        if not bbref_id:
            missing += 1
            continue

        last_ts = _rate_limit(last_ts)
        try:
            df = get_splits(bbref_id, year=season, pitching_splits=True)
        except Exception as exc:
            logging.warning("get_splits failed for %s (%s): %s", mlbam_id, bbref_id, exc)
            continue

        feats = _extract_platoon_features(df)
        if not feats:
            logging.warning("No platoon splits for %s (%s)", mlbam_id, bbref_id)
            continue

        row = {
            "season": season,
            "mlbam_id": mlbam_id,
            "bbref_id": bbref_id,
            "name": name_map.get(mlbam_id),
        }
        row.update(feats)
        rows.append(row)

    if missing:
        logging.warning("Missing BBRef ID mapping for %d pitchers", missing)
    logging.info("Fetched platoon splits for %d pitchers", len(rows))
    return rows


def write_csv(rows: List[Dict[str, Optional[float]]], path: str) -> None:
    if not rows:
        logging.warning("No data to write: %s", path)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logging.info("Wrote %d rows -> %s", len(df), path)


# ---------------------
# CLI
# ---------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--mlbam-ids", type=str, help="Comma-separated MLBAM pitcher IDs")
    p.add_argument("--pitcher-csv", type=str, help="CSV containing pitcher MLBAM IDs")
    p.add_argument("--out", type=str, help="Output CSV path")
    p.add_argument("--limit", type=int, help="Limit number of pitchers (for testing)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.mlbam_ids and not args.pitcher_csv:
        raise SystemExit("Provide --mlbam-ids or --pitcher-csv")

    mlbam_ids: List[int] = []
    if args.mlbam_ids:
        mlbam_ids = [int(x.strip()) for x in args.mlbam_ids.split(",") if x.strip()]
    elif args.pitcher_csv:
        mlbam_ids = _load_ids_from_csv(args.pitcher_csv)

    if not mlbam_ids:
        raise SystemExit("No valid MLBAM IDs found")

    rows = fetch_platoon_splits(args.season, mlbam_ids, limit=args.limit)

    out_path = args.out or f"./data/pybaseball/platoon_splits_{args.season}.csv"
    write_csv(rows, out_path)


if __name__ == "__main__":
    main()
