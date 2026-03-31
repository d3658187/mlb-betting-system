#!/usr/bin/env python3
"""Build a local SQLite cache DB (data/mlb.db) from CSV/pybaseball sources.

This does NOT replace the PostgreSQL production pipeline.
It provides an offline cache for local development and fallback feature generation.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import pandas as pd

import feature_builder


def _parse_seasons(raw: str) -> List[int]:
    raw = raw.strip()
    if "-" in raw:
        start, end = [int(x) for x in raw.split("-")]
        return list(range(start, end + 1))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _normalize_sql_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw_cols = []
    for c in out.columns:
        col = str(c).strip().lower()
        col = col.replace("%", "pct").replace("/", "_").replace(" ", "_").replace("-", "_")
        while "__" in col:
            col = col.replace("__", "_")
        col = col.strip("_") or "col"
        raw_cols.append(col)

    seen = {}
    final_cols = []
    for c in raw_cols:
        if c not in seen:
            seen[c] = 1
            final_cols.append(c)
        else:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")

    out.columns = final_cols
    return out


def _write_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    out = _normalize_sql_columns(df)
    if "game_date" in out.columns:
        out["game_date"] = out["game_date"].astype(str)
    out.to_sql(name, conn, if_exists="replace", index=False)
    return len(out)


def run(db_path: Path, data_dir: Path, seasons: Sequence[int]) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    games = feature_builder.load_pybaseball_games(str(data_dir), seasons)
    team_batting = feature_builder.load_pybaseball_team_stats(str(data_dir), seasons, kind="batting")
    team_pitching = feature_builder.load_pybaseball_team_stats(str(data_dir), seasons, kind="pitching")
    pitcher_stats = feature_builder.load_pybaseball_pitcher_stats(str(data_dir), seasons)
    starters = feature_builder.load_pybaseball_starting_pitchers(str(data_dir), seasons)
    platoon = feature_builder.load_pybaseball_platoon_splits(str(data_dir), seasons)

    with sqlite3.connect(db_path) as conn:
        counts = {
            "games": _write_table(conn, "games", games),
            "team_batting": _write_table(conn, "team_batting", team_batting),
            "team_pitching": _write_table(conn, "team_pitching", team_pitching),
            "pitcher_stats": _write_table(conn, "pitcher_stats", pitcher_stats),
            "starting_pitchers": _write_table(conn, "starting_pitchers", starters),
            "platoon_splits": _write_table(conn, "platoon_splits", platoon),
        }

        meta = pd.DataFrame(
            [{
                "built_at": datetime.now().isoformat(timespec="seconds"),
                "seasons": ",".join(str(s) for s in seasons),
                "data_dir": str(data_dir),
            }]
        )
        meta.to_sql("metadata", conn, if_exists="replace", index=False)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team, away_team)")
        conn.commit()

    size_kb = db_path.stat().st_size / 1024
    print(f"SQLite DB built: {db_path} ({size_kb:.1f} KB)")
    for k, v in counts.items():
        print(f"  - {k}: {v} rows")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default="data/mlb.db", help="SQLite output path")
    p.add_argument("--data-dir", default="data/pybaseball", help="pybaseball CSV directory")
    p.add_argument("--seasons", default="2022-2026", help="Season range, e.g. 2022-2026 or 2024,2025")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seasons = _parse_seasons(args.seasons)
    run(db_path=Path(args.db_path), data_dir=Path(args.data_dir), seasons=seasons)


if __name__ == "__main__":
    main()
