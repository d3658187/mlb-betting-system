#!/usr/bin/env python3
"""Build team_season_summary table (SQLite) for cold-start proxy features."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from pybaseball import team_batting, team_pitching

_ABBR_NORMALIZE = {
    "AZ": "ARI",
    "KC": "KCR",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "WSH": "WSN",
    "WAS": "WSN",
    "CWS": "CHW",
    "OAK": "ATH",
}

_TEAM_NAME = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "ATH": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",
    "SFG": "San Francisco Giants",
    "SEA": "Seattle Mariners",
    "STL": "St. Louis Cardinals",
    "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals",
}


def _parse_seasons(raw: str) -> List[int]:
    raw = raw.strip()
    if "-" in raw:
        start, end = [int(x) for x in raw.split("-")]
        return list(range(start, end + 1))
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _norm_abbrev(val) -> str:
    if val is None or pd.isna(val):
        return ""
    out = str(val).strip().upper()
    return _ABBR_NORMALIZE.get(out, out)


def _pick(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _prep_year(year: int) -> pd.DataFrame:
    batting = team_batting(year, year)
    pitching = team_pitching(year, year)

    batting = batting.copy()
    pitching = pitching.copy()

    batting["team_abbrev"] = batting["Team"].map(_norm_abbrev)
    pitching["team_abbrev"] = pitching["Team"].map(_norm_abbrev)

    batting = batting[batting["team_abbrev"].ne("")].copy()
    pitching = pitching[pitching["team_abbrev"].ne("")].copy()

    # keep MLB teams only (exclude aggregates)
    batting = batting[batting["team_abbrev"].isin(_TEAM_NAME.keys())].copy()
    pitching = pitching[pitching["team_abbrev"].isin(_TEAM_NAME.keys())].copy()

    merge_cols = ["team_abbrev"]
    b_cols = [
        "teamIDfg",
        "PA",
        "AB",
        "R",
        "H",
        "2B",
        "3B",
        "HR",
        "BB",
        "SO",
        "SB",
        "CS",
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "ISO",
        "wOBA",
        "wRC+",
        "BB%",
        "K%",
    ]
    p_cols = [
        "W",
        "L",
        "ERA",
        "WHIP",
        "IP",
        "H",
        "R",
        "ER",
        "HR",
        "BB",
        "SO",
        "G",
        "K%",
        "BB%",
        "K-BB%",
        "FIP",
        "xFIP",
        "SIERA",
        "WAR",
    ]

    b_keep = merge_cols + [c for c in b_cols if c in batting.columns]
    p_keep = merge_cols + [c for c in p_cols if c in pitching.columns]

    merged = batting[b_keep].merge(
        pitching[p_keep],
        on="team_abbrev",
        how="inner",
        suffixes=("_bat", "_pit"),
    )

    out = pd.DataFrame(index=merged.index.copy())
    out["season_year"] = year
    out["team_abbrev"] = merged["team_abbrev"]
    out["team_name"] = out["team_abbrev"].map(_TEAM_NAME).fillna(out["team_abbrev"])

    team_id_col = _pick(merged, "teamIDfg", "teamIDfg_bat", "teamIDfg_pit")
    out["team_id"] = pd.to_numeric(merged[team_id_col], errors="coerce") if team_id_col else np.nan

    # batting (requested)
    out["bat_pa"] = pd.to_numeric(merged.get("PA"), errors="coerce")
    out["bat_ab"] = pd.to_numeric(merged.get("AB"), errors="coerce")
    out["bat_r"] = pd.to_numeric(merged.get("R_bat", merged.get("R")), errors="coerce")
    out["bat_h"] = pd.to_numeric(merged.get("H_bat", merged.get("H")), errors="coerce")
    out["bat_2b"] = pd.to_numeric(merged.get("2B"), errors="coerce")
    out["bat_3b"] = pd.to_numeric(merged.get("3B"), errors="coerce")
    out["bat_hr"] = pd.to_numeric(merged.get("HR_bat", merged.get("HR")), errors="coerce")
    out["bat_bb"] = pd.to_numeric(merged.get("BB_bat", merged.get("BB")), errors="coerce")
    out["bat_so"] = pd.to_numeric(merged.get("SO_bat", merged.get("SO")), errors="coerce")
    out["bat_sb"] = pd.to_numeric(merged.get("SB"), errors="coerce")
    out["bat_cs"] = pd.to_numeric(merged.get("CS"), errors="coerce")
    out["bat_avg"] = pd.to_numeric(merged.get("AVG"), errors="coerce")
    out["bat_obp"] = pd.to_numeric(merged.get("OBP"), errors="coerce")
    out["bat_slg"] = pd.to_numeric(merged.get("SLG"), errors="coerce")
    out["bat_ops"] = pd.to_numeric(merged.get("OPS"), errors="coerce")

    # optional batting advanced
    out["bat_iso"] = pd.to_numeric(merged.get("ISO"), errors="coerce")
    out["bat_woba"] = pd.to_numeric(merged.get("wOBA"), errors="coerce")
    out["bat_wrc_plus"] = pd.to_numeric(merged.get("wRC+"), errors="coerce")
    out["bat_bb_pct"] = pd.to_numeric(merged.get("BB%"), errors="coerce")
    out["bat_k_pct"] = pd.to_numeric(merged.get("K%"), errors="coerce")

    # pitching (requested)
    out["pit_w"] = pd.to_numeric(merged.get("W"), errors="coerce")
    out["pit_l"] = pd.to_numeric(merged.get("L"), errors="coerce")
    out["pit_era"] = pd.to_numeric(merged.get("ERA"), errors="coerce")
    out["pit_whip"] = pd.to_numeric(merged.get("WHIP"), errors="coerce")
    out["pit_ip"] = pd.to_numeric(merged.get("IP"), errors="coerce")
    out["pit_h"] = pd.to_numeric(merged.get("H_pit", merged.get("H")), errors="coerce")
    out["pit_r"] = pd.to_numeric(merged.get("R_pit", merged.get("R")), errors="coerce")
    out["pit_er"] = pd.to_numeric(merged.get("ER"), errors="coerce")
    out["pit_hr"] = pd.to_numeric(merged.get("HR_pit", merged.get("HR")), errors="coerce")
    out["pit_bb"] = pd.to_numeric(merged.get("BB_pit", merged.get("BB")), errors="coerce")
    out["pit_so"] = pd.to_numeric(merged.get("SO_pit", merged.get("SO")), errors="coerce")
    out["pit_g"] = pd.to_numeric(merged.get("G"), errors="coerce")

    # optional pitching advanced
    out["pit_k_pct"] = pd.to_numeric(merged.get("K%_pit", merged.get("K%")), errors="coerce")
    out["pit_bb_pct"] = pd.to_numeric(merged.get("BB%_pit", merged.get("BB%")), errors="coerce")
    out["pit_kbb_pct"] = pd.to_numeric(merged.get("K-BB%"), errors="coerce")
    out["pit_fip"] = pd.to_numeric(merged.get("FIP"), errors="coerce")
    out["pit_xfip"] = pd.to_numeric(merged.get("xFIP"), errors="coerce")
    out["pit_siera"] = pd.to_numeric(merged.get("SIERA"), errors="coerce")
    out["pit_war"] = pd.to_numeric(merged.get("WAR"), errors="coerce")

    games = (out["pit_w"] + out["pit_l"]).replace(0, np.nan)
    out["win_pct"] = out["pit_w"] / games

    rs_sq = out["bat_r"] ** 2
    ra_sq = out["pit_r"] ** 2
    denom = rs_sq + ra_sq
    out["pythagorean_win_pct"] = np.where(denom > 0, rs_sq / denom, np.nan)

    out = out.sort_values("team_abbrev").reset_index(drop=True)
    return out


def build_summary(seasons: Sequence[int]) -> pd.DataFrame:
    frames = []
    for year in seasons:
        print(f"Fetching team season summary: {year}")
        frames.append(_prep_year(int(year)))
    out = pd.concat(frames, ignore_index=True)

    out["season_year"] = pd.to_numeric(out["season_year"], errors="coerce").astype("Int64")
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce")
    out = out.drop_duplicates(subset=["season_year", "team_abbrev"], keep="last")
    return out


def write_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("team_season_summary", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_team_season_summary_year_team ON team_season_summary(season_year, team_abbrev)"
        )
        conn.commit()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2023-2025", help="Season range, e.g. 2023-2025")
    p.add_argument("--db-path", default="data/mlb.db", help="SQLite db path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seasons = _parse_seasons(args.seasons)
    summary = build_summary(seasons)
    write_sqlite(summary, Path(args.db_path))
    print(f"Wrote team_season_summary to {args.db_path}: {len(summary)} rows")


if __name__ == "__main__":
    main()
