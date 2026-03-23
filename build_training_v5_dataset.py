#!/usr/bin/env python3
"""Build enriched training dataset (2022-2024) by joining pitcher + team batting stats.

Usage:
  python build_training_v5_dataset.py \
    --base ./data/training_2022_2024.csv \
    --out ./data/training_2022_2024_enhanced_v5.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="./data/training_2022_2024.csv")
    p.add_argument("--out", default="./data/training_2022_2024_enhanced_v5.csv")
    p.add_argument("--pybaseball-dir", default="./data/pybaseball")
    return p.parse_args()


def load_pitcher_stats(pybaseball_dir: Path, season: int) -> pd.DataFrame:
    p = pybaseball_dir / f"pitcher_stats_{season}.csv"
    df = pd.read_csv(p)
    keep = [
        "mlbam_id", "ERA", "WHIP", "K%", "BB%", "K-BB%", "FIP", "xFIP", "SIERA", "WAR", "IP"
    ]
    df = df[keep].copy()
    df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce").astype("Int64")
    return df


def load_team_batting(pybaseball_dir: Path, season: int) -> pd.DataFrame:
    p = pybaseball_dir / f"team_batting_{season}.csv"
    df = pd.read_csv(p)
    keep = [
        "team", "AVG", "OBP", "SLG", "OPS", "ISO", "wOBA", "wRC+", "BB%", "K%", "R", "HR", "SB"
    ]
    df = df[keep].copy()
    df = df[df["team"].notna()]
    df = df[df["team"] != "- - -"]
    return df


def main():
    args = parse_args()
    base_path = Path(args.base)
    out_path = Path(args.out)
    py_dir = Path(args.pybaseball_dir)

    if not base_path.exists():
        raise SystemExit(f"Base CSV not found: {base_path}")

    base = pd.read_csv(base_path)
    base["season"] = pd.to_numeric(base["season"], errors="coerce").astype("Int64")

    seasons = sorted(base["season"].dropna().unique())

    pitcher_map = {}
    batting_map = {}
    for season in seasons:
        try:
            pitcher_map[season] = load_pitcher_stats(py_dir, season)
            batting_map[season] = load_team_batting(py_dir, season)
        except FileNotFoundError:
            continue

    enriched_parts = []
    for season in seasons:
        if season not in pitcher_map or season not in batting_map:
            continue
        df = base[base["season"] == season].copy()

        pit = pitcher_map[season]
        bat = batting_map[season]

        df["home_pitcher_mlbam"] = pd.to_numeric(df["home_pitcher_mlbam"], errors="coerce").astype("Int64")
        df["away_pitcher_mlbam"] = pd.to_numeric(df["away_pitcher_mlbam"], errors="coerce").astype("Int64")

        # Join pitcher stats
        df = df.merge(pit, how="left", left_on="home_pitcher_mlbam", right_on="mlbam_id")
        df = df.rename(columns={
            "ERA": "home_p_ERA", "WHIP": "home_p_WHIP", "K%": "home_p_K%", "BB%": "home_p_BB%",
            "K-BB%": "home_p_K-BB%", "FIP": "home_p_FIP", "xFIP": "home_p_xFIP", "SIERA": "home_p_SIERA",
            "WAR": "home_p_WAR", "IP": "home_p_IP",
        }).drop(columns=["mlbam_id"])

        df = df.merge(pit, how="left", left_on="away_pitcher_mlbam", right_on="mlbam_id")
        df = df.rename(columns={
            "ERA": "away_p_ERA", "WHIP": "away_p_WHIP", "K%": "away_p_K%", "BB%": "away_p_BB%",
            "K-BB%": "away_p_K-BB%", "FIP": "away_p_FIP", "xFIP": "away_p_xFIP", "SIERA": "away_p_SIERA",
            "WAR": "away_p_WAR", "IP": "away_p_IP",
        }).drop(columns=["mlbam_id"])

        # Normalize team keys for merge
        df["home_team"] = df["home_team"].astype(str)
        df["away_team"] = df["away_team"].astype(str)

        # Join team batting
        df = df.merge(bat, how="left", left_on="home_team", right_on="team")
        df = df.rename(columns={
            "AVG": "home_bat_AVG", "OBP": "home_bat_OBP", "SLG": "home_bat_SLG", "OPS": "home_bat_OPS",
            "ISO": "home_bat_ISO", "wOBA": "home_bat_wOBA", "wRC+": "home_bat_wRC+",
            "BB%": "home_bat_BB%", "K%": "home_bat_K%", "R": "home_bat_R", "HR": "home_bat_HR", "SB": "home_bat_SB",
        }).drop(columns=["team"])

        df = df.merge(bat, how="left", left_on="away_team", right_on="team")
        df = df.rename(columns={
            "AVG": "away_bat_AVG", "OBP": "away_bat_OBP", "SLG": "away_bat_SLG", "OPS": "away_bat_OPS",
            "ISO": "away_bat_ISO", "wOBA": "away_bat_wOBA", "wRC+": "away_bat_wRC+",
            "BB%": "away_bat_BB%", "K%": "away_bat_K%", "R": "away_bat_R", "HR": "away_bat_HR", "SB": "away_bat_SB",
        }).drop(columns=["team"])

        # Diff features
        diff_pairs = [
            ("home_p_ERA", "away_p_ERA", "diff_p_ERA"),
            ("home_p_WHIP", "away_p_WHIP", "diff_p_WHIP"),
            ("home_p_K-BB%", "away_p_K-BB%", "diff_p_K-BB%"),
            ("home_bat_wOBA", "away_bat_wOBA", "diff_bat_wOBA"),
            ("home_bat_wRC+", "away_bat_wRC+", "diff_bat_wRC+"),
            ("home_bat_OPS", "away_bat_OPS", "diff_bat_OPS"),
        ]
        for h, a, d in diff_pairs:
            df[d] = pd.to_numeric(df[h], errors="coerce") - pd.to_numeric(df[a], errors="coerce")

        enriched_parts.append(df)

    enriched = pd.concat(enriched_parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} rows={len(enriched)} cols={len(enriched.columns)}")


if __name__ == "__main__":
    main()
