#!/usr/bin/env python3
"""Compare v4 (training_2025_enhanced.csv) vs v5 (historical_features_v4.csv + richer features).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

import model_trainer
from train_v5_more_features import add_h2h_features, build_feature_cols

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def train_v4(csv_path: str):
    df = pd.read_csv(csv_path)
    if "home_win" in df.columns:
        df = df[df["home_win"].notna()].copy()
    model, metrics, feature_cols, feature_importance = model_trainer.train_model(
        df, target="home_win", task="classification"
    )
    return metrics, feature_cols, feature_importance


def train_v5(csv_path: str):
    df = pd.read_csv(csv_path)
    if "home_win" in df.columns:
        df = df[df["home_win"].notna()].copy()
    df = add_h2h_features(df, windows=[5, 10])
    df, feature_cols = build_feature_cols(df)
    model, metrics, feature_cols, feature_importance = model_trainer.train_model(
        df, target="home_win", task="classification", feature_cols=feature_cols
    )
    return metrics, feature_cols, feature_importance


def main():
    v4_csv = "./data/training_2025_enhanced.csv"
    v5_csv = "./data/pybaseball/historical_features_v4.csv"

    if not Path(v4_csv).exists() or not Path(v5_csv).exists():
        raise SystemExit("Missing v4/v5 CSVs")

    v4_metrics, v4_cols, v4_imp = train_v4(v4_csv)
    v5_metrics, v5_cols, v5_imp = train_v5(v5_csv)

    print(json.dumps({
        "v4": {"metrics": v4_metrics, "n_features": len(v4_cols), "top_features": v4_imp[:10]},
        "v5": {"metrics": v5_metrics, "n_features": len(v5_cols), "top_features": v5_imp[:10]},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
