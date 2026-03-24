#!/usr/bin/env python3
"""Train v8 Over/Under (total runs) and Run Line (run margin) regression models.

Usage:
  source .venv/bin/activate
  python train_v8_overunder_runline.py \
    --csv ./data/training_2022_2025_enhanced_v6.csv \
    --out-dir ./models
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    raise SystemExit("xgboost is required for booster output")


LEAKAGE_COLS = {
    "home_score",
    "away_score",
    "total_points",
    "home_win",
    "away_win",
    "home_runs",
    "away_runs",
    "result",
    "winner",
    "winning_team",
    "losing_team",
    "run_margin",
    "cover_spread",
    "total_runs",
}
DROP_COLS = {"mlb_game_id", "game_id", "game_date", "game_datetime", "status"}


def infer_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    feature_cols = [
        c
        for c in df.columns
        if c not in DROP_COLS
        and c not in LEAKAGE_COLS
        and c != target
        and not c.endswith("_id")
        and not c.endswith("_score")
        and not c.endswith("_points")
    ]
    return feature_cols


def prepare_xy(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = infer_feature_columns(df, target)
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    feature_cols = list(X.columns)
    X = X.fillna(0)
    y = pd.to_numeric(df[target], errors="coerce")
    return X, y, feature_cols


def time_series_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, date_col: str) -> Tuple:
    df_sorted = df.sort_values(date_col)
    X_sorted = X.loc[df_sorted.index]
    y_sorted = y.loc[df_sorted.index]
    split_idx = max(int(len(df_sorted) * 0.8), 1)
    X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
    y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def train_regression(df: pd.DataFrame, target: str, date_col: str):
    X, y, feature_cols = prepare_xy(df, target)
    X_train, X_test, y_train, y_test = time_series_split(df, X, y, date_col)

    model = xgb.XGBRegressor(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        objective="reg:squarederror",
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse_val = float(mean_squared_error(y_test, preds))
    metrics = {
        "task": "regression",
        "split_method": "time_series",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mse_val ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        "mse": mse_val,
    }

    return model, metrics, feature_cols


def save_booster(model, metrics, feature_cols, out_dir: Path, model_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    booster_path = out_dir / f"{model_name}.booster"
    meta_path = out_dir / f"{model_name}.meta.json"

    model.get_booster().save_model(str(booster_path))
    meta = {
        "model_type": type(model).__name__,
        "format": "booster",
        "metrics": metrics,
        "feature_cols": feature_cols,
        "trained_at": date.today().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return booster_path, meta_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", default="./models")
    p.add_argument("--date-col", default="game_date_x")
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit("No training data found")

    if args.date_col not in df.columns:
        raise SystemExit(f"date column not found: {args.date_col}")

    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")

    df = df.dropna(subset=["home_runs", "away_runs"])
    df = df.copy()
    df["total_runs"] = df["home_runs"] + df["away_runs"]
    df["run_margin"] = df["home_runs"] - df["away_runs"]

    # Over/Under total runs regression
    ou_model, ou_metrics, ou_features = train_regression(df, "total_runs", args.date_col)
    save_booster(ou_model, ou_metrics, ou_features, Path(args.out_dir), "mlb_v8_overunder")

    # Run line margin regression
    rl_model, rl_metrics, rl_features = train_regression(df, "run_margin", args.date_col)
    save_booster(rl_model, rl_metrics, rl_features, Path(args.out_dir), "mlb_v8_runline")

    print(json.dumps({
        "overunder": ou_metrics,
        "runline": rl_metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
