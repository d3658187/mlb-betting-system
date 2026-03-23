#!/usr/bin/env python3
"""Model training scaffold for MLB spread/margin prediction.

Usage:
  # Train from DB table (classification: cover spread)
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  python model_trainer.py --source db --table model_features --target cover_spread --task classification

  # Train from DB table (regression: run margin)
  DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  python model_trainer.py --source db --table model_features --target run_margin --task regression

  # Train from CSV
  python model_trainer.py --source csv --csv ./features.csv --target cover_spread --task classification
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import joblib

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")
    return create_engine(db_url, pool_pre_ping=True)


def load_features_from_db(table: str, target: str) -> pd.DataFrame:
    engine = get_engine()
    sql = text(f"SELECT * FROM {table} WHERE {target} IS NOT NULL")
    return pd.read_sql(sql, engine)


def load_features_from_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def infer_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    drop_cols = {target, "mlb_game_id", "game_id", "game_date", "game_datetime", "status"}
    leakage_cols = {
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
    }
    return [
        c
        for c in df.columns
        if c not in drop_cols
        and c not in leakage_cols
        and not c.endswith("_id")
        and not c.endswith("_score")
        and not c.endswith("_points")
    ]


def drop_leakage_columns(feature_cols: List[str]) -> List[str]:
    leakage_cols = {
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
    }
    filtered = [
        c
        for c in feature_cols
        if c not in leakage_cols and not c.endswith("_score") and not c.endswith("_points")
    ]
    removed = sorted(set(feature_cols) - set(filtered))
    if removed:
        logging.warning("Dropping leakage columns: %s", removed)
    return filtered


def resolve_task(target: str, task: Optional[str]) -> str:
    if task:
        return task
    target_lower = target.lower()
    if "margin" in target_lower:
        return "regression"
    return "classification"


def build_model(task: str):
    if task == "classification":
        if xgb is not None:
            return xgb.XGBClassifier(
                n_estimators=700,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=5,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=4,
            )
        if lgb is not None:
            return lgb.LGBMClassifier(
                n_estimators=400,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
            )
        # Fallback for environments without xgboost/lightgbm
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            l2_regularization=0.0,
        )

    if xgb is not None:
        return xgb.XGBRegressor(
            n_estimators=700,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=5,
            objective="reg:squarederror",
            n_jobs=4,
        )
    if lgb is not None:
        return lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
        )
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        l2_regularization=0.0,
    )


def train_model(
    df: pd.DataFrame,
    target: str,
    task: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[object, dict, List[str], List[Tuple[str, float]]]:
    task = resolve_task(target, task)
    if feature_cols is None:
        feature_cols = infer_feature_columns(df, target)
    feature_cols = drop_leakage_columns(feature_cols)

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        logging.warning("Dropping all-NaN feature columns: %s", all_nan_cols)
        X = X.drop(columns=all_nan_cols)
    feature_cols = list(X.columns)
    X = X.fillna(0)
    y = df[target]

    if task == "classification":
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
    else:
        y = pd.to_numeric(y, errors="coerce")

    split_method = "random"
    if "game_date" in df.columns:
        df_sorted = df.sort_values("game_date")
        X = df_sorted[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df_sorted[target]
        if task == "classification":
            y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        else:
            y = pd.to_numeric(y, errors="coerce")
        split_idx = max(int(len(df_sorted) * 0.8), 1)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        split_method = "time_series"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if task == "classification" else None
        )

    model = build_model(task)
    model.fit(X_train, y_train)

    metrics = {
        "task": task,
        "split_method": split_method,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    if task == "classification":
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        metrics.update(
            {
                "accuracy": float(accuracy_score(y_test, preds)),
                "roc_auc": float(roc_auc_score(y_test, probs)),
            }
        )
    else:
        preds = model.predict(X_test)
        mse_val = float(mean_squared_error(y_test, preds))
        metrics.update(
            {
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(mse_val ** 0.5),
                "r2": float(r2_score(y_test, preds)),
                "mse": mse_val,
            }
        )

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()
        feature_importance = sorted(
            zip(feature_cols, importances), key=lambda x: x[1], reverse=True
        )
    else:
        feature_importance = []

    return model, metrics, feature_cols, feature_importance


def save_artifacts(model, metrics: dict, feature_cols: List[str], feature_importance: List, out_dir: str, target: str, model_name: Optional[str] = None):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_name = model_name or f"mlb_{target}_model"
    model_path = out_path / f"{base_name}.pkl"
    meta_path = out_path / f"{base_name}.meta.json"

    joblib.dump(model, model_path)
    meta = {
        "metrics": metrics,
        "feature_cols": feature_cols,
        "feature_importance": feature_importance,
        "trained_at": date.today().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    logging.info("Saved model to %s", model_path)
    logging.info("Saved metadata to %s", meta_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["db", "csv"], required=True)
    p.add_argument("--table", default="model_features")
    p.add_argument("--csv", help="CSV path when source=csv")
    p.add_argument("--target", default="cover_spread")
    p.add_argument("--task", choices=["classification", "regression"], help="Override task type")
    p.add_argument("--out-dir", default="./models")
    p.add_argument("--model-name", help="Override output model basename")
    p.add_argument("--feature-cols", help="Comma-separated feature columns to use")
    p.add_argument("--feature-cols-file", help="JSON file containing feature columns list")
    return p.parse_args()


def main():
    args = parse_args()

    if args.source == "db":
        df = load_features_from_db(args.table, args.target)
    else:
        if not args.csv:
            raise SystemExit("--csv is required when source=csv")
        df = load_features_from_csv(args.csv)

    if df.empty:
        raise SystemExit("No training data found")

    feature_cols = None
    if args.feature_cols_file:
        feature_cols = json.loads(Path(args.feature_cols_file).read_text())
    elif args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]

    model, metrics, feature_cols, feature_importance = train_model(
        df, args.target, task=args.task, feature_cols=feature_cols
    )
    logging.info("Training metrics: %s", metrics)
    logging.info("Top feature importance: %s", feature_importance[:15])

    save_artifacts(model, metrics, feature_cols, feature_importance, args.out_dir, args.target, args.model_name)


if __name__ == "__main__":
    main()
