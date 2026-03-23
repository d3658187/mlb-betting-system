#!/usr/bin/env python3
"""Train an ensemble classifier for MLB targets (soft-voting).

Usage:
  python train_ensemble_model.py --csv ./data/training_2022_2024_enhanced_v6.csv --target home_win --out-dir ./models --model-name mlb_v6_ensemble
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

import model_trainer

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", default="home_win")
    p.add_argument("--out-dir", default="./models")
    p.add_argument("--model-name", default="mlb_v6_ensemble")
    p.add_argument("--feature-cols", help="Comma-separated feature columns to use")
    p.add_argument("--feature-cols-file", help="JSON file containing feature columns list")
    return p.parse_args()


def _split(df: pd.DataFrame, feature_cols: List[str], target: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    X = df[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)

    if "game_date" in df.columns:
        df_sorted = df.sort_values("game_date")
        X = df_sorted[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0)
        y = pd.to_numeric(df_sorted[target], errors="coerce").fillna(0).astype(int)
        split_idx = max(int(len(df_sorted) * 0.8), 1)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        return X_train, X_test, y_train, y_test, "time_series"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, "random"


def build_estimators(scale_pos_weight: Optional[float]) -> List[Tuple[str, object]]:
    estimators: List[Tuple[str, object]] = []

    if xgb is not None:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=900,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.2,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=4,
            scale_pos_weight=scale_pos_weight,
        )
        estimators.append(("xgb", xgb_clf))

    if lgb is not None:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=600,
            num_leaves=63,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        estimators.append(("lgb", lgb_clf))

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=400,
        l2_regularization=0.0,
    )
    estimators.append(("hgb", hgb))

    return estimators


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("No training data found")

    if args.target not in df.columns:
        raise SystemExit(f"Target column not found: {args.target}")

    df = df[df[args.target].notna()].copy()

    feature_cols = None
    if args.feature_cols_file:
        feature_cols = json.loads(Path(args.feature_cols_file).read_text())
    elif args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        feature_cols = model_trainer.infer_feature_columns(df, args.target)

    X_train, X_test, y_train, y_test, split_method = _split(df, feature_cols, args.target)

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = (neg / pos) if pos else None

    estimators = build_estimators(scale_pos_weight=scale_pos_weight)
    if len(estimators) == 1:
        model = estimators[0][1]
    else:
        model = VotingClassifier(estimators=estimators, voting="soft")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

    metrics = {
        "task": "classification",
        "split_method": split_method,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{args.model_name}.pkl"
    meta_path = out_dir / f"{args.model_name}.meta.json"

    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps({"metrics": metrics, "feature_cols": feature_cols}, indent=2))

    logging.info("Saved ensemble model to %s", model_path)
    logging.info("Saved metadata to %s", meta_path)
    logging.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
