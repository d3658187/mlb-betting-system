#!/usr/bin/env python3
"""Weekly retrain for v8 model (XGBoost + Platoon Splits).

Rules:
- Only retrain on weekly check (default Monday) unless --force
- If >50 new games since last trained, train new model
- Compare accuracy vs existing model; replace only if better
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    raise SystemExit("xgboost is required")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DEFAULT_FEATURES = [
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


def load_feature_cols(meta_path: Path) -> List[str]:
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            cols = meta.get("feature_cols")
            if cols:
                return cols
        except Exception:
            pass
    return DEFAULT_FEATURES


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}


def save_state(state_path: Path, payload: dict) -> None:
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def build_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.reindex(columns=feature_cols, fill_value=0).copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = pd.to_numeric(df.get("home_win"), errors="coerce").fillna(0).astype(int)
    return X, y


def time_series_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series) -> Tuple:
    if "game_date" in df.columns:
        df_sorted = df.sort_values("game_date")
        X_sorted = X.loc[df_sorted.index]
        y_sorted = y.loc[df_sorted.index]
        split_idx = max(int(len(df_sorted) * 0.8), 1)
        return (
            X_sorted.iloc[:split_idx],
            X_sorted.iloc[split_idx:],
            y_sorted.iloc[:split_idx],
            y_sorted.iloc[split_idx:],
        )
    split_idx = max(int(len(df) * 0.8), 1)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
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
    model.fit(X_train, y_train)
    return model


def eval_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "n_test": int(len(y_test)),
    }


def eval_old_model(model_path: Path, X_test: pd.DataFrame, y_test: pd.Series) -> Optional[dict]:
    if not model_path.exists():
        return None
    try:
        old_model = xgb.XGBClassifier()
        old_model.load_model(str(model_path))
        return eval_model(old_model, X_test, y_test)
    except Exception as exc:
        logging.warning("Failed to load old model: %s", exc)
        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--training-csv", default="./data/training_2022_2025_enhanced_v6.csv")
    p.add_argument("--model-dir", default="./models")
    p.add_argument("--model-name", default="mlb_v8_platoon")
    p.add_argument("--min-new-games", type=int, default=50)
    p.add_argument("--force", action="store_true", help="Force retrain regardless of schedule/new games")
    p.add_argument("--weekday", type=int, default=0, help="Weekday for retrain check (0=Mon)")
    return p.parse_args()


def main():
    args = parse_args()

    today = date.today()
    if not args.force and today.weekday() != args.weekday:
        logging.info("Skip retrain: weekday=%s (configured=%s)", today.weekday(), args.weekday)
        return

    csv_path = Path(args.training_csv)
    if not csv_path.exists():
        logging.warning("Training CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        logging.warning("Training CSV is empty")
        return

    df = df.dropna(subset=["home_win"]).copy()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    model_dir = Path(args.model_dir)
    model_path = model_dir / f"{args.model_name}.booster"
    meta_path = model_dir / f"{args.model_name}.meta.json"
    state_path = model_dir / f"{args.model_name}.state.json"

    feature_cols = load_feature_cols(meta_path)

    state = load_state(state_path)
    last_trained_rows = int(state.get("last_trained_rows", 0))

    current_rows = int(len(df))
    new_games = current_rows - last_trained_rows
    if not args.force and new_games <= args.min_new_games:
        logging.info("Skip retrain: new games=%d <= %d", new_games, args.min_new_games)
        return

    X, y = build_xy(df, feature_cols)
    X_train, X_test, y_train, y_test = time_series_split(df, X, y)

    model = train_model(X_train, y_train)
    new_metrics = eval_model(model, X_test, y_test)

    old_metrics = eval_old_model(model_path, X_test, y_test)

    if old_metrics and new_metrics["accuracy"] <= old_metrics["accuracy"]:
        logging.info(
            "New model not better (new=%.4f old=%.4f). Keep old model.",
            new_metrics["accuracy"],
            old_metrics["accuracy"],
        )
        state["last_checked_at"] = datetime.now().isoformat(timespec="seconds")
        state["last_checked_rows"] = current_rows
        save_state(state_path, state)
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(model_path))

    meta = {
        "model_type": type(model).__name__,
        "format": "booster",
        "metrics": {
            **new_metrics,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "feature_cols": feature_cols,
        "trained_at": date.today().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    state.update(
        {
            "last_trained_at": datetime.now().isoformat(timespec="seconds"),
            "last_trained_rows": current_rows,
            "last_checked_at": datetime.now().isoformat(timespec="seconds"),
            "last_checked_rows": current_rows,
        }
    )
    save_state(state_path, state)

    logging.info("New model saved: %s", model_path)
    logging.info("Metrics: %s", new_metrics)


if __name__ == "__main__":
    main()
