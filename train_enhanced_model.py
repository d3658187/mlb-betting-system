#!/usr/bin/env python3
"""Train enhanced model using richer features (training_2025_enhanced.csv).

Usage:
  python train_enhanced_model.py --csv ./data/training_2025_enhanced.csv --target home_win --out-dir ./models --model-name mlb_2025_model
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

import model_trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="./data/training_2025_enhanced.csv")
    p.add_argument("--target", default="home_win")
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--out-dir", default="./models")
    p.add_argument("--model-name", default="mlb_2025_model")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("No training data found")

    model, metrics, feature_cols, feature_importance = model_trainer.train_model(
        df, target=args.target, task=args.task
    )

    logging.info("Training metrics: %s", metrics)
    logging.info("Top feature importance: %s", feature_importance[:15])

    model_trainer.save_artifacts(
        model,
        metrics,
        feature_cols,
        feature_importance,
        out_dir=args.out_dir,
        target=args.target,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
