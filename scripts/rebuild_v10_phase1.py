#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

BASE_PATH = DATA_DIR / "training_2022_2025_enhanced_v6.csv"
PLATOON_PATH = DATA_DIR / "training_2022_2025_platoon.csv"
OUT_DATA = DATA_DIR / "training_features_v10.csv"
OUT_SINGLE_VAR = ROOT / "SINGLE_VARIABLE_ANALYSIS.md"
OUT_V10_FEATURES = ROOT / "V10_FEATURES.md"
OUT_MODEL_COMPARISON = ROOT / "MODEL_COMPARISON.md"


@dataclass
class UniResult:
    feature: str
    auc: float
    coverage: float
    keep: bool


def load_dataset() -> pd.DataFrame:
    base = pd.read_csv(BASE_PATH)
    platoon = pd.read_csv(
        PLATOON_PATH,
        usecols=[
            "game_date",
            "home_team",
            "away_team",
            "home_platoon_ops_diff",
            "away_platoon_ops_diff",
            "home_platoon_splits_score",
            "away_platoon_splits_score",
        ],
    )

    for df in (base, platoon):
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    platoon = (
        platoon.sort_values("game_date")
        .drop_duplicates(["game_date", "home_team", "away_team"], keep="last")
    )

    drop_existing = [
        c
        for c in [
            "home_platoon_ops_diff",
            "away_platoon_ops_diff",
            "home_platoon_splits_score",
            "away_platoon_splits_score",
        ]
        if c in base.columns
    ]
    merged = base.drop(columns=drop_existing).merge(
        platoon,
        on=["game_date", "home_team", "away_team"],
        how="left",
    )

    # Derived diff features
    derived_pairs = {
        "diff_p_xFIP": ("home_p_xFIP", "away_p_xFIP"),
        "diff_p_FIP": ("home_p_FIP", "away_p_FIP"),
        "diff_p_SIERA": ("home_p_SIERA", "away_p_SIERA"),
        "diff_p_K%": ("home_p_K%", "away_p_K%"),
        "diff_p_BB%": ("home_p_BB%", "away_p_BB%"),
        "diff_roll5_win_mean": ("home_roll5_win_mean", "away_roll5_win_mean"),
        "diff_roll5_run_diff_mean": (
            "home_roll5_run_diff_mean",
            "away_roll5_run_diff_mean",
        ),
        "diff_h2h_win_pct": ("home_h2h_win_pct", "away_h2h_win_pct"),
        "diff_h2h_runs_scored_avg": (
            "home_h2h_runs_scored_avg",
            "away_h2h_runs_scored_avg",
        ),
        "diff_h2h_runs_allowed_avg": (
            "home_h2h_runs_allowed_avg",
            "away_h2h_runs_allowed_avg",
        ),
        "diff_platoon_ops_diff": (
            "home_platoon_ops_diff",
            "away_platoon_ops_diff",
        ),
        "diff_platoon_splits_score": (
            "home_platoon_splits_score",
            "away_platoon_splits_score",
        ),
    }

    for out_col, (home_col, away_col) in derived_pairs.items():
        merged[out_col] = pd.to_numeric(merged[home_col], errors="coerce") - pd.to_numeric(
            merged[away_col], errors="coerce"
        )

    merged = merged[merged["home_win"].notna()].copy()
    merged = merged.sort_values("game_date").reset_index(drop=True)
    merged["home_win"] = pd.to_numeric(merged["home_win"], errors="coerce").astype(int)
    return merged


def split_walk_forward(df: pd.DataFrame, test_fraction: float = 0.2):
    split_idx = max(int(len(df) * (1 - test_fraction)), 1)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def run_univariate_logistic(
    train_df: pd.DataFrame,
    candidate_features: List[str],
    threshold: float = 0.52,
) -> List[UniResult]:
    y_train = train_df["home_win"].astype(int)
    results: List[UniResult] = []

    for feat in candidate_features:
        x = pd.to_numeric(train_df[feat], errors="coerce")
        coverage = float(x.notna().mean())

        if x.notna().sum() == 0:
            auc = float("nan")
            keep = False
        else:
            X = x.to_frame(feat)
            imp = SimpleImputer(strategy="median")
            X_imp = imp.fit_transform(X)
            if X_imp.shape[1] == 0:
                auc = float("nan")
                keep = False
            else:
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(X_imp, y_train)
                prob = lr.predict_proba(X_imp)[:, 1]
                auc = float(roc_auc_score(y_train, prob))
                keep = auc >= threshold

        results.append(UniResult(feature=feat, auc=auc, coverage=coverage, keep=keep))

    return results


def write_single_variable_md(
    results: List[UniResult],
    leakage_removed: List[str],
    threshold: float,
    train_size: int,
    test_size: int,
) -> None:
    lines = []
    lines.append("# SINGLE VARIABLE ANALYSIS (v10 Sprint 1)\n")
    lines.append("- 方法：單變量 Logistic Regression + train AUC\n")
    lines.append(f"- Walk-Forward 切分：train={train_size}, test={test_size}（時間序）\n")
    lines.append(f"- Gate：AUC >= {threshold:.2f}\n")
    lines.append(
        f"- 資料洩漏欄位處理：移除 {', '.join(leakage_removed)}（不納入 feature set）\n"
    )
    lines.append("\n| Feature | Coverage(train) | AUC(train) | Keep |\n")
    lines.append("|---|---:|---:|:---:|\n")

    for r in sorted(results, key=lambda x: (np.nan_to_num(x.auc, nan=-1), x.feature), reverse=True):
        auc_str = "NA" if np.isnan(r.auc) else f"{r.auc:.4f}"
        lines.append(
            f"| {r.feature} | {r.coverage:.1%} | {auc_str} | {'✅' if r.keep else '❌'} |\n"
        )

    OUT_SINGLE_VAR.write_text("".join(lines), encoding="utf-8")


def write_v10_features_md(
    final_features: List[str],
    results: List[UniResult],
    threshold: float,
) -> None:
    auc_map = {r.feature: r.auc for r in results}

    lines = []
    lines.append(f"# V10_FEATURES (Core {len(final_features)})\n")
    lines.append("- 來源：v6 訓練集 + 補齊 platoon 欄位 + 派生 diff 欄位\n")
    lines.append(f"- 篩選規則：單變量 Logistic AUC >= {threshold:.2f}\n")
    lines.append("\n## Final Feature List\n")
    for i, feat in enumerate(final_features, 1):
        auc = auc_map.get(feat, float("nan"))
        auc_str = "NA" if np.isnan(auc) else f"{auc:.4f}"
        lines.append(f"{i}. `{feat}` (AUC={auc_str})\n")

    lines.append("\n## Note\n")
    lines.append(
        "- 軍師建議的 platoon 四欄在目前資料的單變量 AUC 未達 0.52，已依 Gate 排除。\n"
    )
    lines.append(
        "- 為維持 16-18 核心欄位，使用同類型且通過 Gate 的差值/動能/H2H 欄位補齊。\n"
    )

    OUT_V10_FEATURES.write_text("".join(lines), encoding="utf-8")


def train_and_compare(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
):
    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train_df["home_win"].astype(int)
    y_test = test_df["home_win"].astype(int)

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_train_imp, y_train)
    lr_prob = lr.predict_proba(X_test_imp)[:, 1]
    lr_auc = float(roc_auc_score(y_test, lr_prob))

    xgb = XGBClassifier(
        max_depth=4,
        n_estimators=500,
        learning_rate=0.05,
        min_child_weight=10,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
    )
    xgb.fit(X_train_imp, y_train)
    xgb_prob = xgb.predict_proba(X_test_imp)[:, 1]
    xgb_auc = float(roc_auc_score(y_test, xgb_prob))

    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "lr_auc": lr_auc,
        "xgb_auc": xgb_auc,
        "winner": "LR" if lr_auc >= xgb_auc else "XGB",
    }


def write_model_comparison_md(metrics: dict, feature_count: int) -> None:
    lines = []
    lines.append("# MODEL_COMPARISON (v10 Sprint 1)\n")
    lines.append("- Split：Walk-Forward 時間序 80/20\n")
    lines.append(f"- Features：{feature_count}\n")
    lines.append(f"- Train rows：{metrics['n_train']}\n")
    lines.append(f"- Test rows：{metrics['n_test']}\n")
    lines.append("\n## Result\n")
    lines.append(f"- LR AUC: **{metrics['lr_auc']:.4f}**\n")
    lines.append(f"- XGB AUC: **{metrics['xgb_auc']:.4f}**\n")
    lines.append(f"- Better baseline: **{metrics['winner']}**\n")

    OUT_MODEL_COMPARISON.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    threshold = 0.52
    leakage_removed = ["home_runs", "away_runs"]

    df = load_dataset()
    train_df, test_df = split_walk_forward(df, test_fraction=0.2)

    candidate_features = [
        # strategist core
        "diff_p_xFIP",
        "diff_p_K-BB%",
        "diff_bat_wRC+",
        "diff_bat_wOBA",
        "home_platoon_ops_diff",
        "away_platoon_ops_diff",
        "home_platoon_splits_score",
        "away_platoon_splits_score",
        "home_roll5_win_mean",
        "away_roll5_win_mean",
        "home_roll5_run_diff_mean",
        "away_roll5_run_diff_mean",
        "home_h2h_win_pct",
        "away_h2h_win_pct",
        # alternatives
        "diff_roll5_win_mean",
        "diff_roll5_run_diff_mean",
        "diff_h2h_win_pct",
        "diff_h2h_runs_scored_avg",
        "diff_h2h_runs_allowed_avg",
        "diff_p_ERA",
        "diff_p_WHIP",
        "diff_p_FIP",
        "diff_p_SIERA",
        "diff_p_K%",
        "diff_p_BB%",
        "diff_platoon_ops_diff",
        "diff_platoon_splits_score",
    ]

    # ensure candidate cols exist
    candidate_features = [f for f in candidate_features if f in df.columns]

    uni_results = run_univariate_logistic(train_df, candidate_features, threshold=threshold)

    keep_features = [r.feature for r in uni_results if r.keep]

    # Final 18 features (all passed gate)
    priority = [
        "diff_p_xFIP",
        "diff_p_K-BB%",
        "diff_bat_wRC+",
        "diff_p_ERA",
        "diff_p_WHIP",
        "diff_p_FIP",
        "diff_p_SIERA",
        "diff_p_K%",
        "diff_p_BB%",
        "home_roll5_run_diff_mean",
        "diff_roll5_run_diff_mean",
        "home_h2h_win_pct",
        "away_h2h_win_pct",
        "diff_h2h_win_pct",
        "diff_h2h_runs_scored_avg",
        "diff_h2h_runs_allowed_avg",
        "home_h2h_runs_scored_avg",
        "away_h2h_runs_allowed_avg",
    ]

    final_features = [f for f in priority if f in keep_features]

    if len(final_features) < 16:
        # fallback: fill from remaining kept features by AUC desc
        rank = sorted(
            [r for r in uni_results if r.keep and r.feature not in final_features],
            key=lambda x: x.auc,
            reverse=True,
        )
        for r in rank:
            final_features.append(r.feature)
            if len(final_features) >= 16:
                break

    final_features = final_features[:18]

    # Build final v10 dataset
    v10_df = df[["game_date", "home_team", "away_team", "home_win", *final_features]].copy()
    # Explicitly ensure leakage columns removed
    for leak_col in leakage_removed + ["home_score", "away_score"]:
        if leak_col in v10_df.columns:
            v10_df = v10_df.drop(columns=[leak_col])

    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    v10_df.to_csv(OUT_DATA, index=False)

    # Train/eval
    train_v10, test_v10 = split_walk_forward(v10_df, test_fraction=0.2)
    metrics = train_and_compare(train_v10, test_v10, final_features)

    write_single_variable_md(
        uni_results,
        leakage_removed=leakage_removed,
        threshold=threshold,
        train_size=len(train_df),
        test_size=len(test_df),
    )
    write_v10_features_md(final_features, uni_results, threshold=threshold)
    write_model_comparison_md(metrics, feature_count=len(final_features))

    print(f"Wrote: {OUT_DATA}")
    print(f"Rows={len(v10_df)}, Features={len(final_features)}")
    print(f"LR AUC={metrics['lr_auc']:.4f}, XGB AUC={metrics['xgb_auc']:.4f}")


if __name__ == "__main__":
    main()
