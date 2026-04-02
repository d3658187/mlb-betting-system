#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "mlb.db"
MODEL_PATH = ROOT / "models" / "mlb_v10_lr.joblib"

V10_FEATURES = [
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
]

ABBR_ALIAS = {
    "AZ": "ARI",
    "SD": "SDP",
    "SF": "SFG",
    "TB": "TBR",
    "KC": "KCR",
    "WSH": "WSN",
    "CWS": "CHW",
}


def confidence_tier(prob: float) -> str:
    if pd.isna(prob):
        return "LOW"
    if prob > 0.65 or prob < 0.35:
        return "HIGH"
    if prob > 0.55 or prob < 0.45:
        return "MEDIUM"
    return "LOW"


def summarize_confidence(y_true: pd.Series, probs: np.ndarray) -> dict:
    eval_df = pd.DataFrame({"y": pd.to_numeric(y_true, errors="coerce").astype(int), "p": probs})
    eval_df["tier"] = eval_df["p"].apply(confidence_tier)

    out = {}
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        sub = eval_df[eval_df["tier"] == tier]
        out[tier] = {
            "games": int(len(sub)),
            "accuracy": float(((((sub["p"] >= 0.5) & (sub["y"] == 1)) | ((sub["p"] < 0.5) & (sub["y"] == 0))).mean())) if len(sub) else None,
            "avg_prob": float(sub["p"].mean()) if len(sub) else None,
        }
    return out


def _safe_num(v):
    try:
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _canon_abbr(v: str) -> str:
    s = str(v).upper().strip()
    return ABBR_ALIAS.get(s, s)


def _diff(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)


def load_schedule(target_date: str) -> pd.DataFrame:
    src = DATA_DIR / "mlb_stats_api" / "daily" / f"games_{target_date}.csv"
    if not src.exists():
        raise FileNotFoundError(f"schedule file missing: {src}")

    df = pd.read_csv(src)
    out = pd.DataFrame(
        {
            "game_id": pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64"),
            "home_team": df["home_team"].astype(str),
            "away_team": df["away_team"].astype(str),
            "home_team_id": pd.to_numeric(df["home_team_id"], errors="coerce").astype("Int64"),
            "away_team_id": pd.to_numeric(df["away_team_id"], errors="coerce").astype("Int64"),
            "home_pitcher": df["home_probable_pitcher_name"].fillna("TBD").astype(str),
            "away_pitcher": df["away_probable_pitcher_name"].fillna("TBD").astype(str),
        }
    )
    return out


def load_raw_features(target_date: str) -> pd.DataFrame:
    path = DATA_DIR / f"features_{target_date}_raw.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"raw feature file missing: {path}. Run feature_builder.py first."
        )
    df = pd.read_csv(path)
    df["mlb_game_id"] = pd.to_numeric(df.get("mlb_game_id"), errors="coerce").astype("Int64")
    return df


def load_prior_team_summary(prior_year: int = 2025) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        s = pd.read_sql_query(
            """
            SELECT
              season_year,
              team_abbrev,
              team_id,
              bat_r,
              bat_wrc_plus,
              pit_g,
              pit_r,
              pit_era,
              pit_whip,
              pit_fip,
              pit_xfip,
              pit_siera,
              pit_k_pct,
              pit_bb_pct,
              pit_kbb_pct,
              win_pct
            FROM team_season_summary
            WHERE season_year = ?
            """,
            conn,
            params=(prior_year,),
        )

    if s.empty:
        raise RuntimeError(f"team_season_summary missing season_year={prior_year}")

    s["team_abbrev"] = s["team_abbrev"].astype(str).str.upper().str.strip()
    s["pit_g"] = pd.to_numeric(s["pit_g"], errors="coerce").replace(0, np.nan)
    s["proxy_runs_scored_pg"] = pd.to_numeric(s["bat_r"], errors="coerce") / s["pit_g"]
    s["proxy_runs_allowed_pg"] = pd.to_numeric(s["pit_r"], errors="coerce") / s["pit_g"]
    s["proxy_run_diff_pg"] = s["proxy_runs_scored_pg"] - s["proxy_runs_allowed_pg"]
    return s


def build_h2h_map(target_date: str, valid_teams: set[str]) -> Dict[Tuple[str, str], dict]:
    with sqlite3.connect(DB_PATH) as conn:
        hist = pd.read_sql_query(
            """
            SELECT game_date, home_team, away_team, home_score, away_score
            FROM games
            WHERE game_date < ?
              AND home_score IS NOT NULL
              AND away_score IS NOT NULL
            """,
            conn,
            params=(target_date,),
        )

    if hist.empty:
        return {}

    hist["home_team"] = hist["home_team"].astype(str).map(_canon_abbr)
    hist["away_team"] = hist["away_team"].astype(str).map(_canon_abbr)
    hist = hist[
        hist["home_team"].isin(valid_teams) & hist["away_team"].isin(valid_teams)
    ].copy()

    h2h: Dict[Tuple[str, str], dict] = {}
    for _, r in hist.iterrows():
        home = r["home_team"]
        away = r["away_team"]
        hs = _safe_num(r["home_score"])
        aw = _safe_num(r["away_score"])
        if pd.isna(hs) or pd.isna(aw):
            continue

        key = tuple(sorted((home, away)))
        rec = h2h.setdefault(
            key,
            {
                "games": 0,
                "team": {
                    key[0]: {"wins": 0.0, "rs": 0.0, "ra": 0.0},
                    key[1]: {"wins": 0.0, "rs": 0.0, "ra": 0.0},
                },
            },
        )

        rec["games"] += 1
        rec["team"][home]["rs"] += hs
        rec["team"][home]["ra"] += aw
        rec["team"][away]["rs"] += aw
        rec["team"][away]["ra"] += hs

        if hs > aw:
            rec["team"][home]["wins"] += 1.0
        elif aw > hs:
            rec["team"][away]["wins"] += 1.0
        else:
            rec["team"][home]["wins"] += 0.5
            rec["team"][away]["wins"] += 0.5

    return h2h


def build_v10_features(target_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule = load_schedule(target_date)
    raw = load_raw_features(target_date)
    summary = load_prior_team_summary(2025)

    summary_by_abbr = {r["team_abbrev"]: r for _, r in summary.iterrows()}
    team_id_to_abbr = {
        int(r["team_id"]): r["team_abbrev"]
        for _, r in summary.iterrows()
        if pd.notna(r["team_id"])
    }

    h2h_map = build_h2h_map(target_date, set(summary_by_abbr.keys()))

    merged = schedule.merge(
        raw,
        how="left",
        left_on="game_id",
        right_on="mlb_game_id",
        suffixes=("", "_raw"),
    )

    rows = []
    for _, r in merged.iterrows():
        home_out = str(r["home_team"]).upper().strip()
        away_out = str(r["away_team"]).upper().strip()

        home_team_id = _safe_num(r.get("home_team_id"))
        away_team_id = _safe_num(r.get("away_team_id"))

        home_abbr = _canon_abbr(home_out)
        away_abbr = _canon_abbr(away_out)
        if not pd.isna(home_team_id):
            home_abbr = team_id_to_abbr.get(int(home_team_id), home_abbr)
        if not pd.isna(away_team_id):
            away_abbr = team_id_to_abbr.get(int(away_team_id), away_abbr)

        home_proxy = summary_by_abbr.get(home_abbr, {})
        away_proxy = summary_by_abbr.get(away_abbr, {})

        # Pitching blocks (fallback to 2025 team summary)
        home_p_xfip = _safe_num(r.get("home_p_xFIP"))
        away_p_xfip = _safe_num(r.get("away_p_xFIP"))
        if pd.isna(home_p_xfip):
            home_p_xfip = _safe_num(home_proxy.get("pit_xfip"))
        if pd.isna(away_p_xfip):
            away_p_xfip = _safe_num(away_proxy.get("pit_xfip"))

        home_p_era = _safe_num(r.get("home_p_ERA"))
        away_p_era = _safe_num(r.get("away_p_ERA"))
        if pd.isna(home_p_era):
            home_p_era = _safe_num(home_proxy.get("pit_era"))
        if pd.isna(away_p_era):
            away_p_era = _safe_num(away_proxy.get("pit_era"))

        home_p_whip = _safe_num(r.get("home_p_WHIP"))
        away_p_whip = _safe_num(r.get("away_p_WHIP"))
        if pd.isna(home_p_whip):
            home_p_whip = _safe_num(home_proxy.get("pit_whip"))
        if pd.isna(away_p_whip):
            away_p_whip = _safe_num(away_proxy.get("pit_whip"))

        home_p_fip = _safe_num(r.get("home_p_FIP"))
        away_p_fip = _safe_num(r.get("away_p_FIP"))
        if pd.isna(home_p_fip):
            home_p_fip = _safe_num(home_proxy.get("pit_fip"))
        if pd.isna(away_p_fip):
            away_p_fip = _safe_num(away_proxy.get("pit_fip"))

        home_p_siera = _safe_num(r.get("home_p_SIERA"))
        away_p_siera = _safe_num(r.get("away_p_SIERA"))
        if pd.isna(home_p_siera):
            home_p_siera = _safe_num(home_proxy.get("pit_siera"))
        if pd.isna(away_p_siera):
            away_p_siera = _safe_num(away_proxy.get("pit_siera"))

        home_p_kpct = _safe_num(r.get("home_p_K%"))
        away_p_kpct = _safe_num(r.get("away_p_K%"))
        if pd.isna(home_p_kpct):
            home_p_kpct = _safe_num(home_proxy.get("pit_k_pct"))
        if pd.isna(away_p_kpct):
            away_p_kpct = _safe_num(away_proxy.get("pit_k_pct"))

        home_p_bbpct = _safe_num(r.get("home_p_BB%"))
        away_p_bbpct = _safe_num(r.get("away_p_BB%"))
        if pd.isna(home_p_bbpct):
            home_p_bbpct = _safe_num(home_proxy.get("pit_bb_pct"))
        if pd.isna(away_p_bbpct):
            away_p_bbpct = _safe_num(away_proxy.get("pit_bb_pct"))

        home_p_kbbpct = _safe_num(r.get("home_p_K-BB%"))
        away_p_kbbpct = _safe_num(r.get("away_p_K-BB%"))
        if pd.isna(home_p_kbbpct):
            home_p_kbbpct = _safe_num(home_proxy.get("pit_kbb_pct"))
        if pd.isna(away_p_kbbpct):
            away_p_kbbpct = _safe_num(away_proxy.get("pit_kbb_pct"))

        # Batting wRC+
        home_wrc = _safe_num(r.get("home_bat_wRC+"))
        away_wrc = _safe_num(r.get("away_bat_wRC+"))
        if pd.isna(home_wrc):
            home_wrc = _safe_num(home_proxy.get("bat_wrc_plus"))
        if pd.isna(away_wrc):
            away_wrc = _safe_num(away_proxy.get("bat_wrc_plus"))

        # Roll5 run diff with sample-size guard
        home_roll5_raw = _safe_num(r.get("home_roll5_run_diff_mean"))
        away_roll5_raw = _safe_num(r.get("away_roll5_run_diff_mean"))
        home_roll5 = home_roll5_raw
        away_roll5 = away_roll5_raw
        home_roll5_n = _safe_num(r.get("home_roll5_games_count"))
        away_roll5_n = _safe_num(r.get("away_roll5_games_count"))
        home_prior_proxy = bool(r.get("home_prior_season_proxy", False))
        away_prior_proxy = bool(r.get("away_prior_season_proxy", False))

        home_roll5_needs_proxy = (
            pd.isna(home_roll5)
            or pd.isna(home_roll5_n)
            or home_roll5_n < 5
            or (home_prior_proxy and abs(home_roll5) < 1e-12)
        )
        away_roll5_needs_proxy = (
            pd.isna(away_roll5)
            or pd.isna(away_roll5_n)
            or away_roll5_n < 5
            or (away_prior_proxy and abs(away_roll5) < 1e-12)
        )

        if home_roll5_needs_proxy:
            home_roll5 = _safe_num(home_proxy.get("proxy_run_diff_pg"))
        if away_roll5_needs_proxy:
            away_roll5 = _safe_num(away_proxy.get("proxy_run_diff_pg"))

        # H2H with fallback to 2025 team-level proxy
        h2h_key = tuple(sorted((home_abbr, away_abbr)))
        h2h_rec = h2h_map.get(h2h_key)

        if h2h_rec and h2h_rec.get("games", 0) >= 3:
            g = float(h2h_rec["games"])
            home_stats = h2h_rec["team"][home_abbr]
            away_stats = h2h_rec["team"][away_abbr]

            home_h2h_win = home_stats["wins"] / g
            away_h2h_win = away_stats["wins"] / g
            home_h2h_rs = home_stats["rs"] / g
            away_h2h_rs = away_stats["rs"] / g
            home_h2h_ra = home_stats["ra"] / g
            away_h2h_ra = away_stats["ra"] / g
            h2h_proxy_used = False
        else:
            home_h2h_win = _safe_num(home_proxy.get("win_pct"))
            away_h2h_win = _safe_num(away_proxy.get("win_pct"))
            home_h2h_rs = _safe_num(home_proxy.get("proxy_runs_scored_pg"))
            away_h2h_rs = _safe_num(away_proxy.get("proxy_runs_scored_pg"))
            home_h2h_ra = _safe_num(home_proxy.get("proxy_runs_allowed_pg"))
            away_h2h_ra = _safe_num(away_proxy.get("proxy_runs_allowed_pg"))
            h2h_proxy_used = True

        row = {
            "game_id": int(r["game_id"]) if pd.notna(r["game_id"]) else pd.NA,
            "home_team": home_out,
            "away_team": away_out,
            "home_pitcher": r.get("home_pitcher", "TBD"),
            "away_pitcher": r.get("away_pitcher", "TBD"),
            "diff_p_xFIP": _diff(home_p_xfip, away_p_xfip),
            "diff_p_K-BB%": _diff(home_p_kbbpct, away_p_kbbpct),
            "diff_bat_wRC+": _diff(home_wrc, away_wrc),
            "diff_p_ERA": _diff(home_p_era, away_p_era),
            "diff_p_WHIP": _diff(home_p_whip, away_p_whip),
            "diff_p_FIP": _diff(home_p_fip, away_p_fip),
            "diff_p_SIERA": _diff(home_p_siera, away_p_siera),
            "diff_p_K%": _diff(home_p_kpct, away_p_kpct),
            "diff_p_BB%": _diff(home_p_bbpct, away_p_bbpct),
            "home_roll5_run_diff_mean": home_roll5,
            "diff_roll5_run_diff_mean": _diff(home_roll5, away_roll5),
            "home_h2h_win_pct": home_h2h_win,
            "away_h2h_win_pct": away_h2h_win,
            "diff_h2h_win_pct": _diff(home_h2h_win, away_h2h_win),
            "diff_h2h_runs_scored_avg": _diff(home_h2h_rs, away_h2h_rs),
            "diff_h2h_runs_allowed_avg": _diff(home_h2h_ra, away_h2h_ra),
            "h2h_proxy_used": h2h_proxy_used,
            "home_roll5_proxy_used": home_roll5_needs_proxy,
            "away_roll5_proxy_used": away_roll5_needs_proxy,
        }
        rows.append(row)

    features_df = pd.DataFrame(rows)

    # Ensure numeric feature types
    for c in V10_FEATURES:
        features_df[c] = pd.to_numeric(features_df[c], errors="coerce")

    return schedule, features_df


def train_v10_model(calibration_method: str = "isotonic", calibration_cv: int = 5) -> tuple[CalibratedClassifierCV, dict]:
    train_path = DATA_DIR / "training_features_v10.csv"
    df = pd.read_csv(train_path)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    train = df[df["game_date"] < "2025-09-01"].copy()
    test = df[df["game_date"] >= "2025-09-01"].copy()

    X_train = train[V10_FEATURES].apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train["home_win"], errors="coerce").astype(int)
    X_test = test[V10_FEATURES].apply(pd.to_numeric, errors="coerce")
    y_test = pd.to_numeric(test["home_win"], errors="coerce").astype(int)

    base_lr = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("lr", LogisticRegression(max_iter=1000, C=1.0)),
        ]
    )

    # Baseline (uncalibrated)
    base_lr.fit(X_train, y_train)
    raw_prob = base_lr.predict_proba(X_test)[:, 1]

    # Calibrated probabilities (isotonic/sigmoid)
    calibrated = CalibratedClassifierCV(base_lr, method=calibration_method, cv=calibration_cv)
    calibrated.fit(X_train, y_train)
    cal_prob = calibrated.predict_proba(X_test)[:, 1]

    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "calibration_method": calibration_method,
        "calibration_cv": int(calibration_cv),
        "auc_raw": float(roc_auc_score(y_test, raw_prob)),
        "auc_calibrated": float(roc_auc_score(y_test, cal_prob)),
        "brier_raw": float(brier_score_loss(y_test, raw_prob)),
        "brier_calibrated": float(brier_score_loss(y_test, cal_prob)),
        "tiers_raw": summarize_confidence(y_test, raw_prob),
        "tiers_calibrated": summarize_confidence(y_test, cal_prob),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, MODEL_PATH)
    return calibrated, metrics


def predict_for_date(model, target_date: str, features_df: pd.DataFrame) -> pd.DataFrame:
    probs = model.predict_proba(features_df[V10_FEATURES])[:, 1]

    pred = features_df[
        ["game_id", "home_team", "away_team", "home_pitcher", "away_pitcher"]
    ].copy()
    pred["prediction_date"] = target_date
    pred["home_win_prob"] = probs
    pred["away_win_prob"] = 1.0 - probs
    pred["market_home_prob"] = "N/A"
    pred["confidence_tier"] = pred["home_win_prob"].apply(confidence_tier)

    pred = pred[
        [
            "prediction_date",
            "game_id",
            "home_team",
            "away_team",
            "home_pitcher",
            "away_pitcher",
            "home_win_prob",
            "away_win_prob",
            "market_home_prob",
            "confidence_tier",
        ]
    ].sort_values("game_id")
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated v10 LR and predict one MLB slate")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--calibration-method", choices=["isotonic", "sigmoid"], default="isotonic")
    parser.add_argument("--calibration-cv", type=int, default=5)
    args = parser.parse_args()

    target_date = args.date

    schedule, features_df = build_v10_features(target_date)

    schedule_out = DATA_DIR / f"schedule_{target_date}.csv"
    schedule[
        ["game_id", "home_team", "away_team", "home_pitcher", "away_pitcher"]
    ].to_csv(schedule_out, index=False)

    features_out = DATA_DIR / f"features_{target_date}_v10.csv"
    features_df.to_csv(features_out, index=False)

    model, metrics = train_v10_model(
        calibration_method=args.calibration_method,
        calibration_cv=args.calibration_cv,
    )
    pred = predict_for_date(model, target_date, features_df)

    pred_out = DATA_DIR / f"predictions_{target_date}.csv"
    pred.to_csv(pred_out, index=False)

    print(f"Schedule: {schedule_out}")
    print(f"Features: {features_out}")
    print(f"Model: {MODEL_PATH}")
    print(f"Predictions: {pred_out}")
    print(
        "Test AUC raw/calibrated: "
        f"{metrics['auc_raw']:.4f} / {metrics['auc_calibrated']:.4f} "
        f"(train={metrics['n_train']}, test={metrics['n_test']})"
    )
    print(
        "Brier raw/calibrated: "
        f"{metrics['brier_raw']:.4f} / {metrics['brier_calibrated']:.4f} "
        f"[{metrics['calibration_method']}, cv={metrics['calibration_cv']}]"
    )
    print("Tier accuracy (calibrated):", metrics["tiers_calibrated"])


if __name__ == "__main__":
    main()
