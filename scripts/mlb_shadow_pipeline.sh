#!/bin/bash
# MLB Shadow Mode pipeline
# Daily at 10:00 (Asia/Taipei): predict tomorrow, update yesterday, print summary text.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR" "$PROJECT_DIR/data/odds"

DATE=$(date -v+1d +%Y-%m-%d)
YESTERDAY=$(date -v-1d +%Y-%m-%d)
LOG_FILE="$LOG_DIR/shadow_pipeline_${DATE}.log"

if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$PROJECT_DIR/.venv/bin/activate"
fi

cd "$PROJECT_DIR"

run_step() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

# 1) Fetch odds (if API key available in env)
run_step python3 fetch_odds_api.py --date "$DATE" --out "data/odds/the-odds-api_${DATE}.json"

# 2) Build features
run_step python3 feature_builder.py --date "$DATE" --out "data/features_${DATE}.csv"

# 3) v10 calibrated prediction
run_step python3 scripts/v10_lr_daily_predict.py \
  --date "$DATE" \
  --offline-odds-json "data/odds/the-odds-api_${DATE}.json" \
  --offline-features-csv "data/features_${DATE}.csv" \
  --calibration-method isotonic \
  --out "data/predictions_${DATE}.csv"

# 4) Upsert tracker with today's prediction rows
run_step python3 update_tracker.py --predictions "data/predictions_${DATE}.csv" --tracker "data/performance_tracker.csv"

# 5) Backfill yesterday final results
run_step python3 scripts/update_results.py --date "$YESTERDAY" --tracker "data/performance_tracker.csv"

# 6) Print daily summary (Discord-friendly plain text)
run_step python3 scripts/mlb_daily_summary.py --date "$YESTERDAY" --tracker "data/performance_tracker.csv"
