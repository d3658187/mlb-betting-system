#!/bin/bash
# MLB Daily Prediction Pipeline
# 1. Fetch probable starters via pybaseball
# 2. Fetch odds via The Odds API
# 3. Run daily predictor
# 4. Post results to Discord

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"
OUT_DIR="$PROJECT_DIR/data/pybaseball/daily"
ODDS_DIR="$PROJECT_DIR/data/odds"
LOG_DIR="/tmp/mlb_daily"
TODAY=$(date +%Y-%m-%d)

mkdir -p "$LOG_DIR" "$OUT_DIR" "$ODDS_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" | tee -a "$LOG_DIR/mlb_daily_$TODAY.log"; }

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export DATABASE_URL="${DATABASE_URL:-postgresql://duste:duste123@localhost:5432/mlb_betting}"
export THE_ODDS_API_KEY="${THE_ODDS_API_KEY:-a7c8d03285ba725e76f4bcf9583563f2}"

log "=== MLB Daily Pipeline: $TODAY ==="

# Step 1: Fetch probable starters
log "Step 1: Fetching probable starters..."
python3 "$PROJECT_DIR/pybaseball_daily_crawler.py" \
    --date "$TODAY" \
    --out-dir "$OUT_DIR" \
    --include-probable-starters \
    2>&1 | tee -a "$LOG_DIR/mlb_daily_$TODAY.log" || true

STARTERS_FILE="$OUT_DIR/probable_starters_${TODAY}.csv"
if [ -f "$STARTERS_FILE" ]; then
    log "  Found $STARTERS_FILE ($(wc -l < $STARTERS_FILE) lines)"
else
    log "  WARNING: No starters file found, will use model defaults"
fi

# Step 2: Fetch The Odds API
log "Step 2: Fetching odds from The Odds API..."
python3 "$PROJECT_DIR/fetch_odds_api.py" \
    --date "$TODAY" \
    2>&1 | tee -a "$LOG_DIR/mlb_daily_$TODAY.log" || true

ODDS_FILE="$ODDS_DIR/the-odds-api_${TODAY}.json"
if [ -f "$ODDS_FILE" ]; then
    log "  Found $ODDS_FILE"
else
    log "  WARNING: No odds file found"
fi

# Step 3: Run daily predictor
log "Step 3: Running daily predictor..."
if [ -f "$ODDS_FILE" ]; then
    python3 "$PROJECT_DIR/daily_predictor.py" \
        --date "$TODAY" \
        --odds-api \
        2>&1 | tee -a "$LOG_DIR/mlb_daily_$TODAY.log"
else
    log "  Skipping predictor (no odds data)"
fi

# Step 4: Fetch completed results (yesterday)
RESULT_DATE=$(date -v-1d +%Y-%m-%d)
RESULTS_DIR="$PROJECT_DIR/data/results"
mkdir -p "$RESULTS_DIR"
log "Step 4: Fetching completed results for $RESULT_DATE..."
python3 "$PROJECT_DIR/fetch_results.py" \
    --date "$RESULT_DATE" \
    2>&1 | tee -a "$LOG_DIR/mlb_daily_$TODAY.log" || true

# Step 5: Update training data
log "Step 5: Updating training data..."
python3 "$PROJECT_DIR/update_training_data.py" \
    --date "$RESULT_DATE" \
    2>&1 | tee -a "$LOG_DIR/mlb_daily_$TODAY.log" || true

# Step 6: Weekly retrain check
log "Step 6: Weekly retrain check..."
python3 "$PROJECT_DIR/retrain_if_needed.py" \
    2>&1 | tee -a "$LOG_DIR/mlb_daily_$TODAY.log" || true

log "=== Pipeline complete: $TODAY ==="
