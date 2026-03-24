#!/bin/bash
set -euo pipefail

REPO_DIR="/Users/duste/.openclaw/workspace-zongzhihui/mlb_betting_system"
PYTHON_BIN="$REPO_DIR/.venv/bin/python"

cd "$REPO_DIR"

"$PYTHON_BIN" fangraphs_platoon_splits_crawler.py \
  --seasons "2022-2025" \
  --pitcher-csv "$REPO_DIR/data/pybaseball/starting_pitchers_{season}.csv" \
  --out "$REPO_DIR/data/pybaseball/platoon_splits_all.csv"
