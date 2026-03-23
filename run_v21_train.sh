#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Load environment variables
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON_BIN="$(command -v python3 || command -v python)"

$PYTHON_BIN - <<'PY'
import os, sys
print("Python:", sys.executable)
print("DATABASE_URL set:", bool(os.getenv("DATABASE_URL")))
try:
    import pybaseball
    print("pybaseball:", pybaseball.__version__)
except Exception as e:
    print("pybaseball import error:", e)
PY

# Determine date (default: latest game_date in DB)
TARGET_DATE="${1:-}"
if [ -z "$TARGET_DATE" ]; then
  TARGET_DATE=$( $PYTHON_BIN - <<'PY'
import os
from sqlalchemy import create_engine, text
engine = create_engine(os.environ['DATABASE_URL'])
with engine.begin() as conn:
    max_date = conn.execute(text('SELECT MAX(game_date) FROM games')).scalar()
    print(max_date)
PY
  )
fi
TARGET_DATE=${TARGET_DATE:-$(date +%F)}
YEAR="${TARGET_DATE:0:4}"

echo "[v2.1] Running Fangraphs (pybaseball) loader for season ${YEAR}"
if ! $PYTHON_BIN fangraphs_crawler.py --season "$YEAR" --mode both --store-db --init-db; then
  echo "[v2.1] Fangraphs loader failed; continuing without fresh Fangraphs data."
fi

echo "[v2.1] Building features for ${TARGET_DATE}"
$PYTHON_BIN feature_builder.py --date "$TARGET_DATE" --write-db

# Determine training target for classification
COVER_COUNT=$( $PYTHON_BIN - <<'PY'
import os
from sqlalchemy import create_engine, text
engine = create_engine(os.environ['DATABASE_URL'])
with engine.begin() as conn:
    count = conn.execute(text('SELECT COUNT(*) FROM model_features WHERE cover_spread IS NOT NULL')).scalar()
    print(count)
PY
)

if [ "$COVER_COUNT" != "0" ]; then
  echo "[v2.1] Training cover_spread model"
  $PYTHON_BIN model_trainer.py --source db --table model_features --target cover_spread --task classification --out-dir ./models
else
  echo "[v2.1] cover_spread has no labels; training home_win instead"
  $PYTHON_BIN model_trainer.py --source db --table model_features --target home_win --task classification --out-dir ./models
fi

echo "[v2.1] Training run_margin model"
$PYTHON_BIN model_trainer.py --source db --table model_features --target run_margin --task regression --out-dir ./models

echo "[v2.1] Done"
