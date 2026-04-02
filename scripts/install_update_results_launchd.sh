#!/bin/bash
set -euo pipefail

PROJECT_DIR="/Users/duste/.openclaw/workspace-zongzhihui/mlb_betting_system"
PLIST_SRC="$PROJECT_DIR/scripts/launchd/ai.openclaw.mlb.update-results.plist"
PLIST_DST="$HOME/Library/LaunchAgents/ai.openclaw.mlb.update-results.plist"

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_DIR/logs"
cp "$PLIST_SRC" "$PLIST_DST"

launchctl unload "$PLIST_DST" >/dev/null 2>&1 || true
launchctl load "$PLIST_DST"

echo "Installed and loaded: $PLIST_DST"
launchctl list | grep "ai.openclaw.mlb.update-results" || true
