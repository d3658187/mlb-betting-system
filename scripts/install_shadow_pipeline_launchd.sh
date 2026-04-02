#!/bin/bash
set -e

PLIST_SRC="scripts/launchd/ai.openclaw.mlb.shadow-pipeline.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/ai.openclaw.mlb.shadow-pipeline.plist"
LABEL="ai.openclaw.mlb.shadow-pipeline"

echo "=== MLB Shadow Pipeline Installer ==="

# Check plist exists
if [ ! -f "$PLIST_SRC" ]; then
    echo "ERROR: $PLIST_SRC not found"
    exit 1
fi

# Copy to LaunchAgents
cp "$PLIST_SRC" "$PLIST_DEST"
echo "✓ Copied plist to $PLIST_DEST"

# Load
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"
echo "✓ Loaded $LABEL"

# Status
STATUS=$(launchctl print "gui/$(id -u)/$LABEL" 2>&1 || echo "error")
if echo "$STATUS" | grep -q "lastExitStatus"; then
    echo "✓ Service running"
else
    echo "⚠ Status: $STATUS"
fi

echo ""
echo "=== Done ==="
echo "Start:  launchctl start $LABEL"
echo "Stop:   launchctl stop $LABEL"
echo "Logs:   log show --predicate 'processImagePath contains \"mlb\"' --last 5m"
