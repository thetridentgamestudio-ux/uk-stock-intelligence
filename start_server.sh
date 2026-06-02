#!/bin/bash
# Full Python path — avoids LaunchAgent PATH issues where macOS picks Xcode Python
PYTHON=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
PROJECT=/Users/viveksharma/Claude/uk-stock-intelligence

cd "$PROJECT"

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server starting..." >> /tmp/uk-stock-server.log
  "$PYTHON" -m uvicorn backend.app.main:app \
    --host 0.0.0.0 --port 8000 --log-level info \
    >> /tmp/uk-stock-server.log 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server stopped — restarting in 10s..." >> /tmp/uk-stock-server.log
  sleep 10
done
