#!/bin/bash
# Pushes the status page to GitHub every 20 minutes from the LOGIN node.
# Only needed if the jobs cannot reach GitHub (the status page then stops updating on the 'selfLearnt-runs' branch).
#   tmux new -s status  ->  bash jobs/status_loop.sh  ->  Ctrl-b d
cd "$(dirname "$0")/.."
source jobs/cluster.env
module purge 2>/dev/null || true
module load "$PY_MODULE" 2>/dev/null || true
source "$VENV/bin/activate"
while true; do
  python run.py --config "$CONFIG" --stage status > logs/status_loop.log 2>&1 || true
  sleep 1200
done
