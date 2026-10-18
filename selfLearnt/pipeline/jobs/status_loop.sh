#!/bin/bash
# Pushes the status page to GitHub every 20 minutes from the LOGIN node.
# Only needed if the jobs cannot reach GitHub (the status page then stops updating on the 'selfLearnt-runs' branch).
#   tmux new -s status  ->  bash jobs/status_loop.sh  ->  Ctrl-b d
cd "$(dirname "$0")/.."
source jobs/cluster.env
[ -f jobs/cluster.local.env ] && source jobs/cluster.local.env   # your own settings, never committed
conda deactivate 2>/dev/null || true   # the login shell starts in Anaconda base: use the module Python instead
module purge 2>/dev/null || true
module load "$PY_MODULE" 2>/dev/null || true
module load "${CUDA_MODULE:-CUDA/12.4.0}" 2>/dev/null || true
source "$VENV/bin/activate"
while true; do
  python run.py --config "$CONFIG" --stage status > logs/status_loop.log 2>&1 || true
  sleep 1200
done
