#!/usr/bin/env bash
# gpu_wait_run.sh -- lab "queue": wait until the lab GPU has enough free memory, then run one command. DRAFT v1.0 2026-11-04.
# The lab has no scheduler; a job started on a busy GPU fails at model load. This checks free GPU memory every
# INTERVAL seconds (default 300 = 5 min) and starts the command once free >= NEED_GB for two checks in a row (so a
# momentary dip in another user's job does not trigger a start). Every check is logged with a timestamp.
# Usage (inside tmux):  NEED_GB=30 bash gpu_wait_run.sh <command and its arguments ...>
#   e.g.  NEED_GB=30 bash /mnt/hdd/jonathan/STAIR/lab_package/tools/gpu_wait_run.sh \
#             bash /mnt/hdd/jonathan/STAIR/lab_package/gpagentbench/gpab_7b_conditions_lab.sh
# Options (environment): NEED_GB (required), INTERVAL (seconds, default 300), GPU (index, default 0),
#                        LOG (default ./gpu_wait_<timestamp>.log)
set -euo pipefail
: "${NEED_GB:?set NEED_GB, e.g. NEED_GB=30}"
INTERVAL=${INTERVAL:-300}; GPU=${GPU:-0}; LOG=${LOG:-./gpu_wait_$(date +%Y%m%d_%H%M%S).log}
[ $# -ge 1 ] || { echo "usage: NEED_GB=30 bash gpu_wait_run.sh <command ...>"; exit 1; }
echo "$(date -Iseconds) waiting for >= ${NEED_GB} GB free on GPU $GPU (check every ${INTERVAL}s) to run: $*" | tee -a "$LOG"
ok=0
while true; do
  free_mb=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  free_gb=$(( free_mb / 1024 ))
  if [ "$free_gb" -ge "$NEED_GB" ]; then ok=$((ok + 1)); else ok=0; fi
  echo "$(date -Iseconds) free ${free_gb} GB (need ${NEED_GB}) consecutive_ok=${ok}" | tee -a "$LOG"
  [ "$ok" -ge 2 ] && break
  sleep "$INTERVAL"
done
echo "$(date -Iseconds) STARTING: $*" | tee -a "$LOG"
"$@" 2>&1 | tee -a "$LOG"
echo "$(date -Iseconds) FINISHED (exit ${PIPESTATUS[0]})" | tee -a "$LOG"
