#!/bin/bash
# One-look status of the run:   bash ~/STAIR/selfLearnt/pipeline/jobs/status.sh
cd "$(dirname "$0")/.."
source jobs/cluster.env
[ -f jobs/cluster.local.env ] && source jobs/cluster.local.env
VENV_PY="${VENV:-$HOME/stair-env}/bin/python"

echo "================ jobs now (R = running, PD = waiting) ================"
squeue --me -o "%.9i %.30j %.2t %.11M %R" 2>/dev/null

echo
echo "================ finished in the last 2 days (FAILED would show here) ================"
sacct --starttime "$(date -d '2 days ago' +%Y-%m-%dT%H:%M)" -X -n -P --format=JobName,State,Elapsed 2>/dev/null \
  | grep -vE "RUNNING|PENDING|CANCELLED|stair-submitter" | sort | uniq | awk -F'|' '{printf "  %-32s %-10s %s\n", $1, $2, $3}' | tail -40

echo
echo "================ submitter (latest lines) ================"
f=$(ls -t logs/stair-submitter-*.out 2>/dev/null | head -1)
[ -n "$f" ] && { echo "  $f"; grep -E "submitted|set aside|failed:|withdrew|all .* jobs submitted|still to submit|watching" "$f" | tail -6 | sed 's/^/  /'; }

echo
echo "================ results so far (accuracy = lenient, cost = units per question) ================"
"$VENV_PY" - <<'PY' 2>/dev/null
import glob, os, pandas as pd
d = os.environ.get("STAIR_RESULTS_DIR") or os.path.expanduser("~/STAIR/results/common")
fs = sorted(glob.glob(os.path.join(d, "AL-*-seed*.csv")))
if not fs:
    print("  none yet")
for f in fs:
    d = pd.read_csv(f)
    t = d.groupby("policy")[["lenient", "units"]].mean().sort_values("units")
    print(f"  {os.path.basename(f)}")
    for p, r in t.iterrows():
        print(f"    {p:28s} {r.lenient:.3f}   {r.units:8.0f}")
PY

echo
echo "================ disk ================"
du -sh "$HOME" 2>/dev/null | awk '{print "  home folder: "$1" of 150G"}'
