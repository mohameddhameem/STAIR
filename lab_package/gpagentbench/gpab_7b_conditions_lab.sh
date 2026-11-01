#!/usr/bin/env bash
# gpab_7b_conditions_lab.sh -- the 7B three-condition rerun (B / C / D) on the LAB GPU instead of the cluster (cluster queue
# estimate 3 Nov). DRAFT v1.0 2026-11-01 -- run only after Jonathan's go. Same steps as gpab_7b_conditions.sbatch (job
# file 75bdf3b8, reviewed), with the lab's paths and one vLLM server per decoding setting:
#   server 1: model-file defaults (rep. penalty 1.05)  -> COND B  (tolerant parser)
#   server 2: --generation-config vllm (pure T = 0)    -> COND C  (strict parser), then COND D (tolerant parser)
# Same 60 cases, C.1 prompt, simulator = the same 7B server with prompt v1.1, D1 = two re-prompts then invalid_action,
# 10-turn cap; 7B only; no judge (Mgmt / Miss Refer / Miss GP / turns are deterministic). Each condition writes its own
# folder and refuses to overwrite. GPU: one RTX 6000 Ada (46 GB, ~41 GB free on 1 Nov 21:51); the server takes 0.55.
# Usage (lab, inside tmux):  bash /mnt/hdd/jonathan/STAIR/lab_package/gpagentbench/gpab_7b_conditions_lab.sh
set -euo pipefail
STAIR=/mnt/hdd/jonathan/STAIR
PKG=$STAIR/lab_package/gpagentbench
DATA=/mnt/hdd/jonathan/data/gpagentbench
CASES=$STAIR/common/stair_common/splits/gpagentbench/pilot60.txt
ROOT=/mnt/hdd/jonathan/runs/gpagentbench
PY=/mnt/hdd/jonathan/vllm-env2/bin/python
WEAK=Qwen/Qwen2.5-7B-Instruct
export HF_HOME=${HF_HOME:-/mnt/hdd/jonathan/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for C in B C D; do [ -e "$ROOT/pilot60_7b_cond$C" ] && { echo "REFUSING: $ROOT/pilot60_7b_cond$C exists"; exit 1; }; done
[ -f "$CASES" ] || { echo "missing $CASES"; exit 1; }
"$PY" -c "import vllm; print('vllm', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

wait_up () { for i in $(seq 1 120); do curl -s localhost:8011/v1/models >/dev/null 2>&1 && return 0; sleep 10; done; echo "server did not start"; tail -n 40 "$1"; exit 1; }
start_server () {   # $1 = GEN_CFG ("" or "--generation-config vllm"), $2 = log file
  "$PY" -m vllm.entrypoints.openai.api_server --model $WEAK --port 8011 --gpu-memory-utilization 0.55 \
        --max-model-len 16384 --dtype bfloat16 $1 > "$2" 2>&1 &
  SERVER_PID=$!; wait_up "$2"; curl -s localhost:8011/v1/models | head -c 200; echo
}
stop_server () { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; sleep 10; }
run_cond () {      # $1 = COND, $2 = GPAB_TOLERANT, $3 = GEN_CFG text (recorded)
  OUT=$ROOT/pilot60_7b_cond$1; mkdir -p "$OUT"
  echo "COND=$1 GPAB_TOLERANT=$2 GEN_CFG='$3' (lab)" | tee "$OUT/condition.txt"
  sha256sum $PKG/gpab_env.py $PKG/gpab_run.py $PKG/gpab_score.py $CASES | tee "$OUT/inputs_sha256.txt"
  GPAB_TOLERANT=$2 "$PY" $PKG/gpab_run.py --data $DATA --cases $CASES --out "$OUT" --agent-tag weak --agent-model $WEAK \
      --agent-url http://localhost:8011/v1 --sim-model $WEAK --sim-url http://localhost:8011/v1
  GPAB_TOLERANT=$2 "$PY" $PKG/gpab_score.py --run "$OUT" --data $DATA --cases $CASES > "$OUT/score_stdout.json"
  echo "== COND $1 done -> $OUT"
}
trap 'stop_server' EXIT
start_server "" "$ROOT/vllm_7b_modeldefaults.log";        run_cond B 1 ""
stop_server
start_server "--generation-config vllm" "$ROOT/vllm_7b_pure.log"; run_cond C 0 "--generation-config vllm"; run_cond D 1 "--generation-config vllm"
echo "ALL THREE CONDITIONS DONE"
