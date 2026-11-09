#!/usr/bin/env bash
# gpab_7b_lab_485.sh -- GPAgentBench WEAK side (Qwen2.5-7B-Instruct) on ALL 485 TEST CASES, lab GPU. DRAFT 2026-11-09 -- run only
# after Jonathan's go. Built from gpab_7b_conditions_lab.sh condition C (7B agent + 7B simulator, strict parser, pure T = 0, bf16,
# util 0.55, max-model-len 16384) and gpab_72b_lab_485.sh (the 72B-AWQ server block, used here for the Dx/Tx JUDGE only).
# Pairing note (registered): the strong side (test485_72b_lab_strict_pure) was simulated by the 72B; the weak side is simulated by
# the 7B because the two models do not fit the 46 GB card together. Judge = 72B-AWQ on both sides. Stated as a confound of the
# routing table, not hidden. Writes test485_7b_lab_strict_pure/ (refuses to overwrite). Needs ~44 GB free for the judge step ->
# start through the lab queue:
#   NEED_GB=44 LOG=/mnt/hdd/jonathan/runs/gpagentbench/gpu_wait_7b_485.log \
#     bash /mnt/hdd/jonathan/STAIR/lab_package/tools/gpu_wait_run.sh bash /mnt/hdd/jonathan/STAIR/lab_package/gpagentbench/gpab_7b_lab_485.sh
set -euo pipefail
STAIR=/mnt/hdd/jonathan/STAIR
PKG=$STAIR/lab_package/gpagentbench
DATA=/mnt/hdd/jonathan/data/gpagentbench
CASES=$STAIR/common/stair_common/splits/gpagentbench/test485.txt
ROOT=/mnt/hdd/jonathan/runs/gpagentbench
OUT=$ROOT/test485_7b_lab_strict_pure
PY=/mnt/hdd/jonathan/vllm-env2/bin/python
WEAK=Qwen/Qwen2.5-7B-Instruct
STRONG=Qwen/Qwen2.5-72B-Instruct-AWQ
export HF_HOME=${HF_HOME:-/mnt/hdd/jonathan/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GPAB_TOLERANT=0                                   # strict parser, as the strong run and condition C
export GPAB_SIM_PARAMS=7                                 # simulator cost column priced as the 7B (diagnostic only)
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists"; exit 1; }
[ -f "$CASES" ] || { echo "missing $CASES"; exit 1; }
[ -d "$HF_HOME/hub/models--Qwen--Qwen2.5-72B-Instruct-AWQ" ] || { echo "72B-AWQ not in $HF_HOME"; exit 1; }
mkdir -p "$OUT"; cd "$OUT"
echo "7B agent + 7B simulator (bf16, pure T=0, util 0.55, max-model-len 16384) | judge 72B-AWQ | GPAB_TOLERANT=0 | all 485 test cases (lab v1.0)" | tee condition.txt
sha256sum $PKG/gpab_env.py $PKG/gpab_run.py $PKG/gpab_score.py $PKG/prompts/patient_simulator_v1_1.txt $CASES | tee inputs_sha256.txt
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
SERVER_PID=""
wait_up () { for i in $(seq 1 120); do curl -s localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done; echo "server on $1 did not start"; tail -n 40 "$2"; exit 1; }
stop_server () { [ -n "$SERVER_PID" ] && { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; sleep 15; }; SERVER_PID=""; }
trap 'stop_server' EXIT
# --- step 1: weak episodes (7B agent, 7B simulator) ---
"$PY" -m vllm.entrypoints.openai.api_server --model $WEAK --port 8011 --gpu-memory-utilization 0.55 \
      --max-model-len 16384 --dtype bfloat16 --generation-config vllm > vllm_7b.log 2>&1 &
SERVER_PID=$!; wait_up 8011 vllm_7b.log; curl -s localhost:8011/v1/models | head -c 200; echo
"$PY" $PKG/gpab_run.py --data $DATA --cases $CASES --out "$OUT" --agent-tag weak --agent-model $WEAK \
      --agent-url http://localhost:8011/v1 --sim-model $WEAK --sim-url http://localhost:8011/v1
stop_server
# --- step 2: judge (72B-AWQ), same server settings as the strong run ---
"$PY" -m vllm.entrypoints.openai.api_server --model $STRONG --port 8012 --gpu-memory-utilization 0.97 --enforce-eager \
      --max-model-len 8192 --generation-config vllm > vllm_72b_judge.log 2>&1 &
SERVER_PID=$!; wait_up 8012 vllm_72b_judge.log; curl -s localhost:8012/v1/models | head -c 200; echo
"$PY" $PKG/gpab_score.py --run "$OUT" --data $DATA --cases $CASES --judge-url http://localhost:8012/v1 --judge-model $STRONG > score_stdout.json
echo "7B LAB RUN (485) DONE -> $OUT"
