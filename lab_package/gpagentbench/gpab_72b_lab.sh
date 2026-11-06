#!/usr/bin/env bash
# gpab_72b_lab.sh -- GPAgentBench 72B on the LAB GPU with the clean setting from the 7B conditions (strict parser, pure
# temperature 0 = --generation-config vllm). DRAFT v1.0 2026-11-04 -- run only after Jonathan's go.
# ONE server (Qwen2.5-72B-Instruct-AWQ, as in the pilot) serves the doctor-agent, the patient SIMULATOR and the Dx/Tx judge
# -- two models do not fit the 46 GB card together. So vs the pilot (job 347293) TWO things change for the 72B: decoding
# (pure T = 0) and the simulator (72B-AWQ instead of the 7B). Precision stays AWQ (bf16 is the cluster alignment run).
# Same 60 cases, C.1 prompt, simulator prompt v1.1, D1, 10-turn cap. Writes pilot60_72b_lab_strict_pure/ (refuses to overwrite).
# Needs ~44 GB free -> start it through the lab queue so it waits for an (almost) empty GPU:
#   NEED_GB=44 LOG=/mnt/hdd/jonathan/runs/gpagentbench/gpu_wait_72b.log \
#     bash /mnt/hdd/jonathan/STAIR/lab_package/tools/gpu_wait_run.sh bash /mnt/hdd/jonathan/STAIR/lab_package/gpagentbench/gpab_72b_lab.sh
set -euo pipefail
STAIR=/mnt/hdd/jonathan/STAIR
PKG=$STAIR/lab_package/gpagentbench
DATA=/mnt/hdd/jonathan/data/gpagentbench
CASES=$STAIR/common/stair_common/splits/gpagentbench/pilot60.txt
ROOT=/mnt/hdd/jonathan/runs/gpagentbench
OUT=$ROOT/pilot60_72b_lab_strict_pure
PY=/mnt/hdd/jonathan/vllm-env2/bin/python
STRONG=Qwen/Qwen2.5-72B-Instruct-AWQ
export HF_HOME=${HF_HOME:-/mnt/hdd/jonathan/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export GPAB_TOLERANT=0                                   # strict parser (as the pilot; D = C showed tolerance is not needed)
export GPAB_SIM_PARAMS=72                                # simulator cost column priced as the 72B (diagnostic only)
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists"; exit 1; }
[ -d "$HF_HOME/hub/models--Qwen--Qwen2.5-72B-Instruct-AWQ" ] || { echo "72B-AWQ not in $HF_HOME"; exit 1; }
mkdir -p "$OUT"; cd "$OUT"
echo "72B-AWQ agent + 72B-AWQ simulator + judge | GPAB_TOLERANT=0 | --generation-config vllm (pure T=0) | max-model-len 8192, util 0.97, enforce-eager (lab v1.1)" | tee condition.txt
sha256sum $PKG/gpab_env.py $PKG/gpab_run.py $PKG/gpab_score.py $PKG/prompts/patient_simulator_v1_1.txt $CASES | tee inputs_sha256.txt
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
"$PY" -m vllm.entrypoints.openai.api_server --model $STRONG --port 8012 --gpu-memory-utilization 0.97 --enforce-eager \
      --max-model-len 8192 --generation-config vllm > vllm_72b.log 2>&1 &
# v1.1 (6 Nov): 0.97 + --enforce-eager = the settings that served this model on this card for the MedAgentBench 72B rerun
# (rerun72b_pureT0_lab.sh). At 0.95 without eager mode the 4 Nov attempt failed at start-up: 0.86 GiB left for KV cache, 2.5 GiB
# needed for 8,192 tokens (vllm_72b.log in pilot60_72b_lab_strict_pure_ABORTED_4Nov). Eager mode costs speed, not outputs.
SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do curl -s localhost:8012/v1/models >/dev/null 2>&1 && break; sleep 10; done
curl -s localhost:8012/v1/models | head -c 200; echo
"$PY" $PKG/gpab_run.py --data $DATA --cases $CASES --out "$OUT" --agent-tag strong --agent-model $STRONG \
      --agent-url http://localhost:8012/v1 --sim-model $STRONG --sim-url http://localhost:8012/v1
"$PY" $PKG/gpab_score.py --run "$OUT" --data $DATA --cases $CASES --judge-url http://localhost:8012/v1 --judge-model $STRONG > score_stdout.json
echo "72B LAB RUN DONE -> $OUT"
