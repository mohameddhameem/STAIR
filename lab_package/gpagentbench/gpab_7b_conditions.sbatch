#!/bin/bash
# gpab_7b_conditions.sbatch -- 7B-only rerun of the GPAgentBench pilot under one of three conditions, to separate the
# HARNESS effect (parser) from the DECODING effect (model file's repetition penalty 1.05) behind the 7B's manage-case gap
# (46.7 % vs the paper's 78.9 %; pilot audit: 14/15 invalid ends = `[action type="treat">`). DRAFT v1.0 2026-11-01 --
# submit only after Jonathan's go.  Usage:  sbatch --export=ALL,COND=B ~/slr/STAIR/lab_package/gpagentbench/gpab_7b_conditions.sbatch
#   COND=B  tolerant parser ('[' or '<')   + decoding as the pilot (temperature 0 requests, model-file defaults -> rep. penalty 1.05)
#   COND=C  strict parser (as the pilot)   + pure temperature 0, NO model-file defaults (--generation-config vllm)
#   COND=D  tolerant parser                + pure temperature 0
# Everything else = the pilot (job 347293): same 60 cases, same C.1 prompt, simulator = the same 7B server with prompt v1.1,
# D1 = two re-prompts then invalid_action, 10-turn cap. 7B only (the 72B matched the paper); no judge (Mgmt / Miss Refer /
# Miss GP / turns are deterministic). One H200, one server.
#SBATCH --job-name=gpab_7b_cond
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=gpab_7b_cond.%j.log
set -euo pipefail
case "${COND:-}" in
  B) export GPAB_TOLERANT=1; GEN_CFG="" ;;
  C) export GPAB_TOLERANT=0; GEN_CFG="--generation-config vllm" ;;
  D) export GPAB_TOLERANT=1; GEN_CFG="--generation-config vllm" ;;
  *) echo "set COND=B, C or D"; exit 1 ;;
esac
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
STAIR=$HOME/slr/STAIR
PKG=$STAIR/lab_package/gpagentbench
DATA=/common/scratch/users/d/dohyung.ahn.2025/data/gpagentbench
CASES=$STAIR/common/stair_common/splits/gpagentbench/pilot60.txt
OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/gpagentbench/pilot60_7b_cond$COND
VLLM_PY=$HOME/slr/vllm-env/bin/python
WEAK=Qwen/Qwen2.5-7B-Instruct
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists"; exit 1; }
mkdir -p "$OUT"; cd "$OUT"
echo "COND=$COND GPAB_TOLERANT=$GPAB_TOLERANT GEN_CFG='$GEN_CFG'" | tee condition.txt
sha256sum $PKG/gpab_env.py $PKG/gpab_run.py $PKG/gpab_score.py $CASES | tee inputs_sha256.txt
wait_up () { for i in $(seq 1 120); do curl -s localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done; echo "server on $1 did not start"; tail -n 40 $2; exit 1; }
$VLLM_PY -m vllm.entrypoints.openai.api_server --model $WEAK --port 8001 --gpu-memory-utilization 0.85 --max-model-len 16384 --dtype bfloat16 $GEN_CFG > vllm_7b.log 2>&1 &
wait_up 8001 vllm_7b.log
$VLLM_PY $PKG/gpab_run.py --data $DATA --cases $CASES --out $OUT --agent-tag weak --agent-model $WEAK --agent-url http://localhost:8001/v1 --sim-model $WEAK --sim-url http://localhost:8001/v1
$VLLM_PY $PKG/gpab_score.py --run $OUT --data $DATA --cases $CASES
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "7B CONDITION $COND DONE -> $OUT"
