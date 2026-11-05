#!/bin/bash
# gpab_align.sbatch -- GPAgentBench alignment run (enhancement 2, agreed with Jonathan 1 Nov 21:20): BOTH agents on the
# same 60 pilot cases with (a) the patient SIMULATOR = Qwen2.5-72B-Instruct bf16 (closest local equivalent of the paper's
# GPT-4o-mini; our prompt v1.1, theirs unpublished) and (b) the strong AGENT = Qwen2.5-72B-Instruct bf16 (the paper's
# hardware, 4 x A6000 48 GB, fits bf16; precision not stated -- recorded as an inference). Parser / decoding carried from
# the 7B three-condition result (jobs 348360-62) via PARSER and GEN. DRAFT v1.0 2026-11-01 -- submit only after Jonathan's go.
#   PARSER=strict|tolerant   GEN=model|pure   (model = vLLM applies generation_config.json, i.e. rep. penalty 1.05;
#                                              pure = --generation-config vllm, plain temperature 0)
# Usage:  sbatch --export=ALL,PARSER=tolerant,GEN=pure ~/slr/STAIR/lab_package/gpagentbench/gpab_align.sbatch
# v1.1 (5 Nov, simulator ablation): SIMSIZE=7b|32b|72b picks the PATIENT SIMULATOR only (default 72b = the 4 Nov run);
#   everything else identical, so a run differs from the 4 Nov alignment run in exactly one factor. 7b/32b simulators are served
#   from a third vLLM server on GPU 0 beside the 7B agent (bf16; 32B at 0.40 of an H200). Output folder carries the simulator size.
# Servers (2 x H200, started IN SEQUENCE -- lesson of job 346578): 72B bf16 tensor-parallel 2 on port 8002 (agent for the
# strong runs, simulator for both, Dx/Tx judge), then the 7B on port 8001 (weak agent). Judge prompts C.2 unchanged.
#SBATCH --job-name=gpab_align
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=gpab_align.%j.log
set -euo pipefail
case "${PARSER:-}" in strict) export GPAB_TOLERANT=0 ;; tolerant) export GPAB_TOLERANT=1 ;; *) echo "set PARSER=strict|tolerant"; exit 1 ;; esac
case "${GEN:-}" in model) GEN_CFG="" ;; pure) GEN_CFG="--generation-config vllm" ;; *) echo "set GEN=model|pure"; exit 1 ;; esac
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SIMSIZE=${SIMSIZE:-72b}
case "$SIMSIZE" in 7b) SIMMODEL=Qwen/Qwen2.5-7B-Instruct; export GPAB_SIM_PARAMS=7 ;; 32b) SIMMODEL=Qwen/Qwen2.5-32B-Instruct; export GPAB_SIM_PARAMS=32 ;; 72b) SIMMODEL=Qwen/Qwen2.5-72B-Instruct; export GPAB_SIM_PARAMS=72 ;; *) echo "SIMSIZE must be 7b|32b|72b"; exit 1 ;; esac
STAIR=$HOME/slr/STAIR
PKG=$STAIR/lab_package/gpagentbench
DATA=/common/scratch/users/d/dohyung.ahn.2025/data/gpagentbench
CASES=$STAIR/common/stair_common/splits/gpagentbench/pilot60.txt
OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/gpagentbench/pilot60_align_${PARSER}_${GEN}$([ "$SIMSIZE" = 72b ] && echo "" || echo "_sim${SIMSIZE}")
VLLM_PY=$HOME/slr/vllm-env/bin/python
WEAK=Qwen/Qwen2.5-7B-Instruct
STRONG=Qwen/Qwen2.5-72B-Instruct
[ -e "$OUT" ] && { echo "REFUSING: $OUT exists"; exit 1; }
mkdir -p "$OUT"; cd "$OUT"
echo "PARSER=$PARSER GPAB_TOLERANT=$GPAB_TOLERANT GEN=$GEN GEN_CFG='$GEN_CFG' SIM=$SIMMODEL(bf16, SIMSIZE=$SIMSIZE) STRONG_AGENT=$STRONG(bf16)" | tee condition.txt
sha256sum $PKG/gpab_env.py $PKG/gpab_run.py $PKG/gpab_score.py $PKG/prompts/patient_simulator_v1_1.txt $CASES | tee inputs_sha256.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
wait_up () { for i in $(seq 1 180); do curl -s localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done; echo "server on $1 did not start"; tail -n 40 $2; exit 1; }
$VLLM_PY -m vllm.entrypoints.openai.api_server --model $STRONG --port 8002 --tensor-parallel-size 2 --gpu-memory-utilization 0.78 --max-model-len 16384 --dtype bfloat16 $GEN_CFG > vllm_72b.log 2>&1 &
wait_up 8002 vllm_72b.log
CUDA_VISIBLE_DEVICES=0 $VLLM_PY -m vllm.entrypoints.openai.api_server --model $WEAK --port 8001 --gpu-memory-utilization 0.16 --max-model-len 16384 --dtype bfloat16 $GEN_CFG > vllm_7b.log 2>&1 &
wait_up 8001 vllm_7b.log
SIM_URL=http://localhost:8002/v1
if [ "$SIMSIZE" = 7b ]; then SIM_URL=http://localhost:8001/v1; fi      # the 7B agent server doubles as the 7B simulator (as in the pilot)
if [ "$SIMSIZE" = 32b ]; then
  CUDA_VISIBLE_DEVICES=0 $VLLM_PY -m vllm.entrypoints.openai.api_server --model $SIMMODEL --port 8003 --gpu-memory-utilization 0.40 --max-model-len 16384 --dtype bfloat16 $GEN_CFG > vllm_sim32b.log 2>&1 &
  wait_up 8003 vllm_sim32b.log; SIM_URL=http://localhost:8003/v1
fi
for port in 8002 8001; do curl -s localhost:$port/v1/models | head -c 200; echo; done

# episodes: simulator = the 72B (port 8002) for BOTH agents
$VLLM_PY $PKG/gpab_run.py --data $DATA --cases $CASES --out $OUT --agent-tag weak   --agent-model $WEAK   --agent-url http://localhost:8001/v1 --sim-model $SIMMODEL --sim-url $SIM_URL
$VLLM_PY $PKG/gpab_run.py --data $DATA --cases $CASES --out $OUT --agent-tag strong --agent-model $STRONG --agent-url http://localhost:8002/v1 --sim-model $SIMMODEL --sim-url $SIM_URL
# score: deterministic metrics + FLOPs + Dx/Tx judge (72B bf16, C.2 prompts)
$VLLM_PY $PKG/gpab_score.py --run $OUT --data $DATA --cases $CASES --judge-url http://localhost:8002/v1 --judge-model $STRONG
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "ALIGN RUN DONE -> $OUT"
