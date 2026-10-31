#!/bin/bash
# gpab_pilot.sbatch -- GPAgentBench-2K pilot (60 cases x 2 agents + judge) on the SMU cluster, one H200.
# v1.2 2026-11-01 (v1.1 reviewed by Jonathan 31 Oct 22:32; v1.2: servers started in sequence after job 346578's start-up failure; GEN_CFG = model-file defaults, decided 1 Nov 07:30). Header from his prod_*.sbatch. NOT to be submitted before approval.
# Needs: ~/slr/vllm-env (cluster_vllm_env.sh, vLLM 0.19.0) -- no pre-existing cluster env had vLLM (checked 31 Oct)
#        the 7B, 72B-AWQ in $HF_HOME (both present 31 Oct), STAIR pulled at ~/slr/STAIR, data at $DATA.
#SBATCH --job-name=gpab_pilot
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=gpab_pilot.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
STAIR=$HOME/slr/STAIR
PKG=$STAIR/lab_package/gpagentbench
DATA=/common/scratch/users/d/dohyung.ahn.2025/data/gpagentbench          # copied from the lab 31 Oct (jsonl under data/)
CASES=$STAIR/common/stair_common/splits/gpagentbench/pilot60.txt
OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/gpagentbench/pilot60
VLLM_PY=$HOME/slr/vllm-env/bin/python                                    # built by cluster_vllm_env.sh (vLLM 0.19.0)
WEAK=Qwen/Qwen2.5-7B-Instruct
STRONG=Qwen/Qwen2.5-72B-Instruct-AWQ
mkdir -p "$OUT"; cd "$OUT"
[ -f "$CASES" ] || { echo "pilot60.txt missing -- run gpab_pilot_split.py first"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. two vLLM servers on one H200, started IN SEQUENCE (v1.2, 1 Nov): job 346578 failed because the 7B profiled its
#    memory while the 72B was loading in parallel ("Available KV cache memory: -29.57 GiB"). 72B first (0.70), wait
#    until it answers, then the 7B (0.20). GEN_CFG = decoding source: "" = vLLM applies the model's generation_config.json
#    (Qwen: repetition_penalty 1.05; temperature is overridden by our temperature-0 requests); "--generation-config vllm"
#    = pure temperature-0 decoding, nothing from the model file.  DECIDED 1 Nov 07:30 (Jonathan, revising 07:29): MODEL-FILE
#    DEFAULTS ("") -- same serving as the lab's MedAgentBench vLLM (no flag there); requests still send temperature 0.
GEN_CFG=""
wait_up () { for i in $(seq 1 120); do curl -s localhost:$1/v1/models >/dev/null 2>&1 && return 0; sleep 10; done; echo "server on $1 did not start"; tail -n 40 $2; exit 1; }
$VLLM_PY -m vllm.entrypoints.openai.api_server --model $STRONG --port 8002 --gpu-memory-utilization 0.70 --max-model-len 16384 $GEN_CFG > vllm_72b.log 2>&1 &
wait_up 8002 vllm_72b.log
$VLLM_PY -m vllm.entrypoints.openai.api_server --model $WEAK   --port 8001 --gpu-memory-utilization 0.20 --max-model-len 16384 --dtype bfloat16 $GEN_CFG > vllm_7b.log 2>&1 &
wait_up 8001 vllm_7b.log
for port in 8002 8001; do curl -s localhost:$port/v1/models | head -c 300; echo; done

# 2. episodes: weak agent (7B) then strong agent (72B); simulator = 7B on 8001 in both
$VLLM_PY $PKG/gpab_run.py --data $DATA --cases $CASES --out $OUT --agent-tag weak   --agent-model $WEAK   --agent-url http://localhost:8001/v1 --sim-model $WEAK --sim-url http://localhost:8001/v1
$VLLM_PY $PKG/gpab_run.py --data $DATA --cases $CASES --out $OUT --agent-tag strong --agent-model $STRONG --agent-url http://localhost:8002/v1 --sim-model $WEAK --sim-url http://localhost:8001/v1

# 3. churn: the first 10 pilot cases again, per agent, to separate files
head -10 $CASES > $OUT/repeat10.txt
$VLLM_PY $PKG/gpab_run.py --data $DATA --cases $OUT/repeat10.txt --out $OUT --agent-tag weak   --agent-model $WEAK   --agent-url http://localhost:8001/v1 --sim-model $WEAK --sim-url http://localhost:8001/v1 --repeat-tag rep2
$VLLM_PY $PKG/gpab_run.py --data $DATA --cases $OUT/repeat10.txt --out $OUT --agent-tag strong --agent-model $STRONG --agent-url http://localhost:8002/v1 --sim-model $WEAK --sim-url http://localhost:8001/v1 --repeat-tag rep2

# 4. score: deterministic metrics + FLOPs + Dx/Tx judge (72B, C.2 prompts)
$VLLM_PY $PKG/gpab_score.py --run $OUT --data $DATA --cases $CASES --judge-url http://localhost:8002/v1 --judge-model $STRONG
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "PILOT DONE -> $OUT"
