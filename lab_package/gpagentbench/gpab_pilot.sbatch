#!/bin/bash
# gpab_pilot.sbatch -- GPAgentBench-2K pilot (60 cases x 2 agents + judge) on the SMU cluster, one H200.
# DRAFT v1.0 2026-10-31 for Jonathan's review. Header from his prod_*.sbatch. NOT to be submitted before approval.
# Needs: a Python env with vLLM on the cluster  -> set VLLM_PY below  [OPEN: which env? slr-env has torch 2.11, vLLM untested]
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
DATA=/common/scratch/users/d/dohyung.ahn.2025/data/gpagentbench          # [OPEN] copy of the lab's gated data dir
CASES=$STAIR/common/stair_common/splits/gpagentbench/pilot60.txt
OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/gpagentbench/pilot60
VLLM_PY=$HOME/slr/vllm-env/bin/python                                    # [OPEN] env with vLLM
WEAK=Qwen/Qwen2.5-7B-Instruct
STRONG=Qwen/Qwen2.5-72B-Instruct-AWQ
mkdir -p "$OUT"; cd "$OUT"
[ -f "$CASES" ] || { echo "pilot60.txt missing -- run gpab_pilot_split.py first"; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. two vLLM servers on one H200: 7B (agent-weak + simulator) on 8001, 72B-AWQ (agent-strong + judge) on 8002
$VLLM_PY -m vllm.entrypoints.openai.api_server --model $WEAK   --port 8001 --gpu-memory-utilization 0.20 --max-model-len 16384 --dtype bfloat16 > vllm_7b.log 2>&1 &
$VLLM_PY -m vllm.entrypoints.openai.api_server --model $STRONG --port 8002 --gpu-memory-utilization 0.70 --max-model-len 16384 --quantization awq_marlin > vllm_72b.log 2>&1 &
for port in 8001 8002; do
  for i in $(seq 1 120); do curl -s localhost:$port/v1/models >/dev/null 2>&1 && break; sleep 10; done
  curl -s localhost:$port/v1/models | head -c 300; echo
done

# 2. episodes: weak agent (7B) then strong agent (72B); simulator = 7B on 8001 in both
python3 $PKG/gpab_run.py --data $DATA --cases $CASES --out $OUT --agent-tag weak   --agent-model $WEAK   --agent-url http://localhost:8001/v1 --sim-model $WEAK --sim-url http://localhost:8001/v1
python3 $PKG/gpab_run.py --data $DATA --cases $CASES --out $OUT --agent-tag strong --agent-model $STRONG --agent-url http://localhost:8002/v1 --sim-model $WEAK --sim-url http://localhost:8001/v1

# 3. churn: the first 10 pilot cases again, per agent, to separate files
head -10 $CASES > $OUT/repeat10.txt
python3 $PKG/gpab_run.py --data $DATA --cases $OUT/repeat10.txt --out $OUT --agent-tag weak   --agent-model $WEAK   --agent-url http://localhost:8001/v1 --sim-model $WEAK --sim-url http://localhost:8001/v1 --repeat-tag rep2
python3 $PKG/gpab_run.py --data $DATA --cases $OUT/repeat10.txt --out $OUT --agent-tag strong --agent-model $STRONG --agent-url http://localhost:8002/v1 --sim-model $WEAK --sim-url http://localhost:8001/v1 --repeat-tag rep2

# 4. score: deterministic metrics + FLOPs + Dx/Tx judge (72B, C.2 prompts)
python3 $PKG/gpab_score.py --run $OUT --data $DATA --cases $CASES --judge-url http://localhost:8002/v1 --judge-model $STRONG
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "PILOT DONE -> $OUT"
