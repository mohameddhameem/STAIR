#!/bin/bash
# gpab_align.sbatch -- GPAgentBench alignment run (enhancement 2, agreed with Jonathan 1 Nov 21:20): BOTH agents on the
# same 60 pilot cases with (a) the patient SIMULATOR = Qwen2.5-72B-Instruct bf16 (closest local equivalent of the paper's
# GPT-4o-mini; our prompt v1.1, theirs unpublished) and (b) the strong AGENT = Qwen2.5-72B-Instruct bf16 (the paper's
# hardware, 4 x A6000 48 GB, fits bf16; precision not stated -- recorded as an inference). Parser / decoding carried from
# the 7B three-condition result (jobs 348360-62) via PARSER and GEN. DRAFT v1.0 2026-11-01 -- submit only after Jonathan's go.
#   PARSER=strict|tolerant   GEN=model|pure   (model = vLLM applies generation_config.json, i.e. rep. penalty 1.05;
#                                              pure = --generation-config vllm, plain temperature 0)
# Usage:  sbatch --export=ALL,PARSER=tolerant,GEN=pure ~/slr/STAIR/lab_package/gpagentbench/gpab_align.sbatch
# v1.2 (6 Nov, decoding check for the 72B Miss-GP cell): DECODE=pure|model sets the AGENT decoding via gpab_run.py v1.1
#   (pure = temperature 0 on every agent request, all runs to 5 Nov; model = no temperature sent, so the server's
#   generation_config.json applies: Qwen2.5 T 0.7 / top-p 0.8 / top-k 20 / rep. penalty 1.05). DECODE=model REQUIRES GEN=model
#   (a server started with --generation-config vllm has no defaults to apply). Simulator and judge stay at T 0. Output folder
#   carries _agentsample. Decided by Jonathan 6 Nov 17:02: one run, 72B simulator, strict parser -- a reproduction check of the
#   paper's likely decoding, NOT a change to our own (pure T 0 stays the standard).
# v1.1 (5 Nov, simulator ablation): SIMSIZE=7b|32b|72b picks the PATIENT SIMULATOR only (default 72b = the 4 Nov run);
#   everything else identical, so a run differs from the 4 Nov alignment run in exactly one factor. 7b/32b simulators are served
#   from a third vLLM server on GPU 0 beside the 7B agent (bf16; 32B at 0.40 of an H200). Output folder carries the simulator size.
# Servers (2 x H200, started IN SEQUENCE -- lesson of job 346578): 72B bf16 tensor-parallel 2 on port 8002 (agent for the
# strong runs, simulator for both, Dx/Tx judge), then the 7B on port 8001 (weak agent). Judge prompts C.2 unchanged.
# v1.3 (9 Nov, the 485-case run): CASESET=pilot60|test485 picks the case list and the output-folder prefix (default pilot60 =
#   every run to date). test485 = the full test split (sha d868fabb…); both agents + judge ≈ 8× the pilot's 30 min → --time 08:00.
#   Usage: sbatch --export=ALL,PARSER=strict,GEN=model,DECODE=model,CASESET=test485 ~/slr/STAIR/lab_package/gpagentbench/gpab_align.sbatch
#SBATCH --job-name=gpab_align
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=gpab_align.%j.log
set -euo pipefail
case "${PARSER:-}" in strict) export GPAB_TOLERANT=0 ;; tolerant) export GPAB_TOLERANT=1 ;; *) echo "set PARSER=strict|tolerant"; exit 1 ;; esac
case "${GEN:-}" in model) GEN_CFG="" ;; pure) GEN_CFG="--generation-config vllm" ;; *) echo "set GEN=model|pure"; exit 1 ;; esac
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SIMSIZE=${SIMSIZE:-72b}
DECODE=${DECODE:-pure}
CASESET=${CASESET:-pilot60}
case "$CASESET" in pilot60|test485) ;; *) echo "CASESET must be pilot60|test485"; exit 1 ;; esac
case "$DECODE" in pure) ;; model) [ "$GEN" = model ] || { echo "DECODE=model requires GEN=model"; exit 1; } ;; *) echo "DECODE must be pure|model"; exit 1 ;; esac
case "$SIMSIZE" in 7b) SIMMODEL=Qwen/Qwen2.5-7B-Instruct; export GPAB_SIM_PARAMS=7 ;; 32b) SIMMODEL=Qwen/Qwen2.5-32B-Instruct; export GPAB_SIM_PARAMS=32 ;; 72b) SIMMODEL=Qwen/Qwen2.5-72B-Instruct; export GPAB_SIM_PARAMS=72 ;; *) echo "SIMSIZE must be 7b|32b|72b"; exit 1 ;; esac
STAIR=$HOME/slr/STAIR
PKG=$STAIR/lab_package/gpagentbench
DATA=/common/scratch/users/d/dohyung.ahn.2025/data/gpagentbench
CASES=$STAIR/common/stair_common/splits/gpagentbench/${CASESET}.txt
OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/gpagentbench/${CASESET}_align_${PARSER}_${GEN}$([ "$SIMSIZE" = 72b ] && echo "" || echo "_sim${SIMSIZE}")$([ "$DECODE" = pure ] && echo "" || echo "_agentsample")
VLLM_PY=$HOME/slr/vllm-env/bin/python
WEAK=Qwen/Qwen2.5-7B-Instruct
STRONG=Qwen/Qwen2.5-72B-Instruct
# v1.4 (9 Nov): RESUME=1 re-enters an existing $OUT to finish a run cut off by the time limit -- gpab_run.py's replay cache skips
#   every (case, agent) already in episodes_<tag>.jsonl, so only the missing episodes and the judge run. Input hashes must match.
RESUME=${RESUME:-0}
if [ -e "$OUT" ] && [ "$RESUME" != 1 ]; then echo "REFUSING: $OUT exists (RESUME=1 to finish a cut-off run)"; exit 1; fi
if [ "$RESUME" = 1 ]; then
  [ -f "$OUT/inputs_sha256.txt" ] || { echo "RESUME: no inputs_sha256.txt in $OUT"; exit 1; }
  diff <(awk '{print $1}' "$OUT/inputs_sha256.txt") <(sha256sum $PKG/gpab_env.py $PKG/gpab_run.py $PKG/gpab_score.py $PKG/prompts/patient_simulator_v1_1.txt $CASES | awk '{print $1}') \
    || { echo "RESUME: input hashes differ from the original run -- refusing"; exit 1; }
  for f in "$OUT"/episodes_weak.jsonl "$OUT"/episodes_strong.jsonl; do   # a job killed mid-write can leave a truncated last line
    [ -f "$f" ] && $VLLM_PY -c "import json,sys,shutil; p=sys.argv[1]; L=open(p).read().splitlines(); ok=[]
for l in L:
    try: json.loads(l); ok.append(l)
    except Exception: pass
if len(ok)!=len([l for l in L if l.strip()]): shutil.copy(p,p+'.pre_resume'); open(p,'w').write('\n'.join(ok)+'\n'); print('RESUME: dropped',len(L)-len(ok),'bad line(s) from',p)" "$f"
  done
  echo "RESUME $(date -Is): $(wc -l < "$OUT/episodes_weak.jsonl" 2>/dev/null || echo 0) weak, $(wc -l < "$OUT/episodes_strong.jsonl" 2>/dev/null || echo 0) strong episodes on disk" | tee -a "$OUT/resume.log"
fi
mkdir -p "$OUT"; cd "$OUT"
[ "$RESUME" = 1 ] || echo "CASESET=$CASESET PARSER=$PARSER GPAB_TOLERANT=$GPAB_TOLERANT GEN=$GEN GEN_CFG='$GEN_CFG' DECODE=$DECODE SIM=$SIMMODEL(bf16, SIMSIZE=$SIMSIZE) STRONG_AGENT=$STRONG(bf16)" | tee condition.txt
[ "$RESUME" = 1 ] || sha256sum $PKG/gpab_env.py $PKG/gpab_run.py $PKG/gpab_score.py $PKG/prompts/patient_simulator_v1_1.txt $CASES | tee inputs_sha256.txt
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
$VLLM_PY $PKG/gpab_run.py --data $DATA --cases $CASES --out $OUT --agent-tag weak   --agent-model $WEAK   --agent-url http://localhost:8001/v1 --sim-model $SIMMODEL --sim-url $SIM_URL --agent-decoding $DECODE
$VLLM_PY $PKG/gpab_run.py --data $DATA --cases $CASES --out $OUT --agent-tag strong --agent-model $STRONG --agent-url http://localhost:8002/v1 --sim-model $SIMMODEL --sim-url $SIM_URL --agent-decoding $DECODE
# score: deterministic metrics + FLOPs + Dx/Tx judge (72B bf16, C.2 prompts)
$VLLM_PY $PKG/gpab_score.py --run $OUT --data $DATA --cases $CASES --judge-url http://localhost:8002/v1 --judge-model $STRONG
pkill -f "vllm.entrypoints.openai.api_server" || true
echo "ALIGN RUN DONE -> $OUT"
