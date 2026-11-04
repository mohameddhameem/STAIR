#!/usr/bin/env bash
# rerun72b_pureT0_lab.sh -- MedAgentBench 300-task 72B-AWQ rerun with PURE temperature 0 (--generation-config vllm), i.e.
# WITHOUT the model file's repetition_penalty 1.05 that all three campaign servers applied (logs 23/26 Oct: "Default vLLM
# sampling parameters have been overridden by the model's generation_config.json: {'repetition_penalty': 1.05, ...}").
# Requested by Dhameem 4 Nov (Experiment 11 paired comparison); to be reported as a SEPARATE table beside training1_table.json.
# DRAFT v1.0 2026-11-04 -- run only after Jonathan's go. Everything else = the 23-24 Oct reproduction: same harness clone,
# FHIR image, test_data_v2 (sha b6e89b2e...), refsol, prompts, max_round 8, temperature 0 / 2048, --max-model-len 11264,
# --gpu-memory-utilization 0.97 --enforce-eager. Output goes to a NEW folder (MedAgentBenchv1_pureT0); the canonical
# MedAgentBenchv1 record is never touched. Needs the GPU exclusively (~44.5 GB) -> start through the lab queue:
#   NEED_GB=44 LOG=/mnt/hdd/jonathan/runs/mab/gpu_wait_rerun72b.log \
#     bash /mnt/hdd/jonathan/STAIR/lab_package/tools/gpu_wait_run.sh bash /mnt/hdd/jonathan/STAIR/lab_package/medagentbench/rerun72b_pureT0_lab.sh
# After it: training1_build_table.py with STRONG pointed at the new overall.json -> training1_table_pureT0.json (+ sha256).
set -euo pipefail
MAB=/mnt/hdd/jonathan/medagentbench
RUNS=/mnt/hdd/jonathan/runs/mab
VLLM=/mnt/hdd/jonathan/vllm-env2/bin/vllm
MABPY=/mnt/hdd/jonathan/mab-env/bin/python
export HF_HOME=${HF_HOME:-/mnt/hdd/jonathan/hf_cache}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUTNAME=MedAgentBenchv1_pureT0
[ -e "$MAB/outputs/$OUTNAME" ] && { echo "REFUSING: outputs/$OUTNAME exists (the assigner would RESUME into it)"; exit 1; }
[ -f "$MAB/configs/assignments/mab72b_pureT0.yaml" ] || { echo "missing configs/assignments/mab72b_pureT0.yaml (copy of the 72B assignment with output -> $OUTNAME) -- create it first"; exit 1; }
curl -s localhost:8080/fhir/metadata >/dev/null || { echo "FHIR server (docker 'fhir', port 8080) not answering"; exit 1; }
pgrep -f "src.server.task_controller" >/dev/null || { echo "task controller not running -- start the task stack first (runbook)"; exit 1; }
mkdir -p "$RUNS"; cd "$MAB"
sha256sum data/medagentbench/test_data_v2.json src/server/tasks/medagentbench/refsol.py configs/assignments/mab72b_pureT0.yaml | tee "$RUNS/rerun72b_pureT0_inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
# server: identical to the campaign's, plus --generation-config vllm
nohup $VLLM serve Qwen/Qwen2.5-72B-Instruct-AWQ --max-model-len 11264 --gpu-memory-utilization 0.97 --enforce-eager \
      --generation-config vllm --port 8000 > "$RUNS/vllm_72b_pureT0.log" 2>&1 &
SERVER_PID=$!; trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do curl -s localhost:8000/v1/models >/dev/null 2>&1 && break; sleep 10; done
curl -s localhost:8000/v1/models | head -c 200; echo
# the proof line: with --generation-config vllm the override WARNING must be ABSENT
if grep -q "overridden by the model" "$RUNS/vllm_72b_pureT0.log"; then echo "STOP: the server still applied generation_config.json"; exit 1; fi
echo "generation-config override: absent (pure temperature 0 confirmed)" | tee -a "$RUNS/rerun72b_pureT0_inputs_sha256.txt"
export PATH=/mnt/hdd/jonathan/mab-env/bin:$PATH
$MABPY -m src.assigner -c configs/assignments/mab72b_pureT0.yaml 2>&1 | tee "$RUNS/rerun72b_pureT0_assigner.log"
ls -la "$MAB/outputs/$OUTNAME/"*/medagentbench-std/overall.json
$MABPY -c "import json,glob;o=json.load(open(glob.glob('$MAB/outputs/$OUTNAME/*/medagentbench-std/overall.json')[0]));print({k:v for k,v in o.items() if k!='custom'} if isinstance(o,dict) else type(o))"
echo "RERUN DONE -> outputs/$OUTNAME   (next: build training1_table_pureT0.json)"
