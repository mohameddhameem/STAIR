#!/bin/bash
# hotpotqa_samplek.sbatch -- K > 1 weak-answerer cells on HotpotQA (WWx40, SWx40; val 1,110 + test 1,481). DRAFT 17 Nov 2026 --
# submit only after Jonathan's go. Config B grid runner imported unchanged. One H200.
# Smoke:  SPLIT=test ARGS="--end 10" OUT=<...>/samplek_smoke sbatch hotpotqa_samplek.sbatch
# Full:   SPLIT=val sbatch hotpotqa_samplek.sbatch ; SPLIT=test sbatch hotpotqa_samplek.sbatch   (same OUT; one samples_<split>.jsonl each)
#SBATCH --job-name=hp_samplek
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=hp_samplek.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110
export SLR_SHUFFLE=off
SPLIT=${SPLIT:-test}
OUT=${OUT:-/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/samplek_k40}
ARGS=${ARGS:-}
SCRIPT=$HOME/slr/STAIR/lab_package/budget/hotpotqa_samplek_cells.py
source $HOME/slr/slr-env/bin/activate
mkdir -p "$OUT"; cd $HOME/slr
sha256sum "$SCRIPT" $HOME/slr/run_labels_agenttts.py $HOME/slr/data/tts_test.json $HOME/slr/hotpotqa_stores/retr_weak.jsonl $HOME/slr/hotpotqa_stores/retr_strong.jsonl | tee -a "$OUT/inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 "$SCRIPT" --split "$SPLIT" --out "$OUT" --k 40 --temp 0.7 --top-p 0.9 --seed 20261117 $ARGS
ls -l "$OUT"
echo "SAMPLEK $SPLIT DONE -> $OUT"
