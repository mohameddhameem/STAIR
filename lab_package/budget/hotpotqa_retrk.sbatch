#!/bin/bash
# hotpotqa_retrk.sbatch -- §18: k weak retrievals (u5/v5/u10/v10) + strong answer + 40 weak samples per context. DRAFT 18 Nov 2026 -- submit
# only after Jonathan's go. One job per split (SPLIT=val | test), two H200s (the 70B strong answerer in fp16, as every Config B cell).
# Submit:  cd ~/slr && SPLIT=val  sbatch ~/slr/STAIR/lab_package/budget/hotpotqa_retrk.sbatch
#          cd ~/slr && SPLIT=test sbatch ~/slr/STAIR/lab_package/budget/hotpotqa_retrk.sbatch
# Smoke:   cd ~/slr && SPLIT=val ARGS="--end 10" OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/retrk_smoke sbatch .../hotpotqa_retrk.sbatch
#SBATCH --job-name=hotpotqa_retrk
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=hotpotqa_retrk.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110
export SLR_SHUFFLE=off
SPLIT=${SPLIT:?set SPLIT=val or SPLIT=test}
OUT=${OUT:-/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/retrk_k10}
ARGS=${ARGS:-}
SCRIPT=$HOME/slr/STAIR/lab_package/budget/hotpotqa_retrk_cells.py
RUNNER=$HOME/slr/run_labels_agenttts.py
source $HOME/slr/slr-env/bin/activate
mkdir -p "$OUT"; cd $HOME/slr
sha256sum "$SCRIPT" "$RUNNER" $HOME/slr/data/tts_test.json | tee -a "$OUT/inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 "$SCRIPT" --split "$SPLIT" --out "$OUT" --phase all --kret 10 --kans 40 --temp 0.7 --top-p 0.9 --seed 20261118 --dtype fp16 $ARGS
ls -l "$OUT"
