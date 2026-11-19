#!/bin/bash
# hotpotqa_retrk_pilot.sbatch -- §18 PILOT: validation positions 0-399 (the first 400 of slr-val), k_ret 10 (levels 5/10), k_ans 40.
# 19 Nov 2026. Two kinds of job from one file:
#   PHASE=retr  START=0   END=200  -> one H200, sampled retrievals, chunk 1   (~3 h at ~50 s/question with --retr-out 320)
#   PHASE=retr  START=200 END=400  -> one H200, chunk 2
#   PHASE=qa    START=0   END=400  -> two H200s, qa_strong then qa_weak on the combined contexts (after BOTH retr chunks finished)
# Submit (GPU count on the command line, since it differs by phase):
#   cd ~/slr && PHASE=retr START=0   END=200 sbatch --gres=gpu:h200:1 ~/slr/STAIR/lab_package/budget/hotpotqa_retrk_pilot.sbatch
#   cd ~/slr && PHASE=retr START=200 END=400 sbatch --gres=gpu:h200:1 ~/slr/STAIR/lab_package/budget/hotpotqa_retrk_pilot.sbatch
#   cd ~/slr && PHASE=qa   START=0   END=400 sbatch --gres=gpu:h200:2 ~/slr/STAIR/lab_package/budget/hotpotqa_retrk_pilot.sbatch   (after both retr chunks)
#SBATCH --job-name=retrk_pilot
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=retrk_pilot.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110
export SLR_SHUFFLE=off
PHASE=${PHASE:?set PHASE=retr or PHASE=qa}; START=${START:?}; END=${END:?}
OUT=${OUT:-/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/retrk_pilot_val400}
SCRIPT=$HOME/slr/STAIR/lab_package/budget/hotpotqa_retrk_cells.py
RUNNER=$HOME/slr/run_labels_agenttts.py
source $HOME/slr/slr-env/bin/activate
mkdir -p "$OUT"; cd $HOME/slr
sha256sum "$SCRIPT" "$RUNNER" $HOME/slr/data/tts_test.json | tee -a "$OUT/inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COMMON="--split val --out $OUT --kret 10 --kans 40 --temp 0.7 --top-p 0.9 --seed 20261118 --dtype fp16 --retr-out 320 --start $START --end $END"
if [ "$PHASE" = "retr" ]; then
  python3 "$SCRIPT" $COMMON --phase retr
else
  python3 "$SCRIPT" $COMMON --phase qa_strong
  python3 "$SCRIPT" $COMMON --phase qa_weak
fi
ls -l "$OUT"
