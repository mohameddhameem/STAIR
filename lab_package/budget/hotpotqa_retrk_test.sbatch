#!/bin/bash
# hotpotqa_retrk_test.sbatch -- §18 HELD-OUT TEST run of the retrieval-stage k cells (WS path), cluster copy of the lab run started 24 Nov 15:34
# (lab unreachable since 24 Nov ~19:30). Same settings as the pilot (k_ret 10, T 0.7, seed 20261118, fp16, --retr-out 320); only SPLIT=test,
# the output folder and the chunk ranges differ. 25 Nov 2026.
#   retr chunks (1 x H200 each):  for S in 0 200 400 600 800 1000 1200 1400; do E=$((S+200)); [ $E -gt 1481 ] && E=1481;
#       PHASE=retr START=$S END=$E sbatch --gres=gpu:h200:1 ~/slr/STAIR/lab_package/budget/hotpotqa_retrk_test.sbatch; done
#   qa_strong (after all chunks; 2 x H200):  PHASE=qa_strong START=0 END=1481 sbatch --gres=gpu:h200:2 ~/slr/STAIR/lab_package/budget/hotpotqa_retrk_test.sbatch
#SBATCH --job-name=retrk_test
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=retrk_test.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110
export SLR_SHUFFLE=off
PHASE=${PHASE:?set PHASE=retr or PHASE=qa_strong}; START=${START:?}; END=${END:?}
OUT=${OUT:-/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/retrk_test1481}
SCRIPT=$HOME/slr/STAIR/lab_package/budget/hotpotqa_retrk_cells.py
RUNNER=$HOME/slr/run_labels_agenttts.py
source $HOME/slr/slr-env/bin/activate
mkdir -p "$OUT"; cd $HOME/slr
sha256sum "$SCRIPT" "$RUNNER" $HOME/slr/data/tts_test.json | tee -a "$OUT/inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COMMON="--split test --out $OUT --kret 10 --kans 40 --temp 0.7 --top-p 0.9 --seed 20261118 --dtype fp16 --retr-out 320 --start $START --end $END"
if [ "$PHASE" = "retr" ]; then
  python3 "$SCRIPT" $COMMON --phase retr
elif [ "$PHASE" = "qa_strong" ]; then
  python3 "$SCRIPT" $COMMON --phase qa_strong
else
  echo "PHASE must be retr or qa_strong"; exit 2
fi
ls -l "$OUT"
