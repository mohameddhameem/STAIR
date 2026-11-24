#!/bin/bash
# hotpotqa_retrk_tsweep.sbatch -- §19 TEMPERATURE SWEEP for the retrieval-stage k pilot (WS path): same 400 validation questions,
# k_ret 10 (levels 5/10), as hotpotqa_retrk_pilot.sbatch (T = 0.7); only TEMP, the output folder and the qa phase differ.
# 24 Nov 2026 (Jonathan 13:58 "we should run temperature sweep for WS and K > 1").
# Differences from hotpotqa_retrk_pilot.sbatch:
#   1. TEMP=<t> from the environment -> --temp $TEMP (the cells script applies it to the sampled retrievals; the strong answerer is greedy)
#   2. OUT defaults to retrk_tsweep_val400_t<TEMP>  (one folder per temperature; never the pilot's folder)
#   3. PHASE=qa_strong runs ONLY the strong answerer on u5/v5/u10/v10 (the WS(k_ret) cells); qa_weak is not run in the sweep
# Submit, per temperature T in 0.3 1.0 1.3 (GPU count on the command line, as for the pilot):
#   cd ~/slr && TEMP=$T PHASE=retr      START=0   END=200 sbatch --gres=gpu:h200:1 ~/slr/STAIR/lab_package/budget/hotpotqa_retrk_tsweep.sbatch
#   cd ~/slr && TEMP=$T PHASE=retr      START=200 END=400 sbatch --gres=gpu:h200:1 ~/slr/STAIR/lab_package/budget/hotpotqa_retrk_tsweep.sbatch
#   cd ~/slr && TEMP=$T PHASE=qa_strong START=0   END=400 sbatch --gres=gpu:h200:2 ~/slr/STAIR/lab_package/budget/hotpotqa_retrk_tsweep.sbatch   (after both retr chunks of that T)
#SBATCH --job-name=retrk_tsweep
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=retrk_tsweep.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110
export SLR_SHUFFLE=off
TEMP=${TEMP:?set TEMP=0.3, 1.0 or 1.3}
PHASE=${PHASE:?set PHASE=retr or PHASE=qa_strong}; START=${START:?}; END=${END:?}
OUT=${OUT:-/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/retrk_tsweep_val400_t${TEMP}}
SCRIPT=$HOME/slr/STAIR/lab_package/budget/hotpotqa_retrk_cells.py
RUNNER=$HOME/slr/run_labels_agenttts.py
source $HOME/slr/slr-env/bin/activate
mkdir -p "$OUT"; cd $HOME/slr
sha256sum "$SCRIPT" "$RUNNER" $HOME/slr/data/tts_test.json | tee -a "$OUT/inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "TEMP=$TEMP PHASE=$PHASE START=$START END=$END OUT=$OUT"
COMMON="--split val --out $OUT --kret 10 --kans 40 --temp $TEMP --top-p 0.9 --seed 20261118 --dtype fp16 --retr-out 320 --start $START --end $END"
if [ "$PHASE" = "retr" ]; then
  python3 "$SCRIPT" $COMMON --phase retr
elif [ "$PHASE" = "qa_strong" ]; then
  python3 "$SCRIPT" $COMMON --phase qa_strong
else
  echo "PHASE must be retr or qa_strong (qa_weak is not part of the sweep)"; exit 2
fi
ls -l "$OUT"
