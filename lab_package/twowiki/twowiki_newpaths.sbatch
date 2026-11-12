#!/bin/bash
# twowiki_newpaths.sbatch -- §15-A/C new-path cells on 2Wiki val400, K = 1 only (U-S, U-W, S-all, W-all). DRAFT 12 Nov 2026 --
# submit only after Jonathan's go. Config B grid runner imported unchanged (models, prompts, pools); see twowiki_newpaths_cells.py.
# Smoke first:  ARGS="--end 10" sbatch twowiki_newpaths.sbatch   (fresh OUT: OUT=<...>/newpaths_val400_smoke)
# Full:          sbatch twowiki_newpaths.sbatch
#SBATCH --job-name=twowiki_newpaths
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=twowiki_newpaths.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110
export SLR_SHUFFLE=off
RUN=/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/labels6000_merged/val400
OUT=${OUT:-/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/newpaths_val400}
ARGS=${ARGS:-}
SCRIPT=$HOME/slr/STAIR/lab_package/twowiki/twowiki_newpaths_cells.py
source $HOME/slr/slr-env/bin/activate
mkdir -p "$OUT"; cd $HOME/slr
sha256sum "$SCRIPT" $HOME/slr/run_labels_agenttts.py "$RUN/tts_test_2wiki_adapted.json" "$RUN/retr_weak.jsonl" "$RUN/retr_strong.jsonl" | tee "$OUT/inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 "$SCRIPT" --run "$RUN" --out "$OUT" --cells U-S U-W S-all W-all \
    --maxlen 8192 --dtype fp16 $ARGS       # K = 1 only (Jonathan 13:44); SWxk/fuser code kept in the script but NOT run
ls -l "$OUT"
echo "NEWPATHS DONE -> $OUT   (score: python3 -I $HOME/slr/STAIR/lab_package/twowiki/twowiki_newpaths_score.py --run $RUN --new $OUT)"
