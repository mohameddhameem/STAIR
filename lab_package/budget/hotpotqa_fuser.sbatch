#!/bin/bash
# hotpotqa_fuser.sbatch -- §20: AgentTTS's selected HotpotQA strategy on OUR pipeline: the 3B fuser (T 0.3, 512 tokens) over our stored SW x 40
# samples (samplek_k40, T 0.7). 24 Nov 2026. The fuser is a 3B call -> ANY GPU (24 GB is enough: 3B fp16 + a ~13k-token prompt); no H200 needed.
# Smoke:  SPLIT=test ARGS="--end 10" OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/fuser_smoke sbatch --gres=gpu:1 ~/slr/STAIR/lab_package/budget/hotpotqa_fuser.sbatch
# Full:   SPLIT=test sbatch --gres=gpu:1 ~/slr/STAIR/lab_package/budget/hotpotqa_fuser.sbatch        (val: SPLIT=val, same OUT)
#SBATCH --job-name=hp_fuser
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=hp_fuser.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110
export SLR_SHUFFLE=off
SPLIT=${SPLIT:?set SPLIT=val or test}
CELLS=${CELLS:-/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/samplek_k40}
OUT=${OUT:-/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/fuser_k40}
ARGS=${ARGS:-}
SCRIPT=$HOME/slr/STAIR/lab_package/budget/hotpotqa_fuser_cells.py
RUNNER=$HOME/slr/run_labels_agenttts.py
source $HOME/slr/slr-env/bin/activate
mkdir -p "$OUT"; cd $HOME/slr
sha256sum "$SCRIPT" "$RUNNER" $HOME/slr/data/tts_test.json "$CELLS/samples_$SPLIT.jsonl" | tee -a "$OUT/inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "SPLIT=$SPLIT CELLS=$CELLS OUT=$OUT ARGS=$ARGS"
python3 "$SCRIPT" --runner cluster --split "$SPLIT" --cells "$CELLS" --out "$OUT" --ctx S --k 40 --temp 0.3 --top-p 1.0 --max-new 512 --seed 20261124 --dtype fp16 $ARGS
ls -l "$OUT"
