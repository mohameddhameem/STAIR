#!/bin/bash
# twowiki_labels6000.sbatch -- the four cells (retrieval W/S x answering W/S) on the 6,000 2Wiki instances, as a Slurm
# array of 21 chunks (twowiki_label_chunks.py). DRAFT v1.0 2026-11-01 -- submit only after Jonathan's go.
# IDENTICAL to twowiki_cells.sbatch (job 346550, the 100) except: data = one pre-built chunk per array task (no adapter
# step: twowiki_pools.py already writes _id / answer / supporting_titles and asserts gold-in-pool), and one output
# folder per chunk (the runner keys stores by position in its data file).
#   retrieval  weak Qwen2.5-7B-Instruct   / strong Qwen2.5-72B-Instruct (bf16)      } the runner's own defaults =
#   answering  weak Llama-3.2-3B-Instruct / strong Llama-3.1-70B-Instruct (fp16)    } Config B on HotpotQA
#   K = 1, greedy; our prompts; SLR_POOL_K 110 (no trimming); SLR_SHUFFLE off (pool order = seeded order from the builder)
# Array order = chunk_list.txt = test600 (2) -> val400 (2) -> train5000 (17); at most 2 chunks run at once (%2 = 4 H200s).
# Submit:  cd ~/slr && sbatch ~/slr/STAIR/lab_package/twowiki/twowiki_labels6000.sbatch
#SBATCH --job-name=twowiki_labels6000
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --array=0-20%2
#SBATCH --output=twowiki_labels6000.%A_%a.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110
export SLR_SHUFFLE=off
CHUNKS=$HOME/slr/twowiki_pools6000/chunks                                   # scp'd from the lab (see README step)
ROOT=/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/labels6000
RUNNER=$HOME/slr/run_labels_agenttts.py
PY=$HOME/slr/slr-env/bin/python
source $HOME/slr/slr-env/bin/activate

NAME=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CHUNKS/chunk_list.txt")
[ -n "$NAME" ] || { echo "no chunk for task $SLURM_ARRAY_TASK_ID"; exit 1; }
DATA=$CHUNKS/$NAME.json
OUT=$ROOT/$NAME
if [ -f "$OUT/CHUNK_DONE" ]; then echo "REFUSING: $OUT already complete"; exit 0; fi
mkdir -p "$OUT"; cd $HOME/slr
# the chunk must be exactly the one the lab manifest lists
WANT=$($PY -c "import json,sys;print(json.load(open('$CHUNKS/chunks_manifest.json'))['chunks']['$NAME']['sha256'])")
GOT=$(sha256sum "$DATA" | cut -d' ' -f1)
[ "$WANT" = "$GOT" ] || { echo "CHUNK HASH MISMATCH for $NAME: want $WANT got $GOT"; exit 1; }
sha256sum "$RUNNER" "$DATA" | tee "$OUT/inputs_sha256.txt"
echo "task $SLURM_ARRAY_TASK_ID -> $NAME ($($PY -c "import json;print(len(json.load(open('$DATA'))))") instances)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for PH in retr_weak retr_strong qa_weak qa_strong; do
  echo "=== $NAME phase $PH $(date -Iseconds)"
  $PY "$RUNNER" --data "$DATA" --out "$OUT" --phase $PH
done
date -Iseconds > "$OUT/CHUNK_DONE"
ls -l "$OUT"
echo "CHUNK DONE -> $OUT"
