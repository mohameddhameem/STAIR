#!/bin/bash
# twowiki_cells.sbatch -- OUR four cells on AgentTTS's 2Wiki 100: the HotpotQA Config B grid runner, unchanged,
# on the adapted data file. DRAFT 31 Oct 2026 -- submit only after Jonathan's go.
#   retrieval  weak Qwen2.5-7B-Instruct   / strong Qwen2.5-72B-Instruct (bf16)      } the runner's own defaults =
#   answering  weak Llama-3.2-3B-Instruct / strong Llama-3.1-70B-Instruct (fp16)    } what Config B ran on HotpotQA
#   K = 1, greedy; our prompts (retrieval prompt + QA prompt with the short-answer line); POOL_K 110 (no trimming).
# Two H200s as in prod_retr_strong.sbatch (the runner sums visible GPU memory: strong phases need ~160 GB).
# Submit:  cd ~/slr && sbatch ~/slr/STAIR/lab_package/twowiki/twowiki_cells.sbatch
#SBATCH --job-name=twowiki_cells
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=twowiki_cells.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110          # pools are 100-104 passages: trim nothing (HotpotQA used 100 = full pool there)
export SLR_SHUFFLE=off         # as Config B: as-given pool order
SRC=$HOME/slr/agenttts/data/2wikimultihopqa/tts_test.json
OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/cells_their100
RUNNER=$HOME/slr/run_labels_agenttts.py
PY=$HOME/slr/slr-env/bin/python                                           # the env that ran Config B (torch 2.11, transformers 5.14)
source $HOME/slr/slr-env/bin/activate
mkdir -p "$OUT"; cd $HOME/slr
sha256sum "$RUNNER" "$SRC" | tee "$OUT/inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. adapt the data (ids, gold titles; nothing inside passages changes) -- asserts every gold is in its pool
python3 $HOME/slr/STAIR/lab_package/twowiki/twowiki_cells_adapt.py --src "$SRC" --out "$OUT"
DATA=$OUT/tts_test_2wiki_adapted.json

# 2. the four phases, the runner unchanged (its manifest guard records pool_k, models, versions)
for PH in retr_weak retr_strong qa_weak qa_strong; do
  echo "=== phase $PH $(date -Iseconds)"
  $PY "$RUNNER" --data "$DATA" --out "$OUT" --phase $PH
done
ls -l "$OUT"
echo "CELLS DONE -> $OUT   (score with twowiki_cells_score.py -- separate, reviewed script)"
