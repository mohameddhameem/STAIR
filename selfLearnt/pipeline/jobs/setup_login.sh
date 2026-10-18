#!/bin/bash
# Run once on the LOGIN node (needs internet): downloads the models (~600 GB) and the HotpotQA files.
# Long: run it inside tmux or screen, e.g.  tmux new -s setup  ->  bash jobs/setup_login.sh  ->  Ctrl-b d (detach).
# Resumable: run it again and it skips what is already there.
#   Llama models are gated: accept the licence on huggingface.co for Llama 3.2 1B, 3.2 3B, 3.1 8B and 3.1 70B, then
#   `huggingface-cli login` once (or export HF_TOKEN=...). Never commit a token.
set -e
cd "$(dirname "$0")/.."
source jobs/cluster.env
module purge 2>/dev/null || true
module load "$PY_MODULE" 2>/dev/null || true
module load "${CUDA_MODULE:-CUDA/12.4.0}" 2>/dev/null || true
source "$VENV/bin/activate"
export HF_HUB_ENABLE_HF_TRANSFER=1
python run.py --config "$CONFIG" --stage setup
echo "Setup done. Next: sbatch jobs/smoke.sbatch, then python jobs/submit_all.py"
