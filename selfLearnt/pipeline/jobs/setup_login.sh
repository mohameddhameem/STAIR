#!/bin/bash
# Run once on the LOGIN node (needs internet): downloads the models (~500 GB) and the HotpotQA files.
# Long: run it inside tmux or screen, e.g.  tmux new -s setup  ->  bash jobs/setup_login.sh  ->  Ctrl-b d (detach).
# Resumable: run it again and it skips what is already there.
#   Llama models are gated: accept the licence on huggingface.co for Llama 3.1 8B, 3.1 70B and 3.2 3B, then
#   `huggingface-cli login` once (or export HF_TOKEN=...). Never commit a token.
set -e
cd "$(dirname "$0")/.."
source jobs/cluster.env
module purge 2>/dev/null || true
module load "$PY_MODULE" 2>/dev/null || true
source "$VENV/bin/activate"
export HF_HUB_ENABLE_HF_TRANSFER=1
python run.py --config "$CONFIG" --stage setup
echo "Setup done. Next: python jobs/submit_all.py"
