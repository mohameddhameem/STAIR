#!/bin/bash
# GPU smoke test of the whole pipeline on small models (configs/smoke.yaml). Submit with  python jobs/submit_all.py --smoke
# (adds your partition, account, QOS and GPU types from jobs/cluster.env).
# Check logs/stair-smoke-<jobid>.out; it ends with the results tables written. Then submit the full run.
#SBATCH --job-name=stair-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
cd "${SLURM_SUBMIT_DIR:-.}"
source jobs/cluster.env
[ -f jobs/cluster.local.env ] && source jobs/cluster.local.env   # your own settings, never committed
conda deactivate 2>/dev/null || true   # the login shell starts in Anaconda base: use the module Python instead
module purge 2>/dev/null || true
module load "$PY_MODULE" 2>/dev/null || true
module load "${CUDA_MODULE:-CUDA/12.4.0}" 2>/dev/null || true
source "$VENV/bin/activate"
export HF_HOME="$STAIR_PROJECT_DIR/hf_cache" TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STAIR_STATUS_PUSH=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -e
python run.py --config configs/smoke.yaml --stage all
python -c "
import glob, pandas as pd
from stair_common.schema import validate
from stair_common.training import validate_history
for f in sorted(glob.glob('$STAIR_PROJECT_DIR/runs/smoke/results_common/*.csv')): validate(pd.read_csv(f)); print('ok', f)
for f in sorted(glob.glob('$STAIR_PROJECT_DIR/runs/smoke/results_common/history/*.csv')): validate_history(pd.read_csv(f)); print('ok', f)
"
echo "SMOKE TEST PASSED"
