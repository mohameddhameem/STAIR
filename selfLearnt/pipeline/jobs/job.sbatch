#!/bin/bash
# Generic job: runs one stage of run.py. Submitted by jobs/submit_all.py, which sets the resources and STAGE_ARGS.
# A stage that reaches its time limit, or a pre-empted job, is requeued and resumes from its last saved point.
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out

cd "${SLURM_SUBMIT_DIR:-.}"
source jobs/cluster.env
[ -f jobs/cluster.local.env ] && source jobs/cluster.local.env   # your own settings, never committed
conda deactivate 2>/dev/null || true   # the login shell starts in Anaconda base: use the module Python instead
module purge 2>/dev/null || true
module load "$PY_MODULE" 2>/dev/null || true
module load "${CUDA_MODULE:-CUDA/12.4.0}" 2>/dev/null || true
source "$VENV/bin/activate"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$(python -c "import sys; sys.path.insert(0, '.'); from stair.config import load_config; print(load_config('$CONFIG')['paths']['hf_cache'])")}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0   # no FlashInfer JIT compile (CUDA toolkit mismatch on the cluster)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "job $SLURM_JOB_ID on $(hostname): $STAGE_ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

CONFIG="${STAIR_CONFIG:-$CONFIG}"     # the submitter passes the config each job belongs to
python run.py --config "$CONFIG" $STAGE_ARGS
code=$?
if [ "$code" = "3" ]; then
  echo "time limit reached inside the job: requeueing $SLURM_JOB_ID to continue"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit $code
