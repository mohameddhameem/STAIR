#!/bin/bash
# Runs jobs/submit_all.py itself as a small CPU job, so it keeps feeding the queue even if the login node drops tmux
# sessions. Started by:  python jobs/submit_all.py --as-job   (or --as-job --continue logs/plan_<time>.json)
#SBATCH --job-name=stair-submitter
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
cd "${SLURM_SUBMIT_DIR:-.}"
source jobs/cluster.env
[ -f jobs/cluster.local.env ] && source jobs/cluster.local.env
module purge 2>/dev/null || true
source "$VENV/bin/activate"
echo "submitter job $SLURM_JOB_ID on $(hostname): $SUBMIT_ARGS"
python -u jobs/submit_all.py $SUBMIT_ARGS
