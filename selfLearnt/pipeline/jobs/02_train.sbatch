#!/bin/bash
#SBATCH --job-name=stair-train
#SBATCH --output=logs/%x-%A_%a.out
# EDIT the partition name for the SMU cluster
#SBATCH --partition=EDIT_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# one retriever per array task, two at a time (one per A100); 0-3 = the four retrievers in the config
#SBATCH --array=0-3%2
cd "${SLURM_SUBMIT_DIR:-.}"            # submit from the pipeline folder
source jobs/env.sh
# Stops cleanly before the time limit (exit 3); submit again to continue from the last resume point.
KEYS=($(python -c 'import yaml,os; print(" ".join(r["key"] for r in yaml.safe_load(open(os.environ["CONFIG"]))["retrievers"]))'))
export TIME_LIMIT_MIN=690
python run.py --config "$CONFIG" --stage train --retriever "${KEYS[$SLURM_ARRAY_TASK_ID]}"
