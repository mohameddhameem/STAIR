#!/bin/bash
#SBATCH --job-name=stair-retrieve
#SBATCH --output=logs/%x-%A_%a.out
# EDIT the partition name for the SMU cluster
#SBATCH --partition=EDIT_PARTITION
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
cd "${SLURM_SUBMIT_DIR:-.}"            # submit from the pipeline folder
source jobs/env.sh
python run.py --config "$CONFIG" --stage retrieve
python run.py --config "$CONFIG" --stage reach
