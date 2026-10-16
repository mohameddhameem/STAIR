#!/bin/bash
#SBATCH --job-name=stair-test
#SBATCH --output=logs/%x-%A_%a.out
# EDIT the partition name for the SMU cluster
#SBATCH --partition=EDIT_PARTITION
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
cd "${SLURM_SUBMIT_DIR:-.}"            # submit from the pipeline folder
source jobs/env.sh
python run.py --config "$CONFIG" --stage test
