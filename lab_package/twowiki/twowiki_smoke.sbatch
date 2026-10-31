#!/bin/bash
# twowiki_smoke.sbatch -- Row A SMOKE on the SMU cluster (2 instances x 2 samples; outputs never reported).
# DRAFT 31 Oct 2026. Header copied from Jonathan's prod_retr_strong.sbatch (account / partition), one H200.
# Submit:  cd ~/slr && sbatch ~/slr/STAIR/lab_package/twowiki/twowiki_smoke.sbatch
# Watch:   squeue -u $USER ; tail -f ~/slr/twowiki_smoke.<jobid>.log
#SBATCH --job-name=twowiki_smoke
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=twowiki_smoke.%j.log

export ARCHON_ENV=$HOME/slr/archon-env
export AGENTTTS_REPO=$HOME/slr/agenttts
export STAIR_ROOT=$HOME/slr/STAIR
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export OUTROOT=/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash $STAIR_ROOT/lab_package/twowiki/twowiki_repro_run.sh --smoke
