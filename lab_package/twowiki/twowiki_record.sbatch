#!/bin/bash
# twowiki_record.sbatch -- Row A RUN OF RECORD: "AgentTTS (reproduced), their 100" on 2WikiMultiHopQA, SMU cluster, one H200.
# DRAFT 31 Oct 2026 -- submit ONLY after Jonathan's explicit go. Same wrapper and paths as the smoke (job 346008, PASSED
# 31 Oct 21:09-21:10 on honey), without --smoke: 100 instances x 50 generations + fuser, their archon_main.py unchanged.
# The wrapper refuses to run if <OUTROOT>/repro_their100/sidecar.jsonl already exists (one run of record).
# Submit:  cd ~/slr && sbatch ~/slr/STAIR/lab_package/twowiki/twowiki_record.sbatch
# Watch:   squeue -u $USER ; tail -5 ~/slr/twowiki_record.<jobid>.log
#SBATCH --job-name=twowiki_record
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=twowiki_record.%j.log

export ARCHON_ENV=$HOME/slr/archon-env
export AGENTTTS_REPO=$HOME/slr/agenttts
export STAIR_ROOT=$HOME/slr/STAIR
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export OUTROOT=/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash $STAIR_ROOT/lab_package/twowiki/twowiki_repro_run.sh
