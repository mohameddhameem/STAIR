#!/bin/bash
# twowiki_rowc.sbatch -- Row C: "AgentTTS (reproduced), dev[500:1000]" on 2WikiMultiHopQA, SMU cluster, one H200.
# DRAFT v1.0 2026-11-01 -- submit ONLY after Jonathan's go. Same as twowiki_record.sbatch (job 347282) except the wrapper
# (twowiki_rowc_run.sh: --test, 500 instances from ROWC_WORK) and the job name. ~5.5 h expected (the 100 took 1 h 07 m).
# Submit:  cd ~/slr && sbatch ~/slr/STAIR/lab_package/twowiki/twowiki_rowc.sbatch
#SBATCH --job-name=twowiki_rowc
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=twowiki_rowc.%j.log

export ARCHON_ENV=$HOME/slr/archon-env
export AGENTTTS_REPO=$HOME/slr/agenttts
export STAIR_ROOT=$HOME/slr/STAIR
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export OUTROOT=/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki
export ROWC_WORK=/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/rowc_work      # holds data/2wikimultihopqa/tts_test_2_samples500.json

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash $STAIR_ROOT/lab_package/twowiki/twowiki_rowc_run.sh
