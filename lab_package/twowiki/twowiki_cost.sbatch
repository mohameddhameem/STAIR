#!/bin/bash
# twowiki_cost.sbatch -- COST MEASUREMENT of the AgentTTS pipeline on their 100 (12 Nov 2026): same wrapper as twowiki_record.sbatch (job 347282),
# but calls twowiki_repro_cost_run.sh (OUT = repro_their100_cost; sitecustomize token logging). Pass ARGS=--smoke for the 2-instance smoke.
# Submit:  cd ~/slr && sbatch --export=ALL,ARGS=--smoke ~/slr/STAIR/lab_package/twowiki/twowiki_cost.sbatch   (smoke first), then without ARGS.
# The wrapper refuses to run if <OUTROOT>/repro_their100_cost/sidecar.jsonl already exists.
# Watch:   squeue -u $USER ; tail -5 ~/slr/twowiki_cost.<jobid>.log

#SBATCH --job-name=twowiki_cost
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=twowiki_cost.%j.log

export ARCHON_ENV=$HOME/slr/archon-env
export AGENTTTS_REPO=$HOME/slr/agenttts
export STAIR_ROOT=$HOME/slr/STAIR
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export OUTROOT=/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash $STAIR_ROOT/lab_package/twowiki/twowiki_repro_cost_run.sh ${ARGS:-}
