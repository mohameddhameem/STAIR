#!/usr/bin/env bash
# The whole experiment on a GPU machine reached over SSH (e.g. PuTTY through the university VPN).
#
#   bash scripts/run_cluster.sh prepare   # data, splits and model downloads; needs internet (login node)
#   bash scripts/run_cluster.sh pilot     # PILOT random router_train questions, prints F1 and a time estimate
#   bash scripts/run_cluster.sh full      # router_train, calib and all 7,405 validation questions
#   bash scripts/run_cluster.sh eval      # routers and reports only (CPU, seconds)
#   bash scripts/run_cluster.sh judge-pilot  # experiment 7: one System One judge, 3 seeds (GPU, ~15 min)
#   bash scripts/run_cluster.sh judge        # experiment 7: everything in configs/system-one.toml (GPU, ~3 h)
#   bash scripts/run_cluster.sh exp8-prepare # experiment 8: model + dataset downloads only (login node: network, no compute)
#   bash scripts/run_cluster.sh exp8-pools   # experiment 8: AgentTTS-style pools + nano-jev ids (CPU job: scripts/omega-cpu.pbs)
#   bash scripts/run_cluster.sh exp8-score   # experiment 8: nano-jev scores every pool (GPU, minutes)
#   bash scripts/run_cluster.sh exp8-pilot   # experiment 8: every answering cell on EXP8_PILOT (default 20) test questions (GPU)
#   bash scripts/run_cluster.sh exp8-answer  # experiment 8: the cells in CELLS (default all) on all splits (GPU)
#   bash scripts/run_cluster.sh exp8-judge   # experiment 8, phase 2: train and evaluate the option-choosing judge (GPU, <1 h)
#   bash scripts/run_cluster.sh exp8-rescore-validate  # experiment 8 re-scoring: check the commit step and the grader (GPU, ~30 min)
#   bash scripts/run_cluster.sh exp8-rescore           # self-commit, then grade, every cached answer in SPLITS (GPU, hours)
#   bash scripts/run_cluster.sh exp8-judge-graded      # the five-option judge on the grader's labels (GPU, ~20 min)
#   bash scripts/run_cluster.sh exp8-rescore-summary   # the re-scoring summary table only (CPU job: scripts/omega-cpu.pbs)
#   bash scripts/run_cluster.sh exp9-smoke     # experiment 9: the Strands Decider wrapper against its own engine (GPU, ~10 min)
#   bash scripts/run_cluster.sh exp9-judge7    # experiment 9 on experiment 7's pair A: zero-shot + fine-tuned deciders (GPU, hours)
#   bash scripts/run_cluster.sh exp9-judge8    # experiment 9 on experiment 8's five options, PLANS=<configs> (GPU, hours)
# Experiment 9's steps are documented in docs/gpu-jobs-exp9.md.
# Where each step runs: login node = `prepare` (downloads + light parquet export) and `exp8-prepare` (downloads); CPU job
# (scripts/omega-cpu.pbs, queue short) = `eval`, `exp8-pools`, `exp8-rescore-summary`; every other step is a GPU job
# (scripts/omega.pbs). Do not run compute on the login node.
# The re-scoring steps are documented step by step in docs/gpu-jobs-exp8-rescore.md.
#
# Every step resumes: after a disconnect or a killed job, run the same command again.
# An SSH disconnect kills foreground jobs, so start long steps inside tmux/screen, or with
#   nohup bash scripts/run_cluster.sh full > full.log 2>&1 &
# On omega (PBS), submit the GPU steps as jobs instead: qsub -v STEP=full scripts/omega.pbs (docs/omega-cluster.md).
# GPU nodes without internet: run `prepare` on the login node, then export HF_HUB_OFFLINE=1.
# REUSE=runs/gpu copies the stages whose settings match that run (e.g. evidence and small when only
# the large expert changed) instead of recomputing them: see run-experts --reuse-from.
set -euo pipefail

STEP=${1:?usage: run_cluster.sh prepare|pilot|full|eval|judge-pilot|judge|exp8-prepare|exp8-score|exp8-pilot|exp8-answer|exp8-judge|exp8-rescore-validate|exp8-rescore|exp8-judge-graded|exp8-pools|exp8-rescore-summary|exp9-smoke|exp9-judge7|exp9-judge8}
CONFIG=${CONFIG:-configs/gpu.toml}
RUN=${RUN:-runs/gpu}
DATA=${DATA:-data/hotpotqa}
ROUTER_SIZE=${ROUTER_SIZE:-12000}
CALIB_SIZE=${CALIB_SIZE:-2000}
PILOT=${PILOT:-500}
TAU=${TAU:-0.8}
BIN=${BIN-uv run}  # "uv run", or BIN="" with the project venv activated
REUSE=${REUSE:-}

experts() {  # experts <run> <name> <data file> [run-experts options...]; one model on the GPU per command
  local run=$1 name=$2 data=$3
  shift 3
  for stage in evidence small large merge; do
    $BIN run-experts --config "$CONFIG" --data "$data" --run "$run" --name "$name" --stage "$stage" \
      ${REUSE:+--reuse-from "$REUSE"} "$@"
  done
}

case $STEP in
  prepare)
    [ -f "$DATA/distractor_validation.parquet" ] || $BIN prepare-hotpotqa --out "$DATA"
    [ -f "$DATA/nanojev_validation_ids.txt" ] || $BIN nanojev-ids --out "$DATA"
    [ -f "$DATA/splits.json" ] || $BIN make-splits --train "$DATA/distractor_train.parquet" \
      --exclude "$DATA/nanojev_train_ids.txt" --router-size "$ROUTER_SIZE" --calib-size "$CALIB_SIZE" \
      --out "$DATA/splits.json"
    $BIN prefetch-models --config "$CONFIG"
    ;;
  pilot)
    experts "$RUN-pilot" pilot "$DATA/distractor_train.parquet" \
      --splits "$DATA/splits.json" --split router_train --limit "$PILOT"
    RUN_DIR="$RUN-pilot" N=$((ROUTER_SIZE + CALIB_SIZE + 7405)) $BIN python - <<'EOF'
import os
import pandas as pd

df = pd.read_parquet(os.path.join(os.environ["RUN_DIR"], "pilot.parquet"))
print(df[["answer", "small_pred", "large_pred"]].head(15).to_string())
n = int(os.environ["N"])
hours = n * (df["small_seconds"].mean() + df["large_seconds"].mean()) / 3600
print(f"\nsmall F1 {df.small_f1.mean():.3f} | large F1 {df.large_f1.mean():.3f} | "
      f"gold pair in top-2 {df.gold_pair_in_top2.mean():.3f}")
print(f"estimated GPU time for all {n:,} questions: {hours:.1f} h")
EOF
    ;;
  full)
    experts "$RUN" router_train "$DATA/distractor_train.parquet" --splits "$DATA/splits.json" --split router_train
    experts "$RUN" calib "$DATA/distractor_train.parquet" --splits "$DATA/splits.json" --split calib
    experts "$RUN" test "$DATA/distractor_validation.parquet"
    bash "$0" eval
    ;;
  eval)
    $BIN train-router --run "$RUN" --tau "$TAU"
    $BIN eval-routing --run "$RUN"
    $BIN eval-routing --run "$RUN" --exclude "$DATA/nanojev_validation_ids.txt"
    ;;
  judge-pilot)
    $BIN judge train --plan "${PLAN:-configs/system-one.toml}" --data-dir "$DATA" --out "${JUDGE_OUT:-runs/system-one}" \
      --only qpa-large-helps-nanojev-A
    ;;
  judge)
    $BIN judge all --plan "${PLAN:-configs/system-one.toml}" --data-dir "$DATA" --out "${JUDGE_OUT:-runs/system-one}" \
      --exclude "$DATA/nanojev_validation_ids.txt"
    ;;
  exp8-prepare)  # login node: downloads only; the pools are built by the CPU job `exp8-pools`
    $BIN python -c "from datasets import load_dataset; load_dataset('hotpotqa/hotpot_qa', 'distractor')"
    $BIN python -c "from huggingface_hub import snapshot_download as d; d('sdmlai/nano-jev', revision='v1.0'); [d(m, ignore_patterns=['original/*']) for m in ('meta-llama/Llama-3.2-3B-Instruct', 'meta-llama/Llama-3.1-70B-Instruct')]  # original/ is a second copy of the weights (141 GB for the 70B) that transformers does not load"
    ;;
  exp8-pools)  # CPU job (scripts/omega-cpu.pbs): rebuilds the seeded pools exactly
    [ -f runs/exp8/pools/test.parquet ] || $BIN pools build --source validation --start 500 --count 500 --name test
    [ -f runs/exp8/pools/router_train.parquet ] || $BIN pools build --source train --ids results/gpu/router_train.parquet --count 3000 --name router_train
    [ -f runs/exp8/pools/calib.parquet ] || $BIN pools build --source train --ids results/gpu/calib.parquet --count 1000 --name calib
    [ -f runs/exp8/nanojev_validation_ids.txt ] || $BIN nanojev-ids --out runs/exp8
    ;;
  exp8-score)
    for split in test calib router_train; do $BIN pools score --name "$split" --device cuda; done
    $BIN pools recall --name test --exclude runs/exp8/nanojev_validation_ids.txt
    ;;
  exp8-pilot)
    $BIN pool-answer run --plan "${PLAN:-configs/exp8.toml}" --split test --limit "${EXP8_PILOT:-20}" --chunk-size 10
    $BIN pool-answer summary --plan "${PLAN:-configs/exp8.toml}"
    ;;
  exp8-answer)
    only=(); for c in ${CELLS:-}; do only+=(--only "$c"); done
    for split in ${SPLITS:-test calib router_train}; do
      $BIN pool-answer run --plan "${PLAN:-configs/exp8.toml}" --split "$split" "${only[@]}"
    done
    $BIN pool-answer summary --plan "${PLAN:-configs/exp8.toml}"
    ;;
  exp8-judge)
    $BIN exp8-judge all --plan "${PLAN:-configs/exp8-judge.toml}" --exclude runs/exp8/nanojev_validation_ids.txt
    ;;
  exp8-rescore-validate)
    $BIN rescore validate --grader "${GRADER:-Qwen/Qwen2.5-14B-Instruct}"
    ;;
  exp8-rescore)
    for split in ${SPLITS:-test calib router_train}; do
      $BIN rescore commit --split "$split"   # the cells' own models (3B, 70B in 4-bit) finish their answers
      $BIN rescore run --split "$split" --grader "${GRADER:-Qwen/Qwen2.5-14B-Instruct}"
    done
    $BIN rescore summary
    ;;
  exp8-rescore-summary)  # CPU job: rewrites runs/exp8/rescore/summary.csv from the graded files
    $BIN rescore summary
    ;;
  exp8-judge-graded)
    $BIN exp8-judge all --plan "${PLAN:-configs/exp8-judge-graded.toml}" --exclude runs/exp8/nanojev_validation_ids.txt
    ;;
  exp9-smoke)  # must pass before the long experiment 9 jobs
    mkdir -p runs/exp9
    $BIN decider-smoke ${MODELS:-StrandsAgents/strands-decider-2B-hobson-v21 StrandsAgents/strands-decider-E4B-gemma4-v1-2610} \
      --out runs/exp9/smoke.json
    ;;
  exp9-judge7)  # J7STEP=zeroshot|train|eval|all; ONLY="--only <judge name>" to run one judge
    $BIN judge "${J7STEP:-all}" --plan configs/system-one-decider.toml --out runs/system-one ${ONLY:-} \
      --exclude data/hotpotqa/nanojev_validation_ids.txt
    ;;
  exp9-judge8)
    for plan in ${PLANS:-configs/exp8-judge-decider-2b-zs.toml configs/exp8-judge-decider-2b-ft.toml configs/exp8-judge-decider-e4b-zs.toml configs/exp8-judge-decider-e4b-ft.toml}; do
      $BIN exp8-judge all --plan "$plan" --exclude runs/exp8/nanojev_validation_ids.txt
    done
    ;;
  *)
    echo "unknown step: $STEP (prepare|pilot|full|eval|judge-pilot|judge|exp8-prepare|exp8-score|exp8-pilot|exp8-answer|exp8-judge|exp8-rescore-validate|exp8-rescore|exp8-judge-graded|exp8-pools|exp8-rescore-summary|exp9-smoke|exp9-judge7|exp9-judge8)" >&2
    exit 2
    ;;
esac
