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
#   bash scripts/run_cluster.sh exp8-eval      # re-evaluate finished experiment 8 judges, PLANS=<configs> (CPU job)
#   bash scripts/run_cluster.sh exp9-smoke     # experiment 9: the Strands Decider wrapper against its own engine (GPU, ~10 min)
#   bash scripts/run_cluster.sh exp9-judge7    # experiment 9 on experiment 7's pair A: zero-shot + fine-tuned deciders (GPU, hours)
#   bash scripts/run_cluster.sh exp9-judge8    # experiment 9 on experiment 8's five options, PLANS=<configs> (GPU, hours)
#   bash scripts/run_cluster.sh common-prepare # experiments 7+9 under the team's common setup: splits + dev subsets (CPU job)
#   bash scripts/run_cluster.sh common-experts # ... experts on the common train draw (GPU, ~1 h)
#   bash scripts/run_cluster.sh common-judges  # ... judges, J=zeroshot|train (GPU, hours); common-eval: reports (CPU job)
#   bash scripts/run_cluster.sh slr-fetch      # HotpotQA golden set: Jonathan's pinned SLR files (login node: download only)
#   bash scripts/run_cluster.sh slr-ladder     # ... verify them, menu A's ladder at Table 5 prices (CPU job)
#   bash scripts/run_cluster.sh exp10-train    # experiment 10: train one judge set, JUDGE=nano-jev|decider-2b|decider-e4b (GPU)
#   bash scripts/run_cluster.sh exp10-eval     # experiment 10: reports for every judge set trained so far (CPU job)
#   bash scripts/run_cluster.sh slr7-prepare   # experiments 7/9 on the golden set: pair A cut from runs/gpu (CPU job)
#   bash scripts/run_cluster.sh slr7-judges    # ... their judges, J=zeroshot|train, ONLY=... (GPU); slr7-eval: reports (CPU job)
#   bash scripts/run_cluster.sh exp8s-pools    # experiment 8 on the golden set: split + pools (CPU job); then exp8-score, exp8-answer, exp8-judge with EXP8_OUT/PLAN
#   bash scripts/run_cluster.sh nanojev-fetch  # Jev novelty E1: nano-jev's recipe + datasets (login node); nanojev-pretrain: its retrains (GPU)
# The Jev novelty steps (E0-E9) are documented in docs/gpu-jobs-jev.md.
# The common-setup steps are documented in docs/gpu-jobs-common.md; the golden-set steps in docs/gpu-jobs-slr.md.
# Experiment 9's steps are documented in docs/gpu-jobs-exp9.md.
# Where each step runs: login node = `prepare` (downloads + light parquet export), `exp8-prepare` and `slr-fetch`
# (downloads); CPU job (scripts/omega-cpu.pbs, queue short) = `eval`, `exp8-pools`, `exp8-recost`, `exp8-eval`,
# `exp8-rescore-summary`, `common-eval`, `slr-ladder`, `exp10-eval`, `slr7-prepare`, `slr7-eval`, `exp8s-pools`; every other step is a GPU job
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

STEP=${1:?usage: run_cluster.sh prepare|pilot|full|eval|judge-pilot|judge|exp8-prepare|exp8-score|exp8-pilot|exp8-answer|exp8-recost|exp8-judge|exp8-eval|exp8-rescore-validate|exp8-rescore|exp8-judge-graded|exp8-pools|exp8-rescore-summary|exp9-smoke|exp9-judge7|exp9-judge8|save-judges|common-prepare|common-experts|common-judges|common-eval|slr-fetch|slr-ladder|exp10-train|exp10-eval|slr7-prepare|slr7-judges|slr7-eval|exp8s-pools}
CONFIG=${CONFIG:-configs/gpu.toml}
RUN=${RUN:-runs/gpu}
DATA=${DATA:-data/hotpotqa}
ROUTER_SIZE=${ROUTER_SIZE:-12000}
CALIB_SIZE=${CALIB_SIZE:-2000}
PILOT=${PILOT:-500}
TAU=${TAU:-0.8}
BIN=${BIN-uv run}  # "uv run", or BIN="" with the project venv activated
REUSE=${REUSE:-}
EXP8_OUT=${EXP8_OUT:-runs/exp8}  # runs/exp8-common for the common-setup redo (docs/gpu-jobs-common.md)

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
    for split in ${SPLITS:-test calib router_train}; do $BIN pools score --out "$EXP8_OUT" --name "$split" --device cuda; done
    $BIN pools recall --out "$EXP8_OUT" --name test --exclude runs/exp8/nanojev_validation_ids.txt
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
  exp8-recost)  # CPU job: answer costs under AgentTTS Eq. 8 (prompt once per call) from the stored tokens, no model calls
    for split in ${SPLITS:-test calib router_train}; do
      $BIN pool-answer recost --plan "${PLAN:-configs/exp8.toml}" --split "$split"
    done
    $BIN pool-answer summary --plan "${PLAN:-configs/exp8.toml}"
    ;;
  exp8-judge)
    $BIN exp8-judge all --plan "${PLAN:-configs/exp8-judge.toml}" --exclude runs/exp8/nanojev_validation_ids.txt
    ;;
  exp8-rescore-validate)
    $BIN rescore validate --out "$EXP8_OUT" --grader "${GRADER:-Qwen/Qwen2.5-14B-Instruct}"
    ;;
  exp8-rescore)
    for split in ${SPLITS:-test calib router_train}; do
      $BIN rescore commit --out "$EXP8_OUT" --split "$split"   # the cells' own models (3B, 70B in 4-bit) finish their answers
      $BIN rescore run --out "$EXP8_OUT" --split "$split" --grader "${GRADER:-Qwen/Qwen2.5-14B-Instruct}"
    done
    $BIN rescore summary --out "$EXP8_OUT"
    ;;
  exp8-rescore-summary)  # CPU job: rewrites runs/exp8/rescore/summary.csv from the graded files
    $BIN rescore summary
    ;;
  exp8-judge-graded)
    $BIN exp8-judge all --plan "${PLAN:-configs/exp8-judge-graded.toml}" --exclude runs/exp8/nanojev_validation_ids.txt
    ;;
  exp8-eval)  # CPU job: the reports of judges already trained (loads no model), PLANS=<judge configs>
    for plan in ${PLANS:?PLANS=<exp8 judge configs>}; do
      $BIN exp8-judge eval --plan "$plan" --exclude runs/exp8/nanojev_validation_ids.txt
    done
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
  exp8c-pools)  # CPU job, after common-experts: experiment 8's pools for the common setup in runs/exp8-common.
    # test = dev[500:1000] as before: its pools, scores, answers and grades are copied (same questions, same seed).
    mkdir -p runs/exp8-common/pools runs/exp8-common/scores runs/exp8-common/answers runs/exp8-common/rescore
    for d in pools scores; do cp -n runs/exp8/$d/test.parquet runs/exp8-common/$d/; done
    cp -rn runs/exp8/answers/test runs/exp8-common/answers/
    [ -d runs/exp8/rescore/test ] && cp -rn runs/exp8/rescore/test runs/exp8-common/rescore/
    cp -n runs/exp8/nanojev_validation_ids.txt runs/exp8-common/
    [ -f runs/exp8-common/pools/router_train.parquet ] || $BIN pools build --out runs/exp8-common --source train \
      --ids runs/common/router_train.parquet --count "${EXP8_TRAIN:-3000}" --name router_train
    [ -f runs/exp8-common/pools/calib.parquet ] || $BIN pools build --out runs/exp8-common --source validation \
      --start 0 --count 500 --name calib
    ;;
  common-prepare)  # CPU job: the team's common splits, and calib/test cut from runs/gpu's dev cache (no model calls)
    $BIN make-splits --standard --exclude "$DATA/nanojev_train_ids.txt" --out "$DATA/splits-common.json"
    for split in calib test; do
      $BIN subset-cache --from runs/gpu --splits "$DATA/splits-common.json" --split "$split" --run runs/common --name "$split"
    done
    ;;
  common-experts)  # GPU: nano-jev + 1.5B + 7B on the common train draw (~1 h, as the old router_train)
    experts runs/common router_train "$DATA/distractor_train.parquet" --splits "$DATA/splits-common.json" --split router_train
    ;;
  common-judges)  # GPU: J=zeroshot|train (default train), PLAN (default the lenient plan), ONLY="--only <judge> ..."
    $BIN judge "${J:-train}" --plan "${PLAN:-configs/system-one-common.toml}" --data-dir "$DATA" \
      --out "${JOUT:-runs/common-judges}" ${ONLY:-} --save-model
    ;;
  common-eval)  # CPU job: reports for both metrics (all 500, and without nano-jev's questions) + the team results file
    $BIN judge eval --plan configs/system-one-common.toml --data-dir "$DATA" --out runs/common-judges \
      --exclude "$DATA/nanojev_validation_ids.txt"
    $BIN judge eval --plan configs/system-one-common-f1.toml --data-dir "$DATA" --out runs/common-judges-f1 \
      --exclude "$DATA/nanojev_validation_ids.txt"
    $BIN export-common --plan configs/system-one-common.toml --out runs/common-judges \
      --csv results/common/systemOne-hotpotqa-test.csv \
      --history results/common/training-history.csv --plot results/common/loss-curves.png
    ;;
  save-judges)  # retrain judges and keep their weights (for Hugging Face): PLAN=<config>, ONLY="--only <judge> ..."
    $BIN judge train --plan "${PLAN:?PLAN=<judge config>}" --data-dir "$DATA" --out "${JUDGE_OUT:-runs/judge-models}" ${ONLY:-} --save-model
    ;;
  exp9-judge8)
    for plan in ${PLANS:-configs/exp8-judge-decider-2b-zs.toml configs/exp8-judge-decider-2b-ft.toml configs/exp8-judge-decider-e4b-zs.toml configs/exp8-judge-decider-e4b-ft.toml}; do
      $BIN exp8-judge all --plan "$plan" --exclude runs/exp8/nanojev_validation_ids.txt
    done
    ;;
  slr-fetch)  # login node: downloads Jonathan's pinned files (sha256-checked), no compute
    $BIN slr fetch --plan "${SLR_PLAN:-configs/slr.toml}"
    ;;
  slr-ladder)  # CPU job: verify the golden set and write results/slr (seconds)
    $BIN slr all --plan "${SLR_PLAN:-configs/slr.toml}"
    ;;
  exp10-train)  # GPU: one judge set of configs/exp10-slr.toml, every (inputs, seed); resumes. A retrain that keeps the
                # weights: EXP10_OUT=<new run folder> SAVE=1
    $BIN exp10-slr train --judge "${JUDGE:?JUDGE=nano-jev|decider-2b|decider-e4b}" --plan "${PLAN:-configs/exp10-slr.toml}" \
      ${EXP10_OUT:+--out "$EXP10_OUT"} ${SAVE:+--save-model}
    ;;
  exp10-eval)  # CPU job: reports for every judge set trained so far, against Jonathan's routers (loads only his router's tokenizer, to price its calls)
    $BIN exp10-slr eval --plan "${PLAN:-configs/exp10-slr.toml}" ${EXP10_OUT:+--out "$EXP10_OUT"}
    ;;
  exp8s-pools)  # CPU job: experiment 8 on the golden set: its split (questions answered before drawn first) and pools
    $BIN make-splits --slr --router-size "${EXP8_TRAIN:-1000}" --calib-size "${EXP8_CALIB:-500}" \
      --prefer runs/exp8-common/answers/calib/llama3b-top5.parquet --prefer runs/exp8-common/answers/test/llama3b-top5.parquet \
      --out "$DATA/splits-exp8-slr.json"
    for split in router_train calib test; do  # pools are seeded per dev row, so the reused questions' pools rebuild exactly
      [ -f runs/exp8-slr/pools/$split.parquet ] || $BIN pools build --out runs/exp8-slr --source validation \
        --ids "$DATA/splits-exp8-slr.json" --count 100000 --name "$split"
    done
    cp -n runs/exp8/nanojev_validation_ids.txt runs/exp8-slr/
    ;;
  slr7-prepare)  # CPU job: experiments 7/9's pair A on the golden set, cut from runs/gpu's dev cache (no model calls)
    $BIN make-splits --slr --out "$DATA/splits-slr.json"
    for split in router_train calib test; do
      $BIN subset-cache --from runs/gpu --splits "$DATA/splits-slr.json" --split "$split" --run runs/slr-pairA --name "$split"
    done
    ;;
  slr7-judges)  # GPU: experiments 7/9's judges on the golden set, J=zeroshot|train (default train), ONLY="--only <judge> ..."
    $BIN judge "${J:-train}" --plan configs/system-one-slr.toml --data-dir "$DATA" --out runs/slr-judges ${ONLY:-}
    ;;
  nanojev-fetch)  # login node: E1's nano-jev recipe at its pinned commit, SQuAD 2.0 / MultiNLI and the bases (downloads only)
    $BIN nanojev-pretrain fetch --out runs/nanojev-pretrain
    ;;
  nanojev-pretrain)  # GPU: E1's retrained nano-jev arms, VARIANT=repro|nohotpot (default both), ~15 min each
    $BIN nanojev-pretrain train --out runs/nanojev-pretrain ${VARIANT:+--variant "$VARIANT"}
    ;;
  e1-curve)  # CPU job: E1's label curves and the pre-registered H1 tests (HotpotQA: all 1,481 and the 1,303 nano-jev
             # never saw); PLAN / JOUT / DATA for a confirmatory dataset
    EXCL=$DATA/nanojev_validation_ids.txt
    $BIN label-curve --plan "${PLAN:-configs/e1-label-curve.toml}" --out "${JOUT:-runs/e1-label-curve}"       $([ -f "$EXCL" ] && echo --exclude "$EXCL") --bootstrap "${BOOT:-1000}"
    ;;
  e2-prepare-ext)  # CPU job: 2Wiki's secondary test (test_ext) as pair X of configs/e2-2wikimultihopqa.toml
    mkdir -p runs/2wikimultihopqa-pairA-ext
    cp runs/2wikimultihopqa-pairA/{config.toml,router_train.parquet,calib.parquet} runs/2wikimultihopqa-pairA-ext/
    cp runs/2wikimultihopqa-pairA/test_ext.parquet runs/2wikimultihopqa-pairA-ext/test.parquet
    ;;
  e2-judges)  # GPU: E2's judges on DATASET=musique|2wikimultihopqa: nano-jev's zero-shot questions, then in-domain + frozen
    HF_HUB_OFFLINE=1 $BIN judge zeroshot --plan "configs/e2-${DATASET:?}.toml" --data-dir "data/$DATASET" --out "runs/e2-$DATASET"
    HF_HUB_OFFLINE=1 $BIN judge train --plan "configs/e2-$DATASET.toml" --data-dir "data/$DATASET" --out "runs/e2-$DATASET"
    ;;
  e2-eval)  # CPU job: E2's reports on DATASET (entropy, LR routers, judges, paired bootstrap)
    $BIN judge eval --plan "configs/e2-${DATASET:?}.toml" --data-dir "data/$DATASET" --out "runs/e2-$DATASET"
    ;;
  multihop-prepare)  # login node: E2's DATASET=2wikimultihopqa|musique download + light export, then its pre-registered splits
    $BIN prepare-multihop --dataset "${DATASET:?DATASET=2wikimultihopqa|musique}"
    $BIN make-splits --dataset "$DATASET"
    ;;
  multihop-pilot|multihop-experts)  # GPU: E2's pair A on DATASET, the pilot (PILOT_N test questions, default 50) or every split
    case ${DATASET:?DATASET=2wikimultihopqa|musique} in
      2wikimultihopqa) MCONFIG=configs/gpu-2wiki.toml; SOURCE=(train dev dev dev); PARTS=(router_train calib test test_ext) ;;
      musique) MCONFIG=configs/gpu-musique.toml; SOURCE=(train train dev); PARTS=(router_train calib test) ;;
      *) echo "unknown DATASET $DATASET"; exit 1 ;;
    esac
    MDATA=data/$DATASET
    if [ "$STEP" = multihop-pilot ]; then
      CONFIG=$MCONFIG experts "runs/$DATASET-pilot" pilot "$MDATA/dev.parquet" --splits "$MDATA/splits.json" --split test \
        --limit "${PILOT_N:-50}" --seed 0
    else
      for i in "${!PARTS[@]}"; do
        CONFIG=$MCONFIG experts "runs/$DATASET-pairA" "${PARTS[$i]}" "$MDATA/${SOURCE[$i]}.parquet" \
          --splits "$MDATA/splits.json" --split "${PARTS[$i]}"
      done
    fi
    ;;
  pairD-experts)  # GPU: E5's pair D (Llama-3.2-3B / 3.1-8B) on all 7,405 dev questions; evidence copied from runs/gpu.
                  # Both are gated: download them on the login node with your own token first (docs/gpu-jobs-jev.md 5.0)
    HF_HUB_OFFLINE=1 CONFIG=configs/gpu-pairD.toml REUSE=runs/gpu experts runs/gpu-pairD test "$DATA/distractor_validation.parquet"
    ;;
  e5-prepare)  # CPU job: E5's pairs B, C, D cut to the golden set's slr-* splits from their dev caches (no model calls)
    for p in B:runs/gpu-large-14b C:runs/gpu-small-3b D:runs/gpu-pairD; do
      for split in router_train calib test; do
        $BIN subset-cache --from "${p#*:}" --splits "$DATA/splits-slr.json" --split "$split" --run "runs/slr-pair${p%%:*}" --name "$split"
      done
    done
    ;;
  e5-judges)  # GPU: E5's judges, each trained on one pair and scored on every pair (~1-2 h)
    HF_HUB_OFFLINE=1 $BIN judge train --plan configs/e5-transfer.toml --data-dir "$DATA" --out runs/e5-transfer
    ;;
  e5-eval)  # CPU job: E5's reports per pair (all 1,481 and the clean 1,303)
    $BIN judge eval --plan configs/e5-transfer.toml --data-dir "$DATA" --out runs/e5-transfer       --exclude "$DATA/nanojev_validation_ids.txt"
    ;;
  confirm)  # CPU job: the pre-registered confirmatory tests on DATASET: H2 + H3 (musique, 2wikimultihopqa; 2Wiki's
            # secondary test X: H2), H3 exploratory on the HotpotQA golden set (hotpotqa), or H5 (DATASET=e5)
    case ${DATASET:?DATASET=musique|2wikimultihopqa|hotpotqa|e5} in
      hotpotqa) $BIN confirm --family H3 --plan configs/system-one-slr.toml --out runs/slr-judges                   --signals runs/slr-pairA/signals --exclude "$DATA/nanojev_validation_ids.txt" ;;
      e5) $BIN confirm --family H5 --plan configs/e5-transfer.toml --out runs/e5-transfer --exclude "$DATA/nanojev_validation_ids.txt" ;;
      *) for fam in H2 H3; do
           $BIN confirm --family "$fam" --plan "configs/e2-$DATASET.toml" --out "runs/e2-$DATASET" --signals "runs/$DATASET-pairA/signals"
         done
         if [ "$DATASET" = 2wikimultihopqa ]; then
           $BIN confirm --family H2 --plan configs/e2-2wikimultihopqa.toml --out runs/e2-2wikimultihopqa --pair X
         fi ;;
    esac
    ;;
  confirm-free)  # CPU job: H3's priced signals with their extra calls free, beside the family (descriptive)
    case ${DATASET:?DATASET=musique|2wikimultihopqa|hotpotqa} in
      hotpotqa) $BIN confirm --family H3 --free --plan configs/system-one-slr.toml --out runs/slr-judges --signals runs/slr-pairA/signals --exclude "$DATA/nanojev_validation_ids.txt" ;;
      *) $BIN confirm --family H3 --free --plan "configs/e2-$DATASET.toml" --out "runs/e2-$DATASET" --signals "runs/$DATASET-pairA/signals" ;;
    esac
    ;;
  e4-e8)  # CPU job: E4 (zero / few labels) and E8 (risk control) on DATASET=musique|2wikimultihopqa|hotpotqa
    case ${DATASET:?DATASET=musique|2wikimultihopqa|hotpotqa} in
      hotpotqa) EPLAN=configs/system-one-slr.toml; EOUT=runs/slr-judges; EX=(--exclude "$DATA/nanojev_validation_ids.txt") ;;
      *) EPLAN=configs/e2-$DATASET.toml; EOUT=runs/e2-$DATASET; EX=() ;;
    esac
    $BIN zero-label --plan "$EPLAN" --out "$EOUT" "${EX[@]}"
    if [ "$DATASET" = musique ]; then  # H6b: the HotpotQA judge calibrated on HotpotQA, tested on MuSiQue
      $BIN risk-control --plan "$EPLAN" --out "$EOUT" --shift-from configs/system-one-slr.toml runs/slr-judge-models
    else
      $BIN risk-control --plan "$EPLAN" --out "$EOUT" "${EX[@]}"
    fi
    ;;
  e1-menuA-train)  # GPU: E1 on menu A + E6, JUDGES=<judge sets of configs/e1-menuA.toml, ':'-separated: qsub -v splits
                   # on commas>; resumes
    HF_HUB_OFFLINE=1 $BIN exp10-slr train --plan configs/e1-menuA.toml --judge "${JUDGES//:/,}"
    ;;
  e1-menuA-eval)  # CPU job: E1 menu A + E6 reports, his routers priced
    $BIN exp10-slr eval --plan configs/e1-menuA.toml
    ;;
  e7-prepare)  # GPU, ~20 min: E7's pools (experiment 8's golden-set test), MS MARCO's scores on them, recall per ranker
    mkdir -p runs/e7-evidence/pools runs/e7-evidence/scores
    for d in pools scores; do cp -n runs/exp8-slr/$d/test.parquet runs/e7-evidence/$d/; done
    cp -n runs/exp8-slr/nanojev_validation_ids.txt runs/e7-evidence/
    HF_HUB_OFFLINE=1 $BIN pools score --out runs/e7-evidence --name test --scorer cross-encoder/ms-marco-MiniLM-L12-v2 \
      --kind reranker --column msmarco --device cuda
    $BIN pools recall --out runs/e7-evidence --name test --exclude runs/e7-evidence/nanojev_validation_ids.txt
    ;;
  e7-answer)  # GPU: E7's answers, CELLS=<cells of configs/e7-evidence.toml, ':'-separated> (reused cells cost nothing);
              # resumes by chunk
    only=(); for c in ${CELLS//:/ }; do only+=(--only "$c"); done
    HF_HUB_OFFLINE=1 $BIN pool-answer run --plan configs/e7-evidence.toml --split test "${only[@]}"
    ;;
  e7-summary)  # CPU job: E7's table and each cell against nano-jev's 20 fused samples (paired bootstrap)
    $BIN pool-answer summary --plan configs/e7-evidence.toml
    $BIN pool-answer compare --plan configs/e7-evidence.toml --split test --base llama3b-s20fuse-top5
    ;;
  e9-latency)  # GPU job (DEVICE=cuda) or CPU job (DEVICE=cpu, 8 threads): E9's ms per decision
    HF_HUB_OFFLINE=1 $BIN latency --device "${DEVICE:-cuda}" --threads 8 --out runs/e9-latency
    ;;
  tests)  # CPU job: the test suite where torch is installed, TESTS=<pytest -k expression>; gates a GPU resubmission
    $BIN python -m pytest -q tests ${TESTS:+-k "$TESTS"}
    ;;
  e3-signals)  # GPU: E3's weak-model signals on DATASET=hotpotqa|musique|2wikimultihopqa (pair A's every split), ~1 h
    case ${DATASET:?DATASET=hotpotqa|musique|2wikimultihopqa} in
      hotpotqa) SRUN=runs/slr-pairA; SDATA=$DATA; SARGS="--split router_train --split calib --split test"
        SFILES="--split-data distractor_validation.parquet --split-data distractor_validation.parquet --split-data distractor_validation.parquet" ;;
      musique) SRUN=runs/musique-pairA; SDATA=data/musique; SARGS="--split router_train --split calib --split test"
        SFILES="--split-data train.parquet --split-data train.parquet --split-data dev.parquet" ;;
      2wikimultihopqa) SRUN=runs/2wikimultihopqa-pairA; SDATA=data/2wikimultihopqa
        SARGS="--split router_train --split calib --split test --split test_ext"
        SFILES="--split-data train.parquet --split-data dev.parquet --split-data dev.parquet --split-data dev.parquet" ;;
      *) echo "unknown DATASET $DATASET"; exit 1 ;;
    esac
    HF_HUB_OFFLINE=1 $BIN signals --run "$SRUN" --data-dir "$SDATA" $SARGS $SFILES
    ;;
  e1-judges)  # GPU: E1's label-budget grid on pair A, MATCH=<name substring> (e.g. qp-large / qpa-large); resumes
    HF_HUB_OFFLINE=1 $BIN judge train --plan "${PLAN:-configs/e1-label-curve.toml}" --data-dir "$DATA" \
      --out "${JOUT:-runs/e1-label-curve}" ${MATCH:+--match="$MATCH"}
    ;;
  slr7-eval)  # CPU job: reports on slr-test, all 1,481 and without nano-jev's questions (loads no model); JOUT=<judge run>
    $BIN judge eval --plan configs/system-one-slr.toml --data-dir "$DATA" --out "${JOUT:-runs/slr-judges}" \
      --exclude "$DATA/nanojev_validation_ids.txt"
    ;;
  *)
    echo "unknown step: $STEP (prepare|pilot|full|eval|judge-pilot|judge|exp8-prepare|exp8-score|exp8-pilot|exp8-answer|exp8-recost|exp8-judge|exp8-eval|exp8-rescore-validate|exp8-rescore|exp8-judge-graded|exp8-pools|exp8-rescore-summary|exp9-smoke|exp9-judge7|exp9-judge8|save-judges|common-prepare|common-experts|common-judges|common-eval|slr-fetch|slr-ladder|exp10-train|exp10-eval|slr7-prepare|slr7-judges|slr7-eval|exp8s-pools)" >&2
    exit 2
    ;;
esac
