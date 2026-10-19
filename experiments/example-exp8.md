```yaml
id: DH-08
owner: Dhameem
track: systemOne
status: rerun
title: Experiment 8 - sampled-and-fused 3B answers vs one 70B answer, with a learned judge
question: Can a judge tell when fusing several 3B answers is enough, and when to pay for the 70B?
hypothesis: Fusing 10-40 3B samples closes much of the gap to the 70B; a judge that reads the fused answer escalates only the rest.

code:
  repo: <private repo; shared on request>
  branch: feat/common-framework
  commit: <full 40-char SHA of the pushed commit>
  setup: uv sync --extra llm --extra decider
  commands:
    - bash scripts/run_cluster.sh exp8c-pools     # EXP8_TRAIN=1000
    - bash scripts/run_cluster.sh exp8-score
    - bash scripts/run_cluster.sh exp8-answer     # CELLS per group, see docs/gpu-jobs-common.md
    - bash scripts/run_cluster.sh exp8-rescore
    - bash scripts/run_cluster.sh exp8-judge      # PLAN=configs/exp8-judge-common.toml

data:
  dataset: hotpotqa
  splits:
    train: 1000      # router_train, from the common train draw
    val: 500         # dev[0:500]
    test: 500        # dev[500:1000]
  extra: none

models:
  - id: meta-llama/Llama-3.2-3B-Instruct
    params_b: 3
    quant: none
    role: small
  - id: meta-llama/Llama-3.3-70B-Instruct
    params_b: 70
    quant: 4bit
    role: large
  - id: sdmlai/nano-jev
    params_b: 0.033
    quant: none
    role: retriever, judge

compute:
  gpus: 2
  mem_gb: 96
  walltime_h: 24
  epochs: 3
  early_stopping: best val loss epoch

outputs:
  results_csv: results/common/exp8-hotpotqa-test.csv
  history_csv: results/common/exp8-training-history.csv
  other: none

metrics:
  headline: lenient
  extra: [em, f1, aiq, llm_grade]
  cost_unit: agenttts

reproduce:
  old_results: AIQ per judge on dev[500:1000], with F1 labels and 3,000 router_train questions
  source: docs/experiments/ (the experiment 8 report)
  differences: lenient labels, 1,000 router_train questions, AgentTTS units, the team val split

adopt:
  - path: src/stair/evaluate_routing.py
    why: paired bootstrap of AIQ, including each router's own cost; could move into stair_common

open_questions: none
```

The 70B cell needs 2 GPUs (4-bit). The 3B fusion cells each fit on 1 GPU, and are split into groups so that each
fits a 12 h walltime.
