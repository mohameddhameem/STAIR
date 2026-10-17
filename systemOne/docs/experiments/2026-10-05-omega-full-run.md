# Experiment 1: minimal routing experiment on omega

The first full run of the minimal experiment from [research-report.md](../research-report.md):
route each HotpotQA question to a small or a large reader, using nano-jev evidence features.
Run on 2026-10-05 on omega's L40S, following [experiment-protocol.md](../experiment-protocol.md).

**Result:** the best router keeps 99.6% of the large reader's F1 at 37% of its cost. Evidence
features clearly beat question-only routing, but beat a small-model entropy threshold only
slightly.

## Setup

| | |
|---|---|
| Config | [`configs/gpu.toml`](../../configs/gpu.toml), run directory `runs/gpu` |
| Scorer | nano-jev v1.0 (`sdmlai/nano-jev@v1.0`, revision `ffee345e`, 33M MiniLM): relevance for each of the 10 paragraphs, sufficiency over the top 2 |
| Small expert | Qwen2.5-1.5B-Instruct (revision `989aa798`), bfloat16, reads the **top-2** paragraphs by nano-jev relevance |
| Large expert | Qwen2.5-7B-Instruct (revision `a09a3545`), bfloat16, reads **all 10** paragraphs |
| Routing label | τ = 0.8: an expert counts as correct at F1 ≥ 0.8. Two labels: `small-fails` and `large-helps` |
| Routers | Logistic regression on 3 feature sets: question (5 features), evidence (5), question+evidence (10). C tuned by 5-fold CV on router_train, temperature-scaled on calib |
| Operating point | Chosen on calib: the cheapest threshold within 1% F1 of always-large |
| Cost | GFLOPs per question, 2 × parameters × tokens (1.54B and 7.62B). The scorer adds ~400 GFLOPs per question for every router; this is reported separately and not added to the curves |

**Data** (`make-splits`, seed 42, `level == "hard"` only):

| Split | Source | Questions |
|---|---|---|
| router_train | HotpotQA distractor train | 12,000 |
| calib | HotpotQA distractor train | 2,000 |
| test | HotpotQA distractor validation (all) | 7,405 |
| test, clean | validation without nano-jev's 1,000 validation questions | 6,405 |

The 1,065 train questions nano-jev was trained on are excluded from router_train and calib.

## Execution

| | |
|---|---|
| Code | commit `a21e0be` (no uncommitted changes, recorded in `runs/gpu/provenance.jsonl`) |
| Hardware | omega `a-11`: 1 × NVIDIA L40S 46 GB, 8 CPUs, 64 GB RAM |
| Software | Python 3.12.15, torch 2.14.1+cu126, transformers 5.17.0, scikit-learn 1.9.1 |
| Job | PBS job 42199, `qsub -v STEP=full scripts/omega.pbs`, with backup jobs 42200 and 42201 (`afternotok`) |
| Outcome | Exit 0 on the first attempt, 2 h 15 min. No retries were used and the backup jobs never ran |
| Log | `runs/logs/42199.log` |

Steps beforehand, all on 2026-10-05:

1. `bash scripts/run_cluster.sh prepare` on the login node: data, splits and model downloads.
2. GPU check jobs 42194–42196. The CUDA 13 build of torch could not use driver 555, so torch
   was pinned to cu126 ([omega-cluster.md](../omega-cluster.md)).
3. 50-question pilot, job 42198 (`runs/omega-test-pilot`). It ran without errors and estimated
   2.1 GPU hours for the full run.

Wall-clock time per stage, from `provenance.jsonl` timestamps:

| Split | Evidence | Small | Large |
|---|---|---|---|
| router_train (12,000) | 8 min | 4 min | 50 min |
| calib (2,000) | 1.5 min | 1 min | 8.5 min |
| test (7,405) | 5 min | 3 min | 31 min |

Training the routers and running both evaluations took a further ~23 minutes on CPU.
Throughput was 0.020 s per question for the small expert and 0.248 s for the large one. The
1.5B reading 2 paragraphs is 12× faster than the 7B reading 10.

Output checks: router_train, calib and test have 12,000, 2,000 and 7,405 rows. None has a missing
prediction or a duplicate question id. The log has no errors or tracebacks.

## Results

All numbers are on the 7,405-question test set (`runs/gpu/report.json`) unless marked clean.

### Experts and oracle

| | F1 | EM | GFLOPs/question | Sent to large |
|---|---|---|---|---|
| Always small | 0.613 | 0.483 | 1,051 | 0% |
| Always large | 0.669 | 0.530 | 22,762 | 100% |
| Oracle (cheapest correct expert) | 0.772 | 0.631 | 5,638 | 21% |

- **Evidence:** the gold pair of paragraphs is in nano-jev's top 2 for 75.4% of questions.
  12.3% of questions had at least one scorer input truncated at 512 tokens.
- **Label rates:** `small-fails` 46.3%, `large-helps` 15.4%.
- **Oracle headroom:** the oracle beats always-large by 0.10 F1 while sending only 21% of
  questions to it. On many questions the small expert with 2 good paragraphs beats the large
  expert with all 10.

### Routers

AIQ is the area under the F1-vs-cost convex hull (higher is better). CPT 80% is the cost at
which a method first recovers 80% of the F1 gap between always-small and always-large.

| Method | AIQ | CPT 80% (GFLOPs) | Operating point: F1 | GFLOPs | Sent to large |
|---|---|---|---|---|---|
| Random | 0.641 | 18,419 | | | |
| Small-model entropy threshold (after small runs) | 0.662 | 7,861 | | | |
| question / small-fails | 0.641 | 19,541 | 0.657 | 19,313 | 84% |
| question / large-helps | 0.648 | 16,277 | 0.647 | 11,505 | 48% |
| evidence / small-fails | 0.661 | 6,599 | 0.663 | 6,645 | 25% |
| evidence / large-helps | 0.663 | **5,466** | **0.667** | 8,503 | 34% |
| question+evidence / small-fails | 0.656 | 9,777 | 0.661 | 11,726 | 49% |
| question+evidence / large-helps | **0.665** | 6,454 | 0.660 | **5,933** | 23% |
| Oracle | 0.755 | 5,638 | | | |

- **evidence / large-helps** gives the best F1: 0.667 against 0.669 for always-large, at 37%
  of its cost.
- **question+evidence / large-helps** has the best AIQ and the cheapest operating point: 0.660
  F1 at 26% of always-large's cost.

### Calibration (expected calibration error on test, before → after temperature scaling)

| Router | small-fails | large-helps |
|---|---|---|
| question | 0.018 → 0.019 (T 0.80) | 0.015 → 0.005 (T 0.94) |
| evidence | 0.022 → 0.021 (T 0.98) | 0.017 → 0.011 (T 0.97) |
| question+evidence | 0.017 → 0.016 (T 1.00) | 0.016 → 0.013 (T 0.96) |

All routers are already well calibrated. Temperature scaling helps the `large-helps` routers a
little. The reliability bins are in `report.json`.

### Breakdown at the operating point (F1, share sent to large)

| Router | bridge | comparison | span | yes/no |
|---|---|---|---|---|
| Always small | 0.613 | 0.615 | 0.609 | 0.681 |
| Always large | 0.664 | 0.689 | 0.659 | 0.823 |
| evidence / large-helps | 0.679, 40% | 0.617, 11% | 0.665, 36% | 0.697, 12% |
| question+evidence / large-helps | 0.665, 21% | 0.643, 27% | 0.654, 21% | 0.760, 38% |
| question / large-helps | 0.639, 43% | 0.680, 70% | 0.637, 46% | 0.810, 86% |

- **Bridge questions:** the evidence routers escalate the right ones. evidence / large-helps
  beats always-large on bridge questions (0.679 against 0.664) while escalating only 40%.
- **Comparison and yes/no questions:** the evidence routers rarely escalate these (11–12%), and
  lose most of the large expert's yes/no advantage (0.697 against 0.823). The question-only
  router escalates them more, which is why the combined router does better on yes/no than
  evidence alone.

### Without nano-jev's validation questions (`report-clean.json`, 6,405 questions)

Overall F1, EM and AIQ move by less than 0.003. Examples: always-large F1 is 0.671, AIQ for
question+evidence / large-helps is 0.666, and evidence / large-helps scores F1 0.668 at 8,459
GFLOPs. So there is no sign that nano-jev having seen these questions inflates the results.

## Findings

1. **Evidence features carry the routing signal.** Evidence-only routers reach AIQ 0.661–0.663.
   Question-only routers reach 0.641–0.648, and question / small-fails scores exactly the random
   AIQ: its curve never rises above the straight line between the two experts, so it adds
   nothing.
2. **`large-helps` beats `small-fails`** for every feature set, as the protocol predicted.
   `small-fails` also escalates questions both experts miss, which wastes large-expert calls.
3. **The gain over the entropy baseline is small.** The best AIQ is 0.665 against 0.662 for the
   small model's own entropy. The entropy baseline also needs the small model to run first,
   whereas the routers decide before either expert runs. A significance test (paired bootstrap
   over questions) is needed before claiming a difference.
4. **Most of the oracle headroom is unused.** The best AIQ is 0.665 against 0.755 for the oracle.
   The oracle escalates 21% and gains 0.10 F1 over always-large. No router beats always-large
   overall, because none predicts well when the small expert is *right*.
5. **The weak spot is comparison and yes/no questions**, where the evidence features hardly
   ever escalate.

## Not done yet

Required by [experiment-protocol.md](../experiment-protocol.md) but not in this run:

(τ sensitivity and significance tests were done in
[experiment 2](2026-10-06-cascade-routers.md): the +0.003 AIQ over the entropy baseline is not
significant, CI [−0.002, +0.009].)

- sensitivity to τ (0.8 only so far);
- a hand audit of ~200 EM/F1 disagreements;
- reliability diagram plots (the data is in `report.json`);
- confidence intervals and significance tests for the AIQ differences;
- `configs/gpu-vllm.toml`, which hasn't been tried on omega yet.

## Reproducing

```bash
bash scripts/run_cluster.sh prepare                 # login node
qsub -v STEP=full scripts/omega.pbs                 # ~2.25 h on one L40S
```

The results are in `runs/gpu/`: `report.json`, `report-clean.json`, `curves.csv`,
`curves-clean.csv`, `routers.pkl` and the per-question caches (`router_train.parquet`,
`calib.parquet`, `test.parquet`). `runs/` is not tracked by git: copy it somewhere permanent
before cleaning up.
