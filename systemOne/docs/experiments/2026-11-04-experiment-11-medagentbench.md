# Experiment 11: the System One judge on MedAgentBench, by replay

Plan: [experiment-11-plan.md](../experiment-11-plan.md). Job sheet: [gpu-jobs-exp11.md](../gpu-jobs-exp11.md).
Run on OMEGA, 2026-11-04. Jobs 44881–44887. The test set was read once (job 44887).

**Credit.** The replay table, the split, the cost accounting and the baselines are Jonathan's MedAgentBench campaign
(STAIR branch `jonathan`). Andy's per-task work (branch `selfLearnt`) and nonLLM's replay (branch `nonLLM`) are the
other two codebases on the same split; they are not used here beyond reading.

## Setup

- **Data.** Jonathan's `training1_table.json` (sha256 `186545f0…`), with the weak and strong episodes already recorded.
  The split is patient-disjoint: train 90 tasks (30 patients), val 60 (19), test 150 (49). Two task-1 items have no
  patient ID and count as one group each. The task file is `test_data_v2.json` (sha256 `b6e89b2e…`).
- **Provenance (Jonathan, 2026-11-04).** The patient records are de-identified real records from Stanford's STARR
  repository, released publicly with the benchmark (arXiv 2501.14654, §2.2).
- **Judges.** Zero-shot nano-jev v1.0 and the Strands Deciders (2B and E4B); fine-tuned nano-jev, MiniLM-L12 and
  MS MARCO MiniLM (5 seeds each); fine-tuned Decider 2B and E4B (3 seeds each). Each judge gives P(escalate) from the
  task and the weak attempt.
- **Operating points.** Chosen on val only, before the test read, and committed to the plan first (see the plan's
  "Operating points" section, `select.json` sha256 `37647bb9…`).
- **Cost.** Weak-first: delivering costs the weak episode, escalating costs the weak plus the strong episode, as
  Jonathan counts them. Units are as in `stair_common.cost`.
- **Metric.** AIQ over the weak-first cost range (our standard), with 95% intervals from a patient-level bootstrap.

## Results (test, 150 tasks, read once)

| Router | Accuracy at point A | Share of always-escalate cost | AIQ |
|---|---|---|---|
| "Finished" rule at point A (escalates all) | 54.00 | 100% | 0.330 (random) |
| Category policy | 63.33 | 67.7% | 0.524 |
| Category × finished | 63.33 | 67.7% | 0.529 |
| Zero-shot nano-jev | 53.33 | 99.6% | 0.383 |
| Zero-shot Decider 2B | 50.67 | 97.1% | 0.400 |
| Zero-shot Decider E4B | 53.33 | 99.6% | 0.431 |
| Fine-tuned nano-jev | 63.33 | 59.0% | 0.531 |
| Fine-tuned MiniLM | 59.33 | 52.1% | 0.521 |
| Fine-tuned MS MARCO | 62.67 | 85.5% | 0.522 |
| Fine-tuned Decider 2B | 62.00 | 58.1% | 0.532 |
| Fine-tuned Decider E4B | 62.67 | 56.3% | 0.532 |

Paired AIQ differences (patient bootstrap, 95% interval):

- Category × finished beats category: +0.004 (0.002, 0.007).
- Fine-tuned Decider 2B and E4B beat category × finished: +0.003 (0.002, 0.007) and +0.003 (0.002, 0.007).
- Fine-tuned nano-jev beats MS MARCO MiniLM: +0.009 (0.001, 0.016).
- Fine-tuned nano-jev vs category × finished: +0.003 (−0.0005, 0.007), p = 0.045. Borderline.
- Fine-tuned nano-jev vs plain MiniLM: +0.010 (−0.001, 0.024), p = 0.04. Borderline.

The "finished" row is the rule at point A, which picks the threshold that escalates every task (validation accuracy
56.67, the same as always-escalate). It is not the rule the plan intended. For reference only, and not a declared
point: escalating just the 85 test tasks whose weak attempt did not finish gives 34.67% accuracy at 75.3% of
always-escalate cost. That is below category routing.

## Hypotheses

- **H11a (zero labels):** not supported at the strict threshold. Zero-shot Decider E4B closes 50% of the gap between
  random and fine-tuned nano-jev (0.50); zero-shot nano-jev closes 26%. No interval was computed for the gap.
- **H11b (90 labels matches Jonathan's Training 1):** the fine-tuned judges reach 62–63% at 52–59% of always-escalate
  cost. Jonathan's Training 1 is 61.33% at 54.1%. The paired comparison needs his per-task decisions (pending; see
  below), so no equivalence claim is made.
- **H11c (decision pretraining, nano-jev over MiniLM at 90 labels):** directionally supported, interval touches zero.
- **H11d (the attempt adds signal beyond the template):** supported only weakly. The fine-tuned Deciders beat the
  category × finished policy by a small margin, and nano-jev does not clearly beat it. Most of the signal is
  type-level, as the plan predicted: the weak model is right almost only on task 1 and partly on task 4.

## Caveats

- **Exploratory.** n = 150 with one template per category. AIQ differences are around 0.01 and the intervals are wide
  once resampled by patient.
- **Second read of the test set.** Jonathan's routers and nonLLM's replay have already read it. This is the single
  read of our judges.
- **Serving setting.** All 72B episodes in the table ran with `repetition_penalty` 1.05 from the model's
  generation config (confirmed by Jonathan, 2026-11-04). Our results are on that setting. A 300-task 72B rerun with
  `--generation-config vllm` has been accepted and is Jonathan's to queue; it would be a new table, not a change to
  this one.
- **Jonathan's decisions not yet joined.** The `his_decisions` input is empty, so the report has no paired comparison
  with his routers. His flat per-task CSV is pending.

## Pending

1. Paired comparison with Jonathan's Training 1 and Training 3 decisions, once his CSV arrives.
2. Rerun on the `--generation-config vllm` table, if Jonathan queues it.
3. Confirm with Andy which serving path gave his 52.33 rerun.

## Files

- `results/exp11-mab/select.json`: validation results and declared points.
- `results/exp11-mab/test/report.json`: the full test report, including intervals and contrasts.
- `results/exp11-mab/test/decisions.parquet`: per-task decisions for each router and point.
- `results/exp11-mab/test/job-44887.log`: the test job's log.
