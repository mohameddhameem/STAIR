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
  generation config (confirmed by Jonathan, 2026-11-04). Andy's 72B runs used the same default path (confirmed
  2026-11-04, from the vLLM start-up log). Our results are on that setting. Andy's earlier 52.33 vs 56.00 gap was not
  the penalty (plain greedy gave 53.33). It came from his FHIR records server running on a UTC+8 clock, which is not
  part of this experiment, since the table is replayed.
- **Pure-temperature table.** Jonathan's rerun with `--generation-config vllm` (no repetition penalty, temperature 0)
  is on branch `jonathan` (`training1_table_pureT0.json`, sha256 `def234cd…`). Only the strong model's outcome
  changes, on 12 tasks (11 up, 1 down; task 5 +5, task 10 +2). The strong model scores 59.33 there, against 56.00
  on the canonical table, so always-escalate is 59.33 and the oracle 68.33. The penalty's effect on the 72B is about
  +3 points, concentrated in task 5, the magnesium orders. Both figures are single runs on different servings.
  Our routers were trained and read on the canonical table; the sensitivity analysis below repeats the pipeline on
  the pure-temperature table.

## Paired comparison with Jonathan's routers

His per-task decisions (`training1_3_test_decisions.csv` on branch `jonathan`, sha256 `f54b730b…`) are joined by
`tid` to our test decisions, with a patient-cluster bootstrap (5,000 resamples, seed 0). The script is
`scripts/exp11_paired_vs_jonathan.py`; the numbers are in `results/exp11-mab/paired_vs_jonathan.json`. This uses
decisions already made: no new test read. Its tallies match his, and it reproduces his Training 1 figures (61.33 at
54.1%; 60.00 at 51.2%).

| Our router (point A) | Accuracy | Cost (% of always-escalate) | vs Training 1, θ 0.50 (61.33 at 54.1%) | vs Training 3, θ 0.50 (63.33 at 81.5%) |
|---|---|---|---|---|
| Fine-tuned nano-jev | 63.33 | 59.0 | +2.00 acc [0.00, +4.52]; +4.9 cost [+2.3, +7.9] | 0.00 acc; −22.5 cost [−33.0, −12.8] |
| Fine-tuned Decider E4B | 62.67 | 56.3 | +1.33 acc [0.00, +3.47]; +2.2 cost [+0.4, +4.3] | −0.67 acc [−2.11, 0.00]; −25.2 cost [−36.0, −15.2] |
| Fine-tuned Decider 2B | 62.00 | 58.1 | +0.67 acc [0.00, +2.13]; +4.0 cost [+1.4, +6.9] | −1.33 acc [−3.43, 0.00]; −23.4 cost [−34.3, −12.7] |
| Fine-tuned MS MARCO | 62.67 | 85.5 | +1.33 acc [0.00, +3.40]; +31.4 cost [+19.9, +43.5] | −0.67 acc [−2.13, 0.00]; +4.0 cost [+0.7, +7.2] |
| Fine-tuned MiniLM | 59.33 | 52.1 | −2.00 acc [−4.64, 0.00]; −2.0 cost [−3.7, −0.4] | −4.00 acc [−7.33, −1.30]; −29.4 cost [−40.7, −18.4] |
| Category policy | 63.33 | 67.7 | +2.00 acc [0.00, +4.52]; +13.6 cost [+9.4, +18.2] | 0.00 acc; −13.8 cost [−24.1, −3.7] |

How to read it:

- **Against Training 1,** the judges are slightly more accurate at slightly higher cost. The accuracy intervals touch
  zero, so no claim of superiority is made.
- **Against Training 3,** the fine-tuned nano-jev matches its accuracy at about 22 points less cost. This is a
  different accounting: Training 3 pays its round-2 weak cost on delivered tasks, and our routers pay the full weak
  episode. The accuracy comparison is the cleaner one.
- The same 150 tasks are used for both routers, and both were read once: his by him, ours here.

## Pending

1. Jonathan's 300-task greedy rerun (`training1_table_pureT0.json`, sha256 to follow). Compare it with the current
   table when he shares it. Jonathan's registered prediction: within ±1–2 points of 56.00 for the strong model.
2. Rerun on the `--generation-config vllm` table, if Jonathan queues it (optional).

Closed: the serving path behind Andy's 52.33 rerun is confirmed (default generation config, repetition_penalty 1.05;
the gap came from the records server clock, see Serving setting above).

## Files

- `results/exp11-mab/select.json`: validation results and declared points.
- `results/exp11-mab/test/report.json`: the full test report, including intervals and contrasts.
- `results/exp11-mab/test/decisions.parquet`: per-task decisions for each router and point.
- `results/exp11-mab/test/job-44887.log`: the test job's log.

## Sensitivity: the pure-temperature table

Same plan and code, on Jonathan's pure-temperature-0 table (`configs/exp11-mab-pureT0.toml`, output `runs/exp11-mab-pureT0`).
The judges were re-trained on this table, so their labels change on the 12 tasks where the strong outcome changed.
It is a second read of the same 150 test tasks, reported as a sensitivity check. The canonical table above stays the primary result.

### Operating points (declared before the sensitivity test read, 2026-11-06)

select.json sha256 `34703bc72b32b900d6ca5a8bf9f360b023ad056a20d4467fe55a1223fab93f5d` (validation only; written by `exp11-mab select` with configs/exp11-mab-pureT0.toml on OMEGA, job 45116).
Always-escalate on val (pure-T0 table): 61.67%.

| Router | A: threshold | A val % | A val units | B: threshold | B val % | B val units |
|---|---|---|---|---|---|---|
| finished | 0.0000 | 61.67 | 1579.7 | 0.0000 | 61.67 | 1579.7 |
| category | 0.2222 | 70.00 | 1077.9 | 0.7778 | 63.33 | 865.6 |
| category x finished | 0.2222 | 70.00 | 1077.9 | 0.8889 | 61.67 | 835.4 |
| zero-shot nano-jev | 0.5852 | 61.67 | 1579.7 | 0.5852 | 61.67 | 1579.7 |
| zero-shot decider-2b | 0.4612 | 61.67 | 1565.5 | 0.4612 | 61.67 | 1565.5 |
| zero-shot decider-e4b | 0.2774 | 61.67 | 1579.7 | 0.2774 | 61.67 | 1579.7 |
| nano-jev | 0.2710 | 70.00 | 991.1 | 0.9026 | 61.67 | 812.8 |
| minilm | 0.2222 | 70.00 | 940.4 | 0.8758 | 61.67 | 835.4 |
| msmarco | 0.2617 | 70.00 | 1384.2 | 0.5457 | 61.67 | 850.5 |
| decider-2b | 0.1956 | 70.00 | 991.1 | 0.8655 | 61.67 | 820.0 |
| decider-e4b | 0.2509 | 70.00 | 991.1 | 0.8420 | 61.67 | 820.0 |

A and B are defined as above. No test label was read before this table was committed.

### Sensitivity results (test, 150 tasks, the second read of the same tasks)

Job 45117, output `results/exp11-mab-pureT0/test/`. The table is in the same order as the canonical one. Always-escalate
is 56.67 on these 150 tasks (54.00 canonical).

| Router (point A) | Canonical: acc / cost / AIQ | Pure-T0: acc / cost / AIQ |
|---|---|---|
| Category policy | 63.33 / 67.7 / 0.524 | 66.00 / 67.7 / 0.540 |
| Category × finished | 63.33 / 67.7 / 0.529 | 66.00 / 67.7 / 0.543 |
| Fine-tuned nano-jev | 63.33 / 59.0 / 0.531 | 66.00 / 60.9 / 0.547 |
| Fine-tuned MiniLM | 59.33 / 52.1 / 0.521 | 65.33 / 59.1 / 0.544 |
| Fine-tuned MS MARCO | 62.67 / 85.5 / 0.522 | 64.67 / 87.4 / 0.544 |
| Fine-tuned Decider 2B | 62.00 / 58.1 / 0.532 | 66.00 / 59.7 / 0.545 |
| Fine-tuned Decider E4B | 62.67 / 56.3 / 0.532 | 66.00 / 62.1 / 0.545 |
| Zero-shot nano-jev | 53.33 / 99.6 / 0.383 | 56.00 / 99.6 / 0.398 |
| Zero-shot Decider 2B | 50.67 / 97.1 / 0.400 | 53.33 / 97.1 / 0.418 |
| Zero-shot Decider E4B | 53.33 / 99.6 / 0.431 | 56.00 / 99.6 / 0.451 |

Paired AIQ contrasts (patient bootstrap, 95% interval; canonical in brackets):

- Nano-jev vs category × finished: +0.0015 (0.0015, 0.007) on the pure-T0 table; canonical −0.0005 to 0.0066, borderline.
- Nano-jev vs MiniLM: +0.0009 to 0.0051 on pure-T0; canonical −0.0008 to 0.0244, borderline.
- Nano-jev vs MS MARCO: −0.0053 to 0.0113 on pure-T0; canonical +0.0013 to 0.016.
- Decider 2B and E4B vs category × finished: positive on both tables; pure-T0 margins are smaller (+0.0005 to 0.0038 and +0.0009 to 0.0045).

What changes: on the pure-temperature table, nano-jev's margins over category × finished and MiniLM are clearly above zero, and
the margin over MS MARCO is no longer. What stays: the fine-tuned judges match or edge the template policy, and the
zero-shot judges trail it. The differences are still around 0.001–0.005 AIQ, so the conclusion stays exploratory.
The canonical table remains the primary result.
