# Experiment 11 (plan): the System One judge on MedAgentBench, by replay

**Status: running, 2026-11-04.** Jonathan's table is committed on his branch (sha256 186545f0…) and was pulled and checked
(`exp11-mab check` passes). Gate 0 was cleared on 2026-11-04 by the project owner, who states the team has SMU cluster
clearance for this data; the MedAgentBench provenance question (synthetic vs de-identified profiles) is still open
for the write-up. Jobs run on OMEGA per [gpu-jobs-exp11.md](gpu-jobs-exp11.md).

## Why

- **A second kind of task.** Our judges have been tested only on multi-hop QA (HotpotQA, 2Wiki, MuSiQue).
  MedAgentBench is an agent benchmark: the model works in a simulated electronic health record by calling a FHIR API.
- **No new model runs.** Jonathan's MedAgentBench campaign (STAIR branch `jonathan`, `lab_package/medagentbench/`)
  recorded a weak and a strong episode for all 300 tasks. A router can be trained and tested by replay.
- **The same decision as our `qpa` judge.** His router reads the task and the weak model's attempt, then delivers
  the weak answer or escalates to the strong model. It is a Qwen2.5-1.5B LoRA trained with GRPO on reward, without
  labels, on 90 tasks.
- **It tests our two novelty claims where labels are scarce.** From the Jev programme:
  - typed decisions route with zero or few labels (H4);
  - typed-decision pretraining helps when labels are scarce (H1).

  MedAgentBench has 90 training tasks.

## What his registration establishes

- **Models, on all 300 tasks:** weak Qwen2.5-7B-Instruct scores 11.67% (35/300); strong Qwen2.5-72B-Instruct-AWQ
  scores 56.00%; the oracle scores 65.00.
  - The weak model is right almost only on task1 (29/30) and partly on task4 (20%).
  - Four-way split of the 300 tasks: both right 8, weak only 27, strong only 160, neither 105.
- **The decision is mostly type-level.** Each category has one instruction template, so a router reading only the
  task text reduces to a per-category policy. Within a category, the signal is in the weak attempt.
- **Accounting (weak-first):** deliver costs the weak episode `cw`; escalate costs `cw + cs`. Units are
  `stair_common.cost.units` at measured tokens.
- **Split:** patient-disjoint (STAIR draft PR #13): train 90 (calib minus `calib_sweep`), val 60 (`calib_sweep`),
  test 150. Each side has 49 patients.
- **His results on test** (each router read once):

  | Policy | Accuracy | Mean units | Share of always-escalate |
  |---|---|---|---|
  | Always deliver | 12.00 | 293.3 | 19% |
  | Always escalate | 54.00 | 1,566.6 | 100% |
  | Oracle | 63.33 | | |
  | His Training 1 router (full weak attempt, θ 0.50) | 61.33 | 847.7 | 54.1% |
  | His Training 3 router (first two rounds, θ 0.50) | 63.33 | 1,276.3 | 81.5% |

## What we need from Jonathan

1. **`training1_table.json`** (300 rows: `tid`, `split`, `view`, `qw`, `qs`, `cw`, `cs`, `weak_rounds`) and its
   sha256 (his registration gives `186545f0…`).
   - `view` is the instruction (plus context) and a summary of the weak attempt: its final answer, capped at 400
     characters, or "did not finish" and its last action.
2. **His routers' per-task test decisions** (Training 1 at OP-A and OP-B, Training 3), so the comparisons are paired.
3. **Optional:**
   - `training3_table.json` (round-2 views), to compare early escalation too;
   - the τ risk scores, for his guardrail tally.

## Gate 0: where it may run

- **Not on this machine.** Windows Application Control blocks torch here, so scoring needs OMEGA or Jonathan's lab.
- **OMEGA forbids personal, confidential or sensitive data.**
  - MedAgentBench is MIT-licensed and public.
  - But its tasks name patients (name, date of birth, MRN) from the benchmark's EHR profiles, and the weak answers
    can quote record values.
  - Its README does not say whether the profiles are synthetic or de-identified real records.
- **Before anything goes to OMEGA:**
  - confirm the profiles' provenance from the MedAgentBench paper (NEJM AI);
  - ask SMU whether the benchmark is acceptable on OMEGA.
- **If it is not:** Jonathan runs our scoring script on his lab machine and returns per-task scores only.

## Design

**Decision and label.** Escalate exactly when the oracle would: the weak model is wrong and the strong one right
(`qw = 0`, `qs = 1`). This is pair A's `large-helps` label. Tasks neither model solves are delivered, at the lower
cost.

**Judges.** Each gives P(escalate) from the `view`.

| Judge | Labels used | Where |
|---|---|---|
| nano-jev v1.0, zero-shot: "Is the proposed answer correct?" (P(escalate) = P(no)) | 0 | CPU |
| Strands Decider 2B and E4B, zero-shot, the same question | 0 | GPU |
| nano-jev fine-tuned, `qpa` item, `large-helps` | 90 | CPU or GPU |
| Strands Decider 2B and E4B fine-tuned, the same item | 90 | GPU |
| Controls for H1: plain MiniLM-L12 and MS MARCO MiniLM, fine-tuned like nano-jev | 90 | CPU or GPU |

- **Fine-tuning:** the 90 train tasks, with the epoch picked on val NLL. Five seeds for the 33M judges, three for the
  deciders.
- **Inputs:** the item is built from the `view` as `qpa` items are. nano-jev reads 512 tokens, so check how many
  views it truncates.

**Baselines:**
- always deliver, always escalate, the oracle;
- **per-category policy:** each category's escalation rate on train, as a score swept like a judge's. This is the
  template lookup his analysis predicts;
- **"finished" rule:** escalate when the weak model did not call FINISH;
- **the two combined:** the escalation rate on train per (category, finished);
- **his Training 1 and Training 3 routers**, paired if he shares their decisions.

**Operating points.** Picked on val before the test read, as his OP-A and OP-B:
- **A:** the most accurate threshold on val, with the cheapest on ties;
- **B:** the cheapest threshold at or above always-escalate's val accuracy.

**Metrics:**
- accuracy and mean units at A and B;
- his reward R' (λ 0.10), for comparability;
- AIQ over the weak-first cost range, our standard;
- each judge's own price, by `budget()` at measured tokens, reported with and without. It is small next to an
  episode's ~1,500 units.

**Intervals:** a paired bootstrap that resamples patients, not tasks (5,000 replicates; patients from `eval_MRN` in
`test_data_v2.json`).

**Test discipline:** one test read per judge, after its operating points are written into this plan, as his
routers had.

## Hypotheses (exploratory: 150 test tasks, a type-level problem)

- **H11a, zero labels:** a zero-shot typed judge closes at least half of the AIQ gap between random routing and the
  fine-tuned judge. One untrained typed question closed 77% of it on HotpotQA (H4a).
- **H11b, 90 labels:** fine-tuned nano-jev matches his Training 1 router at OP-A. Report the paired difference in
  accuracy and in cost, with intervals; no equivalence claim at this size.
- **H11c, pretraining (H1 again):** starting from nano-jev beats starting from plain MiniLM at 90 labels (AIQ).
- **H11d, the attempt adds signal:** fine-tuned judges beat the per-category policy.
  - The room is small: the weak model is partly right only on task4.
  - A tie here is the expected, honest result: the judge then learns the type-level policy, at almost no price.

## Pitfalls

- **150 test tasks and one template per category.** AIQ differences will be small, and intervals wide once
  resampled by patient.
- **Domain shift for nano-jev.** It was trained on prose passages; the views hold FHIR-style answers such as
  `[null]` and JSON actions.
- **The test set has been read twice already,** once by each of his routers. Ours is one more single read, reported
  as such.
- **Two accountings.** His Training 3 prices escalation at round 2. Compare costs and accuracies across his rows, not
  R'.

## Code and compute (written 2026-10-27, before any data)

- **`src/stair/exp11_mab.py`,** with steps `check`, `zeroshot`, `train`, `select` and `test`:
  - **`check`:** checks the table: its sha256, the 300 tasks, 90 / 60 / 150 with 9 / 6 / 15 per category, and no
    patient on both sides. Under stair_common 0.5.0 it also checks that the split equals the committed one.
  - **Items:** the task is the typed question's text and the weak attempt its state, for both questions.
  - **Reuse:** `judge.make_judge`, `judge.fine_tune` (the epoch picked on val NLL) and `evaluate_routing.aiq`.
  - **`select`:** writes points A and B from val, and refuses once the test is read. `test` reads the test once
    and refuses a second run.
- **`exp8_eval.bootstrap`** gains `groups=`: each replicate resamples patients, each bringing all its tasks.
  `exp8_eval.resample` does one draw. Without `groups` the draws are unchanged.
- **Plan:** `configs/exp11-mab.toml`.
  - Judge sets: nano-jev, plain MiniLM and MS MARCO MiniLM, with 5 seeds; the deciders, with 3 seeds.
  - Contrasts for H11a, H11c and H11d.
- **Tests:** `tests/test_exp11_mab.py`, on a synthetic table with fake judges. They cover:
  - the view parsing, the label, the patient check and the items;
  - the baselines and the operating points;
  - his R' and the patient bootstrap;
  - select and the one test read.
- **Job sheet:** [gpu-jobs-exp11.md](gpu-jobs-exp11.md). Four short GPU jobs (zero-shot scoring, then the fine-tunes,
  about 2 h in all) and three CPU jobs (check, select, test), each with a walltime. It gives the fallback on
  Jonathan's machine.
- **Outputs:**
  - `results/exp11-mab/`: scores, decisions and reports. Views and transcripts are committed only if Gate 0 allows it;
  - a write-up in `docs/experiments/`.
- **Credit:** the replay table, the split, the accounting and the baselines are Jonathan's campaign. The write-up
  says so.

## Operating points (declared before the test read, 2026-11-04)

select.json sha256 `37647bb9f9e01165e34662ab30fb523dd5b7dd36b35fb9caffd21749f00061f4` (validation only; written by `exp11-mab select` on OMEGA, job 44886).
Always-escalate on val: 56.67%.

| Router | A: threshold | A val % | A val units | B: threshold | B val % | B val units |
|---|---|---|---|---|---|---|
| finished | 0.0000 | 56.67 | 1579.7 | 0.0000 | 56.67 | 1579.7 |
| category | 0.1111 | 65.00 | 1077.9 | 0.7778 | 60.00 | 865.6 |
| category x finished | 0.1111 | 65.00 | 1077.9 | 0.8889 | 58.33 | 835.4 |
| zero-shot nano-jev | 0.5852 | 56.67 | 1579.7 | 0.5852 | 56.67 | 1579.7 |
| zero-shot decider-2b | 0.4612 | 56.67 | 1565.5 | 0.4612 | 56.67 | 1565.5 |
| zero-shot decider-e4b | 0.2774 | 56.67 | 1579.7 | 0.2774 | 56.67 | 1579.7 |
| nano-jev | 0.3885 | 65.00 | 963.7 | 0.6581 | 56.67 | 802.5 |
| minilm | 0.7080 | 60.00 | 849.5 | 0.7089 | 56.67 | 802.5 |
| msmarco | 0.2346 | 65.00 | 1407.5 | 0.4888 | 56.67 | 802.5 |
| decider-2b | 0.0780 | 65.00 | 992.7 | 0.8809 | 56.67 | 802.6 |
| decider-e4b | 0.0935 | 65.00 | 913.0 | 0.6939 | 56.67 | 800.6 |

A = the most accurate threshold on val, cheapest on ties; B = the cheapest threshold at or above always-escalate's val accuracy (both from `exp11-mab select`). Per-seed points are in the same select.json; the table lists the judge-set rows as select.json gives them. No test label was read before this table was committed.
