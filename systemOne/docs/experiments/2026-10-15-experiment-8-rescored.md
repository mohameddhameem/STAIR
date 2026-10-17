# Experiment 8, re-scored with an LLM grader: the 70B leads, and the judge learns to pick it

Re-scores the experiment 8 answers ([first write-up](2026-10-14-experiment-8.md),
[more samples](2026-10-15-experiment-8-more-samples.md)) because the placeholder lenient scorer (gold contained
anywhere in the response) rewards long answers. Job sheet: [gpu-jobs-exp8-rescore.md](../gpu-jobs-exp8-rescore.md).
No answer was regenerated. Run on omega on 2026-10-15.

**Result:**

- **Under the grader the 70B is the most accurate single option, not the 40-sample fused 3B.** On the 500 test
  questions the 70B (one answer) scores **0.748** against **0.632** for the 3B with 40 samples fused. The paired
  difference (fused-40 minus 70B) is **−0.116, 95% interval [−0.154, −0.078]**. The ranking is the same on calib and
  router_train (0.710 against 0.574, and 0.708 against 0.586), so it is not a quirk of the test set.
- **The lenient numbers overstated the fused 3B.** Its lenient 0.756 on test becomes 0.632; the 70B goes the other
  way, 0.680 to 0.748. All five 3B cells lose 0.05–0.14 and the 70B gains 0.07.
- **The judge learns to send (almost) everything to the 70B.** At its operating points it picks the 70B for 500 of
  500 test questions (qp) and 499 of 500 (qpa). Its accuracy there, 0.748 at 96 units (10.3% of strong-everything),
  is always-70B's. Its AIQ ties the fixed-policy hull for qp (0.701, difference [−0.000, +0.003]) and is above it for
  qpa (0.713, [+0.007, +0.020]), but neither has an accuracy operating point better than always-70B.
- **The entropy and logistic regression routers fall behind** (AIQ 0.58–0.61, differences to the judge +0.06 to
  +0.15), as before.
- **H3 is not assessed.** AgentTTS's 74.6 is a lenient number, so it is not comparable with these graded scores.

## Setup

| | |
|---|---|
| Grader | `Qwen/Qwen2.5-14B-Instruct` (revision `cf98f3b3`), bf16, one answer per forward pass. P(CORRECT) is the next-token mass of the CORRECT-prefix against the INCORRECT-prefix tokens; `correct` = P ≥ 0.5 |
| Commit step (version 4) | The model that wrote each response (3B, or the 70B in 4-bit) replays its own conversation and finishes with the cue "So, in as few words as possible, the final answer is:". The grader grades that short answer against the gold. The commit's cost (0.13–0.17 units for the 3B cells, 2.4 for the 70B) is added to each option's cost |
| Why not versions 1–3 | v1 graded identical answers wrong ([results/exp8-rescore/README.md](../../results/exp8-rescore/README.md)); v2 passed 62% of pasted-passage dumps ([v2](../../results/exp8-rescore/v2/README.md)); v3's extractor dropped 24% of correct bare answers ([v3](../../results/exp8-rescore/v3/README.md)) |
| Judge | The same five options, seeds, training and thresholds as the lenient run, with the grader's labels (`configs/exp8-judge-graded.toml`, `metric = "correct"`). 3 seeds, qp and qpa |
| Operating points | Budget (most accurate calib threshold within 25% of 929 units) and near-best, as before |

## Validation

Job 42694 ([results/exp8-rescore/v4/](../../results/exp8-rescore/v4/README.md)); a first attempt, 42693, failed
because the 70B in 4-bit does not fit one GPU, and it was rerun with 2 GPUs without code changes.

| Gate | Result | Required |
|---|---|---|
| Agreement with EM = 1 short answers (n = 269) | 0.996 | ≥ 0.95 |
| Agreement with F1 = 0 short answers (n = 109) | 0.917 | ≥ 0.80 |
| Written verdict agrees with the score | 0.994 | ≥ 0.95 |
| Verdict mass | 0.998 | ≥ 0.90 |
| Commits grounded when graded correct (3B ×1 / 3B ×40 / 70B) | 0.978 / 1.000 / 0.993 | ≥ 0.90 |
| Commits non-empty | 1.000 | ≥ 0.98 |

Of 10 disagreements read, 8 are aliases or formats the grader accepts correctly ("UDC" for "University of the
District of Columbia", "writers" for "writer"); 2 are grader errors ("no" against a long gold string, and an identical
"Benedict of Nursia" at P = 0.38).

## Execution

| | |
|---|---|
| Code | `feat/exp8-rescore` at `c181713` |
| Jobs | 42695 (test, 2 GPUs, exit 0, 30 min), 42700 (calib + router_train, 2 GPUs, exit 0, 3 h 1 min), 42702 (summary, CPU, exit 0), 42712 (graded judge, 1 GPU, exit 0, 17 min). Backups 42696 and 42701 never ran |
| Checks | 19 files (7 + 6 + 6), the right row counts, unique ids, no missing `p_correct`; no traceback or retry in any log; 6 `done.json` in the judge output; `report.json` and `report-clean.json` have `metric = "correct"` |
| Only fix | The 70B commit needs 2 GPUs (resource request, not logic). Two 2-GPU jobs could not run in parallel under the 2-GPU limit, so the test job ran first |

## Results

### Per cell (test, 500 questions)

| Cell | Lenient | **Graded** | Commit EM | Commit F1 | Units (answer + commit) | % of 929 |
|---|---|---|---|---|---|---|
| 3B, top 3, one answer | 0.584 | 0.532 | 0.332 | 0.463 | 6.0 | 0.7% |
| 3B, top 5, one answer | 0.608 | 0.524 | 0.306 | 0.437 | 7.5 | 0.8% |
| 3B, 10 samples fused | 0.726 | 0.588 | 0.166 | 0.339 | 64.9 | 7.0% |
| 3B, 20 samples fused | 0.736 | 0.618 | 0.206 | 0.381 | 116.5 | 12.5% |
| 3B, 40 samples fused | 0.756 | 0.632 | 0.212 | 0.404 | 220.0 | 23.7% |
| 70B, top 3, one answer | 0.628 | 0.700 | 0.514 | 0.656 | 66.0 | 7.1% |
| **70B, top 5, one answer** | 0.680 | **0.748** | **0.546** | **0.692** | 96.0 | 10.3% |

Calib and router_train agree on the order. Graded accuracy on calib / router_train: 3B ×1 0.476 / 0.518,
3B ×10 0.542 / 0.570, 3B ×20 0.559 / 0.572, 3B ×40 0.574 / 0.586, 70B 0.710 / 0.708.

The committed short answer's token overlap with the response (`grounded`) is 0.94–0.996 on test, so the commits are
mostly what the response already said. One example of each disagreement with the lenient scorer (3B ×40, test):

- **lenient = 1, graded = 0** (100 of 500): gold "Joel David Coen"; the response ends "Ethan Jesse Coen is 3 years
  older than Joel David Coen" and commits to "Ethan Jesse Coen". The gold appears in the response but it is not the
  answer.
- **lenient = 0, graded = 1** (38 of 500): gold "Cynthia Ann Stephanie 'Cyndi' Lauper"; the response commits to "Cyndi
  Lauper", a shorter form the strict substring test misses.

### Policies (test, graded accuracy; 5 options)

| Policy | AIQ | Accuracy | Units | % of 929 |
|---|---|---|---|---|
| Fixed-policy hull | 0.701 [0.664, 0.736] | | | |
| Entropy threshold, budget / near-best | 0.612 [0.571, 0.654] | 0.630 / 0.636 | 206 / 164 | 22% / 18% |
| Logistic regression, before the answer | 0.582 [0.546, 0.627] | 0.632 / 0.632 | 220 / 211 | 24% / 23% |
| Logistic regression, after the answer | 0.610 [0.571, 0.652] | 0.630 / 0.634 | 196 / 175 | 21% / 19% |
| Judge qp, budget / near-best | 0.701 [0.664, 0.737] | 0.748 / 0.746 | 96.4 / 96.0 | 10.4% / 10.3% |
| Judge qpa, budget / near-best | 0.713 [0.677, 0.750] | 0.748 / 0.750 | 96.5 / 95.7 | 10.4% / 10.3% |
| Oracle (cheapest correct option) | 0.800 | 0.816 | 30.3 | 3.3% |

Paired differences in AIQ (95% interval; p is the share of resamples at or below zero): qp against the hull
[−0.000, +0.003], p = 0.48 (a tie); qp against the entropy threshold [+0.060, +0.118]; against logistic regression
before [+0.086, +0.146] and after [+0.061, +0.118] (all p < 0.001); qpa against the hull [+0.007, +0.020] (p < 0.001);
qpa against qp [+0.006, +0.019] (qpa better). The clean subset (438 questions) gives the same picture: hull 0.707,
judge qp 0.707, qpa 0.719, entropy 0.610, logistic regression 0.580 and 0.609.

What the judge picks at its operating point: qp, t = 0.93, 70B for 500 of 500; qpa, t = 0.96, 70B for 499 and 3B ×20
for 1. The AIQ differences come from the lower thresholds on the curve, not from the operating point.

## Hypotheses

- **H2 (70B on nano-jev's top-k within 2 points of 72.2):** graded 0.748 on test (top 5) and 0.700 (top 3). The 72.2
  was a lenient number from the team's grid, so the comparison is not valid. On the lenient scale this run gave 68.0
  (phase 1).
- **H3 (≥ 74.6 at ≤ 25% of strong-everything):** AgentTTS's 74.6 is a lenient number and **not comparable** with
  graded scores. A fair comparison needs AgentTTS's own answers graded by the same grader; the team's reproduction has
  them. Neither "met" nor "not met" is claimed. For reference, our best graded policy is 0.748 at 10.3% of the cost.
- **RQ8** is therefore open, pending AgentTTS's answers on the same grader.

## Findings and cautions

1. **The fused-3B result of the earlier write-ups does not survive the grader.** The fused cells gain by listing the
   gold anywhere in long responses. With the committed short answer, 40 samples add 0.044 over 10 (0.632 against
   0.588), against 0.030 lenient, and the cell is still 0.116 below the 70B.
2. **The judge still beats the entropy and logistic regression routers, but that is no longer the story.** It beats the entropy
   and logistic regression routers by 0.06–0.15 AIQ, mostly because those two escalate to the most expensive option,
   which is now the weaker 40-sample cell.
3. **With the 70B this strong and no cheaper option close to it, the judge has little to choose.** 3B ×1 scores 0.524,
   the 70B 0.748, and the oracle 0.816 at 30 units shows headroom, but the judge cannot find it from the question and
   passages.
4. **500 test questions** give about ±4 points; the paired interval on the key difference excludes zero comfortably.
5. **The grader makes errors**: 99.6% agreement on EM = 1 answers and 91.7% on F1 = 0, with two clear errors among 10
   disagreements read. The grader is one model; a second grader would test that.
6. **The commit step is a design choice**: it costs 0.13–2.4 units and uses each model's own conversation. A different
   cue could change small differences between cells.
7. **The 70B is 4-bit** (NF4), as in phase 1.

## Next steps

- **Grade AgentTTS's own answers** with the same grader, to answer H3.
- **A cheaper strong option**: 70B top 3 (0.700 at 66 units) is a middle option the judge has not been trained on.
- **A three-way cascade baseline** for a fair comparison with the entropy and logistic regression routers.

## Reproducing

```bash
qsub -l select=1:ncpus=8:ngpus=2:mem=96gb -l walltime=01:30:00 -v STEP=exp8-rescore-validate scripts/omega.pbs
qsub -l select=1:ncpus=8:ngpus=2:mem=96gb -l walltime=04:00:00 -v STEP=exp8-rescore,SPLITS=test scripts/omega.pbs
qsub -l select=1:ncpus=8:ngpus=2:mem=96gb -l walltime=16:00:00 -v "STEP=exp8-rescore,SPLITS=calib router_train" scripts/omega.pbs
qsub -l walltime=00:15:00 -v STEP=exp8-rescore-summary scripts/omega-cpu.pbs
qsub -l select=1:ncpus=8:ngpus=1:mem=32gb -l walltime=01:00:00 -v STEP=exp8-judge-graded scripts/omega.pbs
```
