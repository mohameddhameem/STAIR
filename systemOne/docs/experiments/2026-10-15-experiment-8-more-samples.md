# Experiment 8, part 2: 20- and 40-sample fused options

> **Metric caveat (added 2026-10-15): do not quote these accuracies or the H3 verdict until re-scoring.**
> The lenient scorer used here counts an answer as correct when the gold answer appears anywhere in it. That
> favours long answers. The fused answers average 220–260 words, against 41 for the 70B. On the 500 test questions:
>
> | Setup | Words | Lenient | Gold in the first 40 words | Gold in the first sentence |
> |---|---|---|---|---|
> | 3B, one answer | 87 | 0.608 | 0.354 | 0.178 |
> | 70B, one answer | 41 | 0.680 | 0.616 | 0.524 |
> | 3B, 10 samples fused | 252 | 0.726 | 0.466 | 0.390 |
> | 3B, 40 samples fused | 223 | 0.756 | 0.508 | 0.422 |
> | No model: nano-jev's top 5 passages pasted as the answer | | **0.798** | | |
>
> - **No model beats every model.** Pasting the passages scores 0.798, above every answer setup.
> - **With length held fixed, the ranking reverses:** the 70B, not the fused 3B, is the best answerer.
> - **Many late hits are wrong.** 167 of the 40-sample cell's 378 lenient hits occur only after its first sentence.
>   In a random sample of 8 such hits, about 5 hedged or gave a different answer first and then quoted a passage
>   that contains the gold. Example: the Peabody Hotel "named after George Peabody", while the gold is "ducks".
> - **It also misses right answers:** "November 22, 1761" against the gold "22 November".
> - **AgentTTS's 74.6 is affected too.** It answers in long form and was re-scored with a lenient scorer, so it is
>   likely inflated the same way.
>
> **What stands and what does not.**
> - **Stands:** retrieval (phase 0), the costs, data integrity, and the evaluation code. A CPU rerun of the
>   five-option evaluation from the tracked judge probabilities reproduces `report.json` exactly.
> - **Not established:** "H3 met (75.6 against 74.6)", "the fused 3B beats the 70B", and the judge's
>   accuracy and gain, because its labels come from the lenient scorer.
>
> Re-scoring with an LLM grader and short-answer extraction is in [gpu-jobs-exp8-rescore.md](../gpu-jobs-exp8-rescore.md).

Follows [the first write-up](2026-10-14-experiment-8.md), where the best fused option was 10 samples and **H3 was not
met** (73.2 lenient at 6.6% of the cost, against AgentTTS's 74.6). This run adds two options, the 3B with 20 and with
40 samples fused (AgentTTS used 40), and reruns the judge with five options. Plan and hypotheses:
[experiment-8-plan.md](../experiment-8-plan.md). Run on omega on 2026-10-14 and 2026-10-15; no earlier answers were
rerun.

**Result:**

- **H3 is met on this test set.** The 3B with 40 samples fused scores **75.6 lenient at 220 units (23.7% of
  strong-everything)**, above AgentTTS's 74.6. The judge, choosing per question, scores **75.4 at 180 units (19.4%)**.
- **The judge saves cost but does not add accuracy over always-40.** Against always-40 its test accuracy is 0.2 points
  lower (interval −1.4 to +1.0) at 40 fewer units. Its AIQ is 0.727 against 0.721 for the best fixed mixture, and that
  difference is not significant (p = 0.10).
- **The 40-sample gain is probably partly a lucky test set.** On calib and router_train, 40 samples scores the same as
  10 (0.724 against 0.722, and 0.720 against 0.718). On test it is 3.0 points higher (0.756 against 0.726).
- **The cheap routers are far behind the judge** (AIQ 0.689–0.691, paired differences of +0.02 to +0.05, all
  significant), as in the first write-up.

## Setup

Unchanged from the first write-up except for the options:

| | |
|---|---|
| Options (cost order) | **3B, one answer** (7.3 units on test); **3B, 10 samples fused** (64.7); **70B 4-bit, one answer** (93.6); **3B, 20 fused** (116.3); **3B, 40 fused** (219.8). All read nano-jev's top-5 passages |
| New cells | `llama3b-s20fuse-top5` (batch 2) and `llama3b-s40fuse-top5` (batch 1), the same 4,500 questions as the other cells |
| Judge | The same 33M nano-jev cross-encoder, state **qp** (question + passages) and **qpa** (adds the 3B's single answer), 3 seeds, 3,000 router_train questions (300 as dev), thresholds picked on 1,000 calib questions, applied once to 500 test questions |
| Label | The cheapest option, in cost order, whose answer is lenient-correct, else the cheapest. On router_train: 3B once 77%, fused-10 14%, 70B 5%, fused-20 2%, fused-40 1% |
| Operating points | **Budget:** the most accurate calib threshold within 25% of strong-everything (232 units). **Near best:** the cheapest threshold within the small drop of the best calib accuracy, as in the first write-up |
| Metric | Lenient accuracy (the gold answer contained in the response), a placeholder for the team's scorer. Strict EM is ≈ 0 because the models answer in sentences |

## Execution

| | |
|---|---|
| Code | `feat/exp8-phase1-results` at `8ce5896` |
| Hardware | omega `a-11`, 1 × NVIDIA L40S per job, two jobs at once |
| Pilot | Job 42611 on 20 test questions: 16.9 s per question (20 samples) and 14.8 s (40), no out-of-memory |
| Answer jobs | 42613 → 42614 → 42615 (20 samples: test, calib, router_train) and 42617 → 42618 → 42619 (40 samples). **All exit 0**, no traceback, retry or out-of-memory in any log. About 21 h wall on the two GPUs. The backups (42616, 42620) never ran |
| Judge job | 42681 (`exp8-judge`): **exit 0 in 17 min**. The chained judge job 42621 was removed by PBS and never ran (its dependency on the unrun backups was not satisfied), so I submitted it again without dependencies. Its backup 42682 never ran |
| Checks | Every new cell has 500 / 1,000 / 3,000 rows with unique ids and no empty predictions. The judge wrote 6 `done.json` (qp and qpa × 3 seeds) |
| Outputs | `results/exp8-phase1/` (the six new answer files, `summary.csv`, job logs 42611, 42613–42615, 42617–42619) and `results/exp8-phase2b/` (judge probabilities, `eval/report.json`, `report-clean.json`, curves, `plan.toml`, `provenance.jsonl`, job log 42681) |

## Results

Test set, 500 questions. Cost is in AgentTTS units, as a share of strong-everything (929). AIQ is over the range of the
options' costs, which grew, so **it is not comparable with the first write-up's 0.695**. Brackets are 95% paired
bootstrap intervals (1,000 resamples).

| Policy | AIQ | Accuracy | Units | % of 929 |
|---|---|---|---|---|
| Always 3B, one answer | | 0.608 | 7.3 | 0.8% |
| Always 3B, 10 samples fused | | 0.726 | 64.7 | 7.0% |
| Always 70B, one answer | | 0.680 | 93.6 | 10.1% |
| Always 3B, 20 samples fused | | 0.736 | 116.3 | 12.5% |
| **Always 3B, 40 samples fused** | | **0.756** | **219.8** | **23.7%** |
| Fixed-policy hull | 0.721 [0.688, 0.757] | | | |
| Entropy threshold | 0.689 [0.653, 0.728] | 0.756 (the budget point is always-40) | 227.2 | 24.5% |
| Logistic regression, before the answer | 0.689 [0.654, 0.730] | 0.756 (always-40) | 219.8 | 23.7% |
| Logistic regression, after the answer | 0.691 [0.654, 0.731] | 0.756 (always-40) | 227.2 | 24.5% |
| **Judge qp, budget point** (t = 0.99) | **0.727** [0.693, 0.762] | **0.754** | **180.3** | **19.4%** |
| Judge qp, near-best point (t = 0.91) | | 0.728 | 61.2 | 6.6% |
| Judge qpa, budget point (t = 0.98) | 0.724 [0.694, 0.763] | 0.740 | 143.0 | 15.4% |
| Judge qpa, near-best point (t = 0.90) | | 0.728 | 66.6 | 7.2% |
| Oracle (cheapest correct option) | 0.838 | 0.850 | 28.2 | 3.0% |
| *AgentTTS recipe, team's re-score* | | *0.746* | *~938* | *~101%* |

The seed spread of the judge's AIQ is small: qp 0.724–0.727, qpa 0.724–0.726. The judge's own cost is 0.4 units per
question, included above.

Paired differences in AIQ (95% interval; p is the share of resamples at or below zero):

| Judge qp against | Difference | p |
|---|---|---|
| Fixed-policy hull | −0.004 to +0.015 | 0.10 (not significant) |
| Entropy threshold | +0.022 to +0.053 | < 0.001 |
| Logistic regression, before | +0.019 to +0.054 | < 0.001 |
| Logistic regression, after | +0.020 to +0.052 | < 0.001 |
| Judge qpa | −0.007 to +0.008 | 0.41 (a tie) |

**What the judge picks.**

| Point | 3B once | 3B ×10 | 70B | 3B ×20 | 3B ×40 |
|---|---|---|---|---|---|
| qp, t = 0.99 | 0 | 0 | 1 (acc 0.0) | 191 (acc 0.84) | 308 (acc 0.70) |
| qp, t = 0.91 | 61 (acc 0.85) | 380 (acc 0.73) | 58 (acc 0.60) | 1 | 0 |

At the budget point it never uses the single answers and splits the questions between 20 and 40 samples. At the
near-best point it behaves as it did in the first write-up (fused-10 for three quarters of the questions) and reaches
72.8 at 61 units, 4 points below always-40 for a quarter of the cost.

On the clean subset (438 questions nano-jev did not see in validation): hull 0.720, entropy 0.695, logistic regression
0.695 and 0.698, judge qp 0.731, qpa 0.727. The judge's budget point is 0.756 at 181 units, always-40 is 0.756.

## Hypotheses

- **H3 (≥ 74.6 lenient at ≤ 25% of strong-everything): met.** Always-40 reaches 75.6 at 23.7% and the judge reaches
  75.4 at 19.4%. This is a statement about this test set. The noise at 500 questions is about ±3.8 points, so neither
  figure is distinguishable from AgentTTS's 74.6, and the claim is "matches at about a fifth of the cost", not "beats".
- **The judge's value on these options is cost.** It does not significantly beat the fixed hull in AIQ. At the budget
  point it matches always-40 and saves 18% of the cost (40 units). The first write-up's AIQ gain over the hull
  was +0.003 to +0.018 (significant); with the added options it is −0.004 to +0.015.
- **RQ8:** a nano-jev retriever with 40 samples of a 3B reader reaches AgentTTS's accuracy at about 24% of its cost, and
  a System One judge cuts this to about 19%.

## Findings

1. **More samples help on test but the evidence for them is thin.** From 10 to 40 samples the test accuracy rises 3.0
   points (0.726 to 0.756). On calib it rises 0.2 points (0.722 to 0.724) and on router_train 0.2 points (0.718 to
   0.720), over 1,000 and 3,000 questions. The calibration split is also where the judge's budget point lands at 0.724.
   The test split may simply favour this cell, so the 75.6 should not be relied on as the cell's true accuracy.
2. **The judge is the cheapest way to hold that accuracy.** On test it gives up 0.2 points of always-40 for 40 fewer
   units. On calib its budget point is also level with always-40 (0.724 at 178 units against 208).
3. **The cheap routers have no room below always-40.** Their budget point is always-40: they escalate from the 3B to
   the most expensive option and none of the trade-off points beats the fixed mixture. This is again partly the
   option set, since they cannot choose the cheaper fused options.
4. **The answer still adds nothing.** qpa ties qp (p = 0.41), as in experiment 7 and the first write-up.
5. **The oracle's headroom grew.** With five options the oracle reaches 85.0 at 28 units (3.0%). The judge is 10 points
   below it in accuracy at six times the cost, so most of the possible gain is not captured.

## Cautions

- **500 test questions.** AIQ intervals are about ±0.035 and accuracy at one point about ±3.8 points. Only the paired
  differences against the entropy and logistic regression routers are significant.
- **The lenient scorer is a placeholder** (see the first write-up). The answers are cached, so only scoring reruns when
  the team's scorer is available.
- **Different recipe from AgentTTS.** Our 40-sample fusion reads the question and top-5 passages, not the whole
  100-candidate pool, and the 70B is 4-bit. The cost comparison is between recipes, not between identical systems.
- **AIQ across experiments.** The 5-option AIQ (0.727) is not comparable with the 3-option one (0.695): the range of
  costs changed. Compare only within a table.
- **Baselines.** The entropy and logistic regression routers use two options (cheapest and most expensive). A cascade
  through the fused options (3B → fused → 40 samples) would be a fairer baseline and was not built.
- **Label counts.** 69 router_train questions carry the 20-sample label and 37 the 40-sample one, so the judge has
  little to learn those options from.

## Next steps

- **Re-score with the team's lenient scorer** on the cached answers (CPU only).
- **Three-way cascade baseline** (3B → fused → 40 samples) for a fair comparison.
- **More test questions or a second seed of the test pools** if a claim above AgentTTS's 74.6 is needed; at 500
  questions it cannot be supported.
- **A 70B with fusion**, which was not tried.

## Reproducing

```bash
# answers: see docs/gpu-jobs-exp8.md section 6 (about 21 h on two GPUs)
qsub -v STEP=exp8-judge scripts/omega.pbs     # 17 min on one L40S
# evaluation alone, on CPU, from results/exp8-phase2b/ probabilities:
uv run exp8-judge eval --plan configs/exp8-judge.toml --exclude runs/exp8/nanojev_validation_ids.txt
```
