# Experiment 8: nano-jev retrieval + a System One judge on the AgentTTS setting

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

Plan and hypotheses: [experiment-8-plan.md](../experiment-8-plan.md). Phase 0 (retrieval) and phase 1 (the answer
cells) are written up there; this document covers phase 2 (the judge) and phase 3 (the evaluation), run on omega on
2026-10-14. It reuses phase 1's cached answers, so **no LLM was rerun**.

**Result:**

- **A 33M judge that picks the answer option per question beats every other policy we built on the same options.**
  Its AIQ is 0.695 against 0.687 for the best fixed mixture, 0.657 for the entropy threshold and 0.652–0.657 for
  logistic regression. All paired differences are significant.
- **It does not reach AgentTTS's accuracy.** At its operating point it scores 73.2 lenient at 61 cost units (6.6% of
  strong-everything), against 74.6 for AgentTTS at about 101%. The curve's best point is 73.2, so **H3 is not met**,
  though the gap is inside the ±3.9 points of noise of 500 questions.
- **Most of its gain is a cost saving.** It sends 88–95% of questions to the fused 3B, which is 72.6 on its own. Against
  always-fused it adds 0.6 points and saves 3 cost units.

## Setup

| | |
|---|---|
| Question | *Which answer option should be used for this question: {question}* |
| Options | Ordered by cost: **3B, one answer** (6.8 units); **3B, 10 samples fused** (61.8); **70B, one answer** (88.2). All three read nano-jev's top-5 passages |
| Judge | nano-jev v1.0 (33M cross-encoder, revision `ffee345e`), as in experiment 7. Its scores are softmaxed over the three options |
| State | **qp**: question + top-5 passages (decides before any answer). **qpa**: also the 3B's single answer |
| Label | The cheapest option whose answer is lenient-correct, or the cheapest option when none is. On router_train: 81% cheapest, 14% fused, 5% 70B |
| Training | 3,000 router_train questions (2,700 train, 300 dev to pick the epoch), lr 5e-5, up to 3 epochs, batch 16, 3 seeds. A policy's probabilities are the mean of the 3 seeds |
| Policy | The cheapest option whose cumulative probability reaches a threshold t. Sweeping t from 0 to 1 gives a cost-accuracy curve |
| Operating point | Chosen on the 1,000 calib questions: the most accurate threshold within 25% of strong-everything (232 units). Applied once to the 500 test questions |
| Judge cost | Counted: 0.25 units per question for the 3-seed ensemble (0.03% of strong-everything), added to every judge point |
| Baselines | The fixed-policy hull (always one option, and random mixtures), the oracle, the entropy threshold on the 3B's answer, logistic regression on nano-jev's scores before the answer, and the same with the answer's entropy added (after) |
| Statistics | Paired bootstrap, 1,000 resamples of the test questions. The clean subset drops the 62 test questions nano-jev saw in its own validation (438 left) |

## Execution

| | |
|---|---|
| Code | `feat/exp8-phase1-results` at `fdc91bd` (the judge code is from `c05f15f`) |
| Hardware | omega `a-11`, 1 × NVIDIA L40S |
| Job | PBS 42609 (`qsub -v STEP=exp8-judge scripts/omega.pbs`), backup 42610. **Exit 0 on the first attempt in 6 min 38 s.** No errors, retries or warnings; the backup never ran |
| Training | 54 s per seed. Best dev NLL 0.550–0.595 (best epoch 1 or 2), against 0.60 for predicting the label prior, so the judges learn a ranking rather than calibration |
| Scoring | 3.9 ms per question |
| Outputs | `results/exp8-phase2/`: `eval/report.json`, `report-clean.json`, `curves*.csv`, every seed's `done.json` and calib/test probabilities, `plan.toml`, `provenance.jsonl`, the job log |

## Results

Test set, 500 questions. Lenient accuracy is the primary metric. Cost is in AgentTTS units (one unit is one 3B pass),
and the share is of strong-everything (929 units). AIQ is the area under the non-decreasing upper hull over
7–94 units, and the brackets are 95% bootstrap intervals.

| Policy | AIQ | Operating point: accuracy | Units | % of 929 |
|---|---|---|---|---|
| Always 3B, one answer | | 0.608 | 7.3 | 0.8% |
| Always 70B, one answer | | 0.680 | 93.6 | 10.1% |
| Always 3B, 10 samples fused | | 0.726 | 64.7 | 7.0% |
| Fixed-policy hull | 0.687 [0.649, 0.724] | | | |
| Entropy threshold | 0.657 [0.620, 0.696] | 0.680 | 96.6 | 10.4% |
| Logistic regression, before the answer | 0.652 [0.616, 0.692] | 0.680 | 93.6 | 10.1% |
| Logistic regression, after the answer | 0.657 [0.620, 0.695] | 0.678 | 91.7 | 9.9% |
| **Judge qp** (before) | **0.695** [0.659, 0.733] | **0.732** | **61.3** | **6.6%** |
| **Judge qpa** (after) | **0.695** [0.660, 0.734] | **0.732** | **65.4** | **7.0%** |
| Oracle (cheapest correct option) | 0.783 | 0.796 | 19.4 | 2.1% |
| *AgentTTS recipe, team's re-score* | | *0.746* | *~938* | *~101%* |

Paired differences in AIQ (95% interval; p is the share of resamples at or below zero):

| Judge qp against | Difference | p |
|---|---|---|
| Fixed-policy hull | +0.003 to +0.018 | 0.003 |
| Entropy threshold | +0.018 to +0.061 | < 0.001 |
| Logistic regression, before | +0.020 to +0.066 | 0.001 |
| Logistic regression, after | +0.019 to +0.063 | < 0.001 |
| Judge qpa | −0.004 to +0.007 | 0.46 (a tie) |

On the clean subset (438 questions) the picture is the same: judge qp 0.696 against the hull 0.687, the entropy
threshold 0.669 and logistic regression 0.660 and 0.667; operating point 0.726 at 61 units.

**What the judge does at its operating point (t = 0.90).**

| Judge | 3B once | 3B fused | 70B once |
|---|---|---|---|
| qp | 8% of questions, accuracy 0.89 | 88%, accuracy 0.72 | 4%, accuracy 0.60 |
| qpa | 1%, accuracy 0.67 | 95%, accuracy 0.74 | 4%, accuracy 0.62 |

## Hypotheses

- **H3 (reach 74.6 lenient at ≤ 25% of strong-everything): not met.** The judge reaches 73.2 at 6.6%, and 73.2 is the
  best accuracy anywhere on its curve (no point at 74.6 or above). The cost half of the claim is met with a wide
  margin. The accuracy shortfall is 1.4 points, which is well inside the roughly ±3.9 points of sampling noise at 500
  questions, so the data neither confirms nor rules out parity with AgentTTS.
- **H2 (70B on nano-jev's top-k within 2 points of 72.2):** 68.0, 4.2 below (phase 1).
- **H1 (retrieval):** supported in phase 0.
- **RQ8:** at 7% of the cost, we are within noise of AgentTTS's accuracy, but did not match it.

## Findings

1. **The per-question choice is worth more than any single cell, but not by a lot.** The judge's AIQ is 0.008 above the
   hull of fixed policies (significant), and its operating point beats always-fused by 0.6 points while saving 5% of the
   cost. Most of the distance from always-fused to the oracle (0.796 at 19 units) is unclaimed.
2. **The 70B is hardly used.** With the 3B fused over 10 samples at 72.6, the 70B (68.0, higher cost) is dominated on
   accuracy and cost. The judge picks it for 4% of questions, where it is right only 60–62% of the time. The useful
   contrast is the 3B once against the 3B fused.
3. **The answer adds nothing, again.** qpa ties qp (p = 0.46), as in experiment 7: the signal is in the question and
   the passages.
4. **A trained judge beats the cheap signals by a wide margin on these options.** The entropy and logistic regression
   routers choose between the cheapest and the strongest option, and the strongest here is the 70B, so they spend most
   of their budget on the wrong option. Their operating points (0.68 at 92–97 units) are no better than always-70B.
   That is a fair test of what each policy can do with these options, but it also means the comparison favours a
   judge that knows about the fused option.

## Cautions

- **500 test questions.** The AIQ intervals are about ±0.037, and accuracy at one operating point carries about
  ±3.9 points. The paired differences in the second table are significant; absolute comparisons with AgentTTS are not.
- **The lenient scorer is a placeholder** (the gold answer contained in the response) until the team's scorer is shared.
  Strict EM is about 0 because the models answer in full sentences. The answers are cached, so only scoring reruns.
- **10 samples, not 40,** and a 4-bit 70B. AgentTTS's 74.6 comes from a recipe that costs about 101% of
  strong-everything, and its fusion call reads the whole 100-candidate prompt. Ours is a cheaper, different recipe.
- **Few training questions for the rare labels:** 151 questions carry the 70B label, so the judge has little to learn
  that option from. Its choice of the 70B is the least reliable part.
- **The comparison with logistic regression is not like for like.** Those routers see two options (cheapest and
  strongest), and the judge sees three. A cascade through the fused option would be a fairer baseline.
- **Not tested:** other τ-like thresholds for the label, other options (more samples, the 70B with fusion), and other
  question sets.

## Next steps

- **More samples as another option** (20 or 40, as AgentTTS used): the fused cell is where the accuracy comes from,
  and the judge can only help if there is something to choose.
- **A three-way cascade baseline** (3B → fused → 70B) trained with the existing cascade code, for a fair comparison.
- **The team's lenient scorer**, then rerun the evaluation on CPU.

## Reproducing

```bash
qsub -v STEP=exp8-judge scripts/omega.pbs     # 7 min on one L40S, after phase 1's answers exist
# or, with the judge probabilities in results/exp8-phase2/, the evaluation alone on CPU:
uv run exp8-judge eval --plan configs/exp8-judge.toml --exclude runs/exp8/nanojev_validation_ids.txt
```
