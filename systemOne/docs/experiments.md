# Experiments

Headline numbers for every executed run, next to published HotpotQA results. Each run has its
own write-up in [experiments/](experiments/).

All runs use the **HotpotQA distractor** setting (the question plus 10 paragraphs: 2 gold, 8
distractors). Scores are **answer EM and F1** in %, computed with the official
`hotpot_evaluate_v1.py` normalisation, on the full **dev (validation) set of 7,405 questions**.
Cost is GFLOPs per question.

## Our runs

| Run | Date | Method | EM | F1 | GFLOPs | Sent to large | AIQ |
|---|---|---|---|---|---|---|---|
| [1](experiments/2026-10-05-omega-full-run.md) | 2026-10-05 | Always small: Qwen2.5-1.5B, top-2 nano-jev paragraphs | 48.3 | 61.3 | 1,051 | 0% | |
| | | Always large: Qwen2.5-7B, all 10 paragraphs | 53.0 | 66.9 | 22,762 | 100% | |
| | | Router, evidence / large-helps | 52.9 | 66.7 | 8,503 | 34% | 0.663 |
| | | Router, question+evidence / large-helps | 52.5 | 66.0 | 5,933 | 23% | 0.665 |
| | | Small-entropy threshold (baseline) | | | | | 0.662 |
| | | Random routing (baseline) | | | | | 0.641 |
| | | Oracle router (upper bound) | 63.1 | 77.2 | 5,638 | 21% | 0.755 |
| [2](experiments/2026-10-06-cascade-routers.md) | 2026-10-06 | Cascade router, evidence+small / small-fails | 54.1 | 67.6 | 12,657 | 50% | 0.666 |
| | | Cascade router, question+evidence+small / large-helps | 52.9 | 66.5 | 6,340 | 23% | **0.672** |
| [3a](experiments/2026-10-06-experiments-3-5.md) | 2026-10-06 | Always large: 7B on top-3 nano-jev paragraphs | 55.5 | 69.2 | 7,292 | 100% | |
| | | Router, question+evidence+small / large-helps | 54.8 | 68.2 | 5,007 | 54% | 0.669 |
| [3b](experiments/2026-10-06-experiments-3-5.md) | 2026-10-06 | Always large: 7B on top-5 nano-jev paragraphs | 55.8 | **70.0** | 11,598 | 100% | |
| | | Router, question+evidence+small / large-helps | 54.5 | 68.1 | 4,958 | 34% | 0.682 |
| [4](experiments/2026-10-06-experiments-3-5.md) | 2026-10-06 | Always large: Qwen2.5-14B, all 10 paragraphs | 57.1 | **71.9** | 43,906 | 100% | |
| | | Router, question+evidence+small / large-helps | 57.2 | 71.3 | 23,043 | 50% | 0.697 |
| | | Oracle router (1.5B / 14B) | 64.8 | 79.4 | 11,174 | 24% | 0.772 |
| [5](experiments/2026-10-06-experiments-3-5.md) | 2026-10-06 | Always small: Qwen2.5-3B, top-2 nano-jev paragraphs | 49.3 | 61.8 | 2,105 | 0% | |
| | | Router, evidence+small / large-helps (7B on all 10) | 53.6 | 67.1 | 6,384 | 19% | 0.681 |
| [6](experiments/2026-10-06-multi-stage-cascade.md) | 2026-10-06 | Cascade 1.5B (top 2) → 7B (top 3) → 14B (all 10) | **58.4** | **72.6** | 14,243 | 15% to 14B | 0.721 |
| | | Cascade 1.5B (top 2) → 14B (all 10), same procedure | 57.8 | 72.3 | 35,202 | 78% | 0.695 |
| [7](experiments/2026-10-07-system-one-router.md) | 2026-10-07 | System One judge qp (33M, reads question + top-2 passages, decides before the small model), 1.5B / 7B | 52.5 | 66.0 | 4,882 | 18% | **0.673** |
| | | System One judge qpa (adds the small model's answer; decides after it) | 53.0 | 66.5 | 5,995 | 22% | **0.673** |
| | | Zero-shot nano-jev judges (best: "is the answer correct?") | 52.7 | 66.4 | 7,202 | 27% | 0.664 |

Both experts are instruction-tuned and not fine-tuned on HotpotQA, prompted with two
answer-format examples. Routers are logistic regression, with τ = 0.8.
Run 2 reuses run 1's expert outputs. Its cascade routers decide after the small expert has
answered, and the bootstrap shows they are the only routers significantly better than the entropy
baseline. AIQ 95% CIs are about ±0.008.
Runs 3–5 each change one expert of run 1 (rows not shown are unchanged). Reading only nano-jev's
top 3 or 5 paragraphs makes the 7B both better and cheaper. AIQ depends on each run's own expert
pair, so compare it only within a run. The per-run outputs are in [results/](../results/).
Run 6 chains cached experts from runs 1–4 on CPU. Its AIQ is over 1,051–43,906 GFLOPs (1.5B to 14B),
so it is comparable only within run 6.

Run 7 is the final report's central research question ([system-one-router.md](system-one-router.md), report plan
[final-report-plan.md](final-report-plan.md)). Its judges reuse the caches of runs 1, 4 and 5 and cost about 3% of the
small model (32 GFLOPs). The qp judge beats the best logistic regression router that decides before the small model
(AIQ 0.673 against 0.665, paired interval [+0.004, +0.011]) and the entropy baseline, and the qpa judge ties the best
cascade router (0.672). Like every router so far, none reaches always-large's F1 (66.9) at its operating point.

**The Jev novelty programme** ([results](experiments/2026-10-25-jev-novelty.md), [pre-registration](experiments/2026-10-24-jev-preregistration.md))
re-tests run 7's judge on the HotpotQA golden set and on two confirmatory datasets, MuSiQue and 2WikiMultiHopQA.
- **Confirmed on both:**
  - Fine-tuning from nano-jev beats the same 33M MiniLM without decision training.
  - The passages, not the question, decide escalation.
  - The judge is non-inferior to entropy, the LR routers and every cheap weak-model signal, including a
    hidden-state probe.
  - One untrained typed question recovers 77–86% of a trained judge's gain.
- **Fails:**
  - the few-label advantage;
  - the typed head;
  - question-only routing at chance (it beats random on both new datasets);
  - useful risk control.

## Experiment 8: the AgentTTS setting (lenient accuracy, 500 test questions)

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
> Re-scoring with an LLM grader and short-answer extraction is in [gpu-jobs-exp8-rescore.md](gpu-jobs-exp8-rescore.md).

> **Re-scored with an LLM grader on 2026-10-15:** see [experiments/2026-10-15-experiment-8-rescored.md](experiments/2026-10-15-experiment-8-rescored.md). Under the grader the 70B (0.748 test) leads the 40-sample fused 3B (0.632); the judge sends nearly every question to the 70B; H3 is not assessed because AgentTTS's 74.6 is lenient.

Graded accuracy on test (500 questions; grader Qwen2.5-14B on each model's committed short answer):

| Cell | Lenient | Graded | Units | % of 929 |
|---|---|---|---|---|
| 3B, one answer (top 5) | 0.608 | 0.524 | 7.5 | 0.8% |
| 3B, 10 samples fused | 0.726 | 0.588 | 64.9 | 7.0% |
| 3B, 20 samples fused | 0.736 | 0.618 | 116.5 | 12.5% |
| 3B, 40 samples fused | 0.756 | 0.632 | 220.0 | 23.7% |
| 70B (4-bit), one answer (top 5) | 0.680 | **0.748** | 96.0 | 10.3% |
| Judge qp / qpa, budget point (picks the 70B) | | 0.748 | 96.4 / 96.5 | 10.4% |


Different questions, models and metric from runs 1-7: AgentTTS's 100-candidate pools, Llama-3.2-3B and Llama-3.1-70B
reading nano-jev's top 5 passages, scored by lenient accuracy (the gold answer contained in the response). Cost is in
AgentTTS units, as a share of its strong-everything cost (929). Details: [experiments/2026-10-14-experiment-8.md](experiments/2026-10-14-experiment-8.md); retrieval and answer cells: [experiment-8-plan.md](experiment-8-plan.md).

| Policy | Lenient | Units | % of 929 | AIQ |
|---|---|---|---|---|
| 3B, one answer | 0.608 | 7.3 | 0.8% | |
| 70B (4-bit), one answer | 0.680 | 93.6 | 10.1% | |
| 3B, 10 samples fused | 0.726 | 64.7 | 7.0% | |
| 3B, 20 samples fused | 0.736 | 116.3 | 12.5% | |
| 3B, 40 samples fused | **0.756** | 219.8 | 23.7% | |
| Entropy threshold / logistic regression (best operating point) | 0.680 | 92-97 | 10% | 0.652-0.657 |
| **System One judge (qp, decides before any answer), 3 options** | **0.732** | **61.3** | **6.6%** | **0.695** |
| **System One judge (qp), 5 options, budget point** | **0.754** | **180.3** | **19.4%** | **0.727** (own range) |
| Oracle (cheapest correct option), 3 / 5 options | 0.796 / 0.850 | 19.4 / 28.2 | 2.1% / 3.0% | 0.783 / 0.838 |
| *AgentTTS recipe, team's re-score* | *0.746* | *~938* | *~101%* | |

The judge beats the best fixed mixture (AIQ +0.008) and the entropy and logistic regression routers (+0.02 to +0.06),
all significant. With three options it did not reach AgentTTS's 74.6 (73.2 at best; H3 not met). Adding the 20- and 40-sample
fused options ([experiments/2026-10-15-experiment-8-more-samples.md](experiments/2026-10-15-experiment-8-more-samples.md))
meets H3: the 40-sample cell alone scores 75.6 at 23.7% of the cost and the judge 75.4 at 19.4%. The judge's AIQ gain over
the fixed hull is no longer significant (p = 0.10), and the 40-sample gain is seen on test only (calib: 72.4 against 72.2 for 10 samples).
AIQ is over each table's own cost range, so 0.727 and 0.695 are not comparable.

## Published reference points (answer EM / F1)

| System | Type | Split | EM | F1 | Source |
|---|---|---|---|---|---|
| HotpotQA baseline (Yang et al., 2018) | Fine-tuned, trained from scratch | dev | 44.4 | 58.3 | [paper, Table 4][hotpot] |
| HotpotQA baseline (Yang et al., 2018) | Fine-tuned, trained from scratch | test | 45.5 | 59.0 | [paper, Table 4][hotpot] |
| Llama-2-13B | Few-shot chain-of-thought | dev | 30.9 | 45.8 | [Bhuiya et al., 2024, Table 4][distractors] |
| Llama-2-70B | Few-shot chain-of-thought | dev | 54.1 | 67.7 | [Bhuiya et al., 2024][distractors] |
| Mixtral-8x7B-Instruct | Few-shot chain-of-thought | dev | 50.4 | 68.1 | [Bhuiya et al., 2024][distractors] |
| GPT-3.5 | Few-shot chain-of-thought | dev | 63.4 | 77.2 | [Bhuiya et al., 2024][distractors] |
| Longformer | Fine-tuned | dev | 71.5 | 82.1 | [Bhuiya et al., 2024][distractors] |
| Beam Retrieval (2023, leaderboard #1) | Fine-tuned | test | 72.7 | 85.0 | [leaderboard][lb] |
| Human (one crowd worker) | | 1,000 dev/test samples | 83.6 | 91.4 | [paper, Table 8][hotpot] |
| Human upper bound | | 1,000 dev/test samples | 96.8 | 98.8 | [paper, Table 8][hotpot] |

## How the runs compare

- **Against the original baseline:** both experts beat it without any HotpotQA training. The
  1.5B on 2 paragraphs scores +3.0 F1, and the 7B on 10 paragraphs +8.6 F1, over the baseline's
  58.3 dev F1.
- **Against other LLMs:** the 7B reader (66.9 F1) is close to Llama-2-70B and Mixtral-8x7B
  (67.7 and 68.1 F1), which used few-shot chain-of-thought. Reading the top 5 paragraphs (70.0)
  or using the 14B (71.9) moves past them, but both remain 5–7 F1 below GPT-3.5.
- **Against fine-tuned systems:** our best run (the 14B, 71.9) is 10–13 F1 below fine-tuned readers (82–85 F1),
  and far below humans (91.4 F1).
- **The oracle router** (77.2 F1) matches GPT-3.5 using only the 1.5B and 7B experts. With the
  14B, it reaches 79.4 F1. That is the ceiling a better router could reach with these experts.

## Comparability caveats

- **Answer only.** The leaderboard also ranks supporting-fact and joint EM/F1. Our runs do not
  predict supporting facts.
- **Dev, not test.** Our numbers are on the public dev set. The leaderboard uses the hidden test
  set, though dev and test scores were within 1 F1 for the original baseline.
- **Different setups.** The LLM rows use few-shot chain-of-thought. Our experts answer directly,
  and the small expert reads only nano-jev's top 2 paragraphs.
- **Cost is ours only.** Published results do not report cost, so GFLOPs and AIQ compare only
  our own runs.

[hotpot]: https://arxiv.org/abs/1809.09600
[distractors]: https://arxiv.org/abs/2409.05197
[lb]: https://hotpotqa.github.io/
