# Experiments 7, 8 and 9 redone on the team's common standard

**Complete.** Experiments 7, 8 and 9 have all been run and evaluated.

> **Superseded as the HotpotQA headline (2026-10-21).** The team now follows Jonathan's Stage-Level Routing setup:
> the `slr-*` split and Table 5 prices ([golden-set write-up](2026-10-21-slr-standard.md)). This page stays as the
> AgentTTS-slice history, the comparison with the paper on its own 500 questions.

These are the original experiments, rerun with the STAIR `common/` package so that our numbers can be compared
directly with the rest of the team's.
- The runs used `stair_common` at STAIR `main` `052ebe4`.
- The final re-evaluations (jobs 42847, 42848 and 42849) used 0.2.0 (`c906e3d`). Its scoring and cost functions are the same;
  only docstrings and the schema check changed.

Original reports:
- [experiment 7](2026-10-07-system-one-router.md);
- [experiment 8](2026-10-14-experiment-8.md);
- [experiment 9](2026-10-17-experiment-9.md).

Job sheet: [gpu-jobs-common.md](../gpu-jobs-common.md). Run on OMEGA on 2026-10-17 and 2026-10-18. Outputs:
[results/common](../../results/common/README.md).

**Result:**

- **Decider judges that read the small model's answer (qpa) are the only routers that beat entropy and the nano-jev
  judges, even after paying for themselves.**
  - One fine-tuned judge (one seed, one forward pass) scores AIQ **0.724** with its own cost. That holds for the E4B
    (0.719–0.729 across seeds) and the 2B (0.722–0.726).
  - With cost, all six single-seed qpa deciders beat the entropy threshold, by **[+0.001, +0.041]** to
    **[+0.009, +0.051]**. They also beat both nano-jev judges, by about +0.03.
  - They lead the best LR cascade router (0.706) by about +0.02. But five of the six intervals touch zero (the 2B's
    seed 0 is [−0.003, +0.036]), so on 500 questions that comparison is a tie.
  - The same order holds on the clean subset and with F1 labels.
- **Averaging three seeds doesn't pay for itself.** The mean of three judges scores higher before cost (E4B 0.740,
  2B 0.734), but it runs all three models. With three passes, the E4B mean falls to 0.708, which ties entropy. One
  judge is the better deployment.
- **The 33M nano-jev judges no longer beat the baselines.**
  - qp scores 0.696 and qpa 0.698, which ties entropy (0.697) and the LR routers.
  - In the original experiment 7, qp was +0.011 over entropy on 7,405 questions. On 500 questions that margin is
    inside the noise.
- **Without the answer, the deciders tie everything.**
  - The qp deciders score 0.698–0.702 as one judge with its cost.
  - The zero-shot E4B (0.716 with its cost) beats entropy and nano-jev, but ties the LR cascade router
    ([−0.005, +0.028]).
- **Experiment 8, with five answer options (lenient):**
  - **The answer options:** 3B ×40 fused is the most accurate single option: 0.746, equal to AgentTTS, at 71 units.
    It is cheaper and better than the 70B (0.670 at 94 units).
  - **Against AgentTTS:** its recipe costs at least 302 units under Eq. 8 (97% of strong-everything), so ×40 fused
    reaches its accuracy at under a quarter of its cost.
  - **nano-jev qpa:** it matches the best mix of fixed options (0.711 vs 0.716). It beats entropy by
    **[+0.003, +0.039]** and the LR router after the 3B answers by [+0.001, +0.038]. It ties the LR router before
    them ([−0.002, +0.037]).
  - **On the LLM grade:** the 70B is the best option (0.748). There, the judge beats the fixed mix but only ties
    entropy.
- **The deciders don't help with five options.**
  - Before cost, a fine-tuned qpa decider is level with nano-jev qpa (2B 0.661–0.666, nano-jev 0.658–0.662, graded).
  - Its 2.6–5.1 units a question then leave one judge tied with entropy, and the mean of three seeds below it.
- **One epoch is enough** for experiments 7 and 9. Every judge's best validation loss is at epoch 1, except
  nano-jev qa at epoch 2 (see [Loss curves](#loss-curves)).

**For the report:**
- A typed-decision judge of about 2B parameters, reading the small model's answer, beats uncertainty thresholds and
  the 33M judge, even after paying for itself.
- Against a feature-based LR cascade router, its +0.02 is not detectable on 500 questions.
- With more answer options than two, the cheap 33M judge does as well as the deciders, at a twentieth of their cost
  or less.

**Correction (2026-10-18).** The first draft of this page made two errors:
- **Judge cost:** it charged the decider judges' three-seed mean the cost of one pass, although it runs three models.
  That made its "with own cost" figures (0.729, and [+0.003, +0.040] over the LR cascade router) too high. The
  figures above charge the three-seed mean three passes, and add each single seed at one pass.
- **Experiment 8 baselines:** they escalated to the most expensive option, which under Eq. 8 is the 70B. They now
  escalate to the most accurate option on calib.

**Correction (2026-10-21): experiment 8's qpa judges now pay for the answer they read.**
- **The error:** a qpa judge reads the cheapest option's answer, so that option runs on every question. Its cost
  counted only the option it chose. An escalated question was undercharged by the 3B's answer (7.3 units). The
  entropy baseline already paid both.
- **The fix:** `a105286`, re-evaluated by job 43087, with no retraining.
- **What moved:**
  - Every qpa row of experiment 8 is lower. Lenient nano-jev qpa went from 0.714 to 0.709; its graded score from
    0.664 to 0.653.
  - Its lead over the LR router before the 3B answers is now a tie.
  - Its operating point costs 54 units, not 46.
  - qp judges read no answer, so they are unchanged. Experiments 7 and 9 already charged both calls.
- **The earlier reports:** kept on OMEGA in `runs/exp8-common-before-qpa-fix`.

**Correction (2026-10-22): experiment 8's ranking is paid once.**
- **The error:**
  - Every option's cost includes nano-jev's ranking of the 100-passage pool (about 2.9 units), which runs once per
    question.
  - A router that reads the 3B answer and then escalates (entropy, LR after the answers, the qpa judges) was charged
    for that ranking twice.
  - This was in the entropy and LR baselines from the start, and in the qpa judges since the 2026-10-21 fix.
- **The fix:** `dd5b76f`, re-evaluated by job 43116, with no retraining.
- **What moved:** the "after" routers are a little cheaper, so their AIQs rise by about 0.002–0.004.
  - Lenient nano-jev qpa: 0.709 → 0.711. Entropy: 0.691 → 0.694.
  - Graded nano-jev qpa: 0.653 → 0.657.
  - The qpa operating point: 54 → 51 units.
  - The conclusions stand. The figures below are the corrected ones.
- **The earlier reports:** kept on OMEGA in `runs/before-ranking-fix/exp8-common`.

## What changed from the original runs

| | Original (experiments 7 and 9) | Common standard |
|---|---|---|
| Test set | HotpotQA dev, all 7,405 questions | HotpotQA dev[500:1000], 500 questions (AgentTTS's test slice, `stair_common.splits`) |
| Calibration (picks the operating point) | Our own calib split | dev[0:500], the team's val split |
| Judge training data | Our own split | The team's train draw from the HotpotQA train file |
| Correctness labels and headline metric | F1 | Lenient accuracy (`stair_common.score`); F1 rerun as a check |
| Cost unit | GFLOPs | AgentTTS units (`stair_common.cost`, 1 unit = 1,152 GFLOPs). Samples from one prompt pay it once (AgentTTS Eq. 8), which matters only for experiment 8's sampled cells |
| Judge cost | One pass | One pass per model run: a seed alone pays one, the three-seed mean pays three |
| Bootstrap | Paired, AIQ without the judge's cost | Paired, also **with each router's own cost** |
| Models | Pair A: Qwen2.5-1.5B (small) / 7B (large), nano-jev v1.0 retriever | Unchanged |

Baselines on the 500 test questions (lenient):
- always-small scores **0.654** at 0.91 units a question;
- always-large scores **0.674** at 19.5 units;
- the oracle scores 0.782 at 3.2 units, sending 12.8% of questions to the large model.

## Experiments 7 and 9: AIQ, 500 test questions, lenient, AgentTTS units

AIQ is the area under the accuracy-vs-cost curve. The columns:
- **AIQ:** without the judge's cost. For fine-tuned judges it is the mean of the 3 seeds' scores.
- **One judge, with its cost:** each seed's judge on its own, charged one pass. The figure is the mean over the
  seeds, with their range in brackets. The next table gives each seed's interval.
- **Mean of 3 seeds, with its cost:** charged three passes. Its 95% interval comes from the with-cost bootstrap
  (1,000 resamples).

| Router | Decides | AIQ [95% CI] | One judge, with its cost | Mean of 3 seeds, with its cost [95% CI] | Original (7,405 q, F1, GFLOPs) |
|---|---|---|---|---|---|
| Oracle | | 0.774 [0.738, 0.809] | | | 0.755 |
| Decider E4B fine-tuned, **qpa** | after | **0.740** [0.703, 0.777] | **0.724** (0.719–0.729) | 0.708 [0.673, 0.744] | 0.713 |
| Decider 2B fine-tuned, **qpa** | after | **0.734** [0.699, 0.770] | **0.724** (0.722–0.726) | 0.718 [0.683, 0.755] | 0.708 |
| Decider E4B zero-shot, `grounded` | after | 0.726 [0.693, 0.763] | 0.716 [0.683, 0.754] | | 0.692 |
| Decider E4B zero-shot, `correct` | after | 0.725 [0.695, 0.763] | 0.716 [0.685, 0.753] | | 0.693 |
| Decider 2B zero-shot, `correct` | after | 0.715 [0.679, 0.751] | 0.711 [0.675, 0.746] | | 0.675 |
| nano-jev zero-shot, `correct` | after | 0.707 [0.674, 0.746] | 0.707 | | 0.664 |
| Best LR cascade router (question+evidence+small) | after | 0.706 [0.672, 0.745] | | | 0.672 |
| Decider 2B fine-tuned, qp | before | 0.706 [0.672, 0.743] | 0.702 (0.697–0.706) | 0.696 [0.662, 0.732] | 0.680 |
| Decider E4B fine-tuned, qp | before | 0.706 [0.673, 0.748] | 0.698 (0.693–0.702) | 0.687 [0.653, 0.725] | 0.681 |
| Best LR router before the small model (question+evidence) | before | 0.700 [0.667, 0.740] | | | 0.665 |
| nano-jev qpa (experiment 7) | after | 0.698 [0.667, 0.738] | 0.696 (0.693–0.700) | 0.698 | 0.673 |
| Small-entropy threshold | after | 0.697 [0.661, 0.735] | | | 0.662 |
| nano-jev qp (experiment 7) | before | 0.696 [0.663, 0.736] | 0.693 (0.692–0.695) | 0.696 | 0.673 |
| nano-jev qa | after | 0.681 [0.647, 0.719] | | | 0.649 |
| nano-jev q | before | 0.676 [0.641, 0.711] | | | 0.647 |
| Random | | 0.664 [0.630, 0.699] | | | 0.641 |

The judges' own cost, in units a question for one pass:
- nano-jev, 0.03;
- the 2B decider, about 1.1;
- the E4B decider, about 2.1.

The questions themselves cost 0.91 (always-small) to 19.5 (always-large) units, so the deciders' cost is large next to
the small model's. nano-jev's cost changes AIQ by less than 0.001.

Absolute AIQs are not comparable between the last column and the rest: the test set, labels and cost unit all
changed. The **order** is what carries over:
- the deciders' qpa judges come first, as in experiment 9;
- nano-jev drops from "best cheap router" to level with entropy.

### Paired differences with each router's own cost (95% interval, 1,000 resamples)

For one judge, the three intervals are seeds 0, 1 and 2.

| Pair | Difference | Verdict |
|---|---|---|
| E4B qpa, one judge − entropy | [+0.006, +0.043], [+0.009, +0.051], [+0.001, +0.041] | better |
| E4B qpa, one judge − best LR cascade router | [−0.004, +0.033], [+0.001, +0.040], [−0.007, +0.031] | tie (one seed better) |
| E4B qpa, one judge − nano-jev qp | [+0.006, +0.040], [+0.012, +0.047], [+0.004, +0.038] | better |
| E4B qpa, one judge − nano-jev qpa | [+0.005, +0.038], [+0.010, +0.045], [+0.002, +0.035] | better |
| 2B qpa, one judge − entropy | [+0.003, +0.049], [+0.008, +0.049], [+0.005, +0.048] | better |
| 2B qpa, one judge − best LR cascade router | [−0.003, +0.036], [−0.002, +0.039], [−0.003, +0.037] | tie |
| 2B qpa, one judge − nano-jev qpa | [+0.006, +0.041], [+0.008, +0.044], [+0.005, +0.044] | better |
| E4B qpa, mean of 3 seeds − entropy | [−0.009, +0.028] | tie |
| 2B qpa, mean of 3 seeds − entropy | [+0.002, +0.042] | better |
| 2B qpa, mean of 3 seeds − best LR cascade router | [−0.007, +0.032] | tie |
| E4B zero-shot `correct` − entropy | [+0.001, +0.041] | better, just |
| E4B zero-shot `correct` − best LR cascade router | [−0.005, +0.028] | tie |
| 2B / E4B qp, one judge − entropy, LR routers, nano-jev | all straddle 0 | tie |
| E4B qp, mean of 3 seeds − best LR cascade router | [−0.035, −0.001] | worse |
| nano-jev qp − entropy | [−0.021, +0.022] | tie |
| nano-jev qpa − entropy | [−0.019, +0.024] | tie |
| nano-jev qp − best LR router before the small model | [−0.016, +0.009] | tie |
| nano-jev q − entropy | [−0.041, −0.004] | worse |

Checks:
- **Clean subset** (438 questions without nano-jev's own validation questions):
  - one judge with its cost: E4B qpa 0.721, 2B qpa 0.723;
  - best LR cascade router 0.700, nano-jev qpa 0.698, entropy 0.696.
  - Same order.
- **F1 labels** (`runs/common-judges-f1`):
  - 2B qpa: 0.720 as one judge with its cost; the mean of 3 seeds is 0.727 before cost and 0.713 with it;
  - best LR cascade router 0.702, nano-jev qp 0.701, entropy 0.697.
  - Same order. The E4B wasn't retrained on F1 labels.

## Loss curves

`results/common/loss-curves.png` and `training-history.csv` hold 30 histories (10 judges × 3 seeds), in the
`stair_common.training` layout.

- **Best epoch:** the best validation epoch is 1 for every judge except nano-jev qa (2). After it, training loss
  keeps falling while validation loss rises: overfitting.
- **Saved checkpoint:** the best validation epoch, so the reported results come from epoch 1.
- **Lowest validation NLL:** E4B qpa about 0.28–0.30, 2B qpa about 0.30, nano-jev qpa about 0.40. This matches the
  routing order.
- **Seeds:** they agree closely, so the stopping epoch is stable.

For the team: stop early on validation loss rather than fixing the epoch count. Here, one epoch would have saved most
of the 4 hours the E4B job took.

## Experiment 8: choosing among five answer options

- **Setup:**
  - 1,000 router_train questions (a third of the original 3,000) and 500 val questions (calib).
  - The test set is dev[500:1000], as in the original.
  - Answers come from jobs 42818–42829.
- **The judge:** a System One judge reads the question and nano-jev's top 5 passages (qp), plus the 3B's single answer
  (qpa). It outputs a probability for each option.
  - The router takes the cheapest option whose cumulative probability reaches a threshold. Sweeping the threshold
    gives the curve.
  - The training label is the cheapest option that answered correctly.
- **The baselines** route between two options: the 3B's single answer, and the most accurate option on calib.
  - On lenient that is 3B ×40 fused; on the LLM grade it is the 70B (`baseline_options` in each report).
  - **Entropy** escalates on the 3B's answer entropy.
  - **The LR routers** are logistic regressions on nano-jev's evidence features, trained on router_train. One uses
    those features alone (before the 3B answers); the other adds the 3B's entropy (after).

**Cost rule corrected (2026-10-18).**
- **The rule:** AgentTTS counts S samples from one prompt as prompt + S × decode (App. A.2, Eq. 8: the samples share
  the prompt's prefill). Our answer files counted the prompt once per sample.
- **The fix:** `pool-answer recost` (job 42842) rewrote them from their stored token counts. It made no new model
  calls, and kept the old figures as `units_prompt_per_sample`.

Mean AgentTTS units a question on test, each including nano-jev's ranking:

| Option | Prompt per sample (old) | Eq. 8 (now) |
|---|---|---|
| 3B, one answer (top 5) | 7.3 | 7.3 |
| 3B, 10 samples fused | 64.7 | 30.4 |
| 3B, 20 samples fused | 116.3 | 43.9 |
| 3B, 40 samples fused | 219.8 | 71.1 |
| 70B, one answer (top 5) | 93.6 | 93.6 |

- **Reordering:** the fused cells now all cost less than the 70B, so the options are reordered (3B → ×10 → ×20 →
  ×40 → 70B).
- **Retraining:** the judges' labels depend on that order, so they were retrained (42844–42846). The judge trained on
  the old order (42833) is kept in `judge-5opt-prompt-per-sample`. It scored 0.715, against 0.713 for the fixed
  options, so the correction changes the options' costs, not the conclusion.
- **Original reports:** those of 2026-10-14 and 15 used the old rule, so their fused-cell costs are overstated by the
  same factors.

**AgentTTS's own cost, re-derived under Eq. 8 (2026-10-18).**
- **Two scales.** AgentTTS's Table 5 prices (72B retrieval 814, 70B answer 115, 3B answer 3) are its normalised
  budget (Eq. 3–4): the number of 3B samples, sharing one 128-token prompt, that cost the same FLOPs.
  - For a call of u `stair_common` units, that budget is 3u − 2.
  - At the paper's Table 4 token lengths, 3u − 2 reproduces all twelve HotpotQA prices in Table 5, and its 10-sample
    price (Table 7).
  - The team slide's 929 (strong-everything) and 937 (the recipe) are on that scale. Our measured costs are in u.
- **The recipe,** at AgentTTS's Table 4 lengths:

| Call | Units (u) | Table 5 scale (3u − 2) | Team slide |
|---|---|---|---|
| Qwen2.5-72B retrieval (2,048 prompt + 128 generated tokens) | 272.0 | 814 | 814 |
| 40 Llama-3.2-3B answers from one prompt (256 + 40 × 64) | 14.7 | 42 | 120 (40 × 3) |
| One 3B fusion call reading the 40 answers (256 + 2,560 + 64) | 15.0 | 43 | 3 |
| **AgentTTS recipe** | **301.7** | 899 | 937 |
| Strong-everything: 72B retrieval + one 70B answer (256 + 64) | 310.9 | 929 | 929 |

- **Result:** under Eq. 8 the recipe costs **302 units, 97% of strong-everything (311)**, not the slide's ~101%. The
  slide's reading still holds: AgentTTS re-spends the strong budget rather than saving it.
  - The configs now carry `strong_units = 311` and `agenttts_units = 302`.
  - `tests/test_exp8.py` checks Table 5 and both constants against `stair_common.cost.units`.
- **Lower bounds.** These use AgentTTS's own token lengths, which understate its calls. Its retriever reads the whole
  100-passage pool (about 13k tokens on the team's estimate, not 2,048), and its fuser repeats that message. Our
  options' costs are measured, so a comparison of ours with 302 favours AgentTTS.
- **Against our options:**
  - 3B ×40 fused (71.1 units, nano-jev's ranking included) costs 24% of the recipe, at the same lenient accuracy.
  - The 70B option (93.6) is 30% of strong-everything.
- **What it changes:**
  - The experiment 8 reports' reference fields change, and so does their "on budget" operating point (25% of
    strong-everything: 232 units before, 78 now). This write-up doesn't use that point.
  - The AIQs and the near-best operating points depend on neither constant.
  - The reports in `results/exp8-common` were re-evaluated with the new constants (step 8.8, job 42849). Only those
    fields changed: every AIQ, interval, curve and near-best point is identical.
- **Original reports (2026-10-14 and 15):** their "% of 929" divides measured units by a Table 5 price. Their
  shares are about three times too small: the 70B's ~10% is 30%.
  - H3 (74.6 lenient at ≤ 25% of strong-everything) is still met: always-×40 is 0.746 at 22.9%.
  - But that is only because Eq. 8 roughly cancels the scale error. On the old rule, always-×40 cost 71% of
    strong-everything.

### The options on test (500 questions)

| Option | Lenient | LLM grade | Units a question |
|---|---|---|---|
| 3B, one answer (top 5) | 0.598 | 0.524 | 7.3 |
| 3B, 10 samples fused | 0.720 | 0.588 | 30.4 |
| 3B, 20 samples fused | 0.726 | 0.618 | 43.9 |
| 3B, 40 samples fused | **0.746** | 0.632 | 71.1 |
| 70B, one answer (top 5) | 0.670 | **0.748** | 93.6 |
| Oracle (cheapest correct option) | 0.844 | 0.816 | 16.3 (lenient), 21.4 (graded) |

- **Lenient:** 3B ×40 fused equals AgentTTS's 0.746 (as re-scored by the team) at 71 units. AgentTTS's recipe costs at
  least 302 (re-derived above), so this is its accuracy at under a quarter of its cost.
- **The two rulers disagree about the 70B.**
  - Lenient matching scores it 0.670; the LLM grader, reading its committed short answer, gives 0.748.
  - The grader marks the fused 3B answers 0.11 to 0.13 lower than lenient does.
  - Graded costs add the commit step: 0.1 units for the 3B, 2.4 for the 70B.

### Judges (AIQ with each judge's own cost, 95% CI)

**Lenient** (the standard ruler; `judge-5opt`):

| Router | AIQ [95% CI] | One judge, with its cost (seeds) |
|---|---|---|
| Oracle | 0.831 | |
| Best mix of fixed options | 0.716 [0.682, 0.752] | |
| nano-jev qpa | **0.711** [0.679, 0.752] | 0.704, 0.715, 0.706 |
| nano-jev qp | 0.705 [0.672, 0.745] | 0.710, 0.694, 0.704 |
| LR before the 3B answers | 0.696 [0.661, 0.737] | |
| Entropy threshold | 0.694 [0.656, 0.732] | |
| LR after the 3B answers | 0.693 [0.657, 0.735] | |

- **nano-jev qpa vs the baselines:**
  - − entropy is [+0.003, +0.039];
  - − LR after is [+0.001, +0.038];
  - − LR before is [−0.002, +0.037], a tie;
  - − the fixed mix is [−0.015, +0.010], a tie.
- **One seed of qpa − entropy:** [−0.004, +0.030], [+0.007, +0.042] and [−0.004, +0.034].
- **nano-jev qp:** it ties entropy ([−0.004, +0.032]).
- **Operating point** (chosen on calib: the cheapest point within 0.01 of the best):
  - the qpa judge reaches 0.724 at 51 units;
  - entropy reaches 0.730 at 70;
  - always using ×40 gives 0.746 at 71.
  - So the judge trades 0.02 accuracy for over a quarter less cost. The oracle (0.844 at 16 units) shows how much
    room is left.

**LLM grade** (`judge-5opt-graded`, and the deciders in `judge-decider-*`; each decider report carries the same
baselines and nano-jev judges):

| Router | AIQ [95% CI] | One judge, with its cost (seeds) | Units a question, one pass |
|---|---|---|---|
| Oracle | 0.792 | | |
| Entropy threshold | 0.667 [0.629, 0.710] | | |
| LR after the 3B answers | 0.666 [0.628, 0.708] | | |
| nano-jev qpa | 0.657 [0.622, 0.696] | 0.662, 0.658, 0.659 | 0.14 |
| nano-jev qp | 0.656 [0.623, 0.695] | 0.654, 0.653, 0.659 | 0.14 |
| LR before the 3B answers | 0.644 [0.607, 0.685] | | |
| Decider 2B fine-tuned, qpa | 0.641 [0.608, 0.682] | 0.659, 0.659, 0.655 | 2.59 |
| Best mix of fixed options | 0.639 [0.608, 0.678] | | |
| Decider 2B zero-shot, qp | 0.637 [0.601, 0.676] | | 2.47 |
| Decider 2B fine-tuned, qp | 0.636 [0.600, 0.675] | 0.645, 0.645, 0.648 | 2.47 |
| Decider E4B zero-shot, qp | 0.634 [0.599, 0.672] | | 4.89 |
| Decider E4B fine-tuned, qpa | 0.629 [0.593, 0.670] | 0.657, 0.655, 0.647 | 5.12 |
| Decider 2B zero-shot, qpa | 0.627 [0.590, 0.665] | | 2.59 |
| Decider E4B zero-shot, qpa | 0.624 [0.590, 0.664] | | 5.12 |
| Decider E4B fine-tuned, qp | 0.618 [0.583, 0.659] | 0.639, 0.642, 0.634 | 4.89 |

For fine-tuned judges, the AIQ column is the mean of 3 seeds with three passes' cost.

- **nano-jev qpa:**
  - − the fixed mix is [+0.006, +0.029];
  - − entropy is [−0.024, +0.008], a tie;
  - − LR after is [−0.022, +0.008], a tie.
- **Deciders, one judge − entropy:**
  - 2B qpa: [−0.021, +0.011], [−0.021, +0.009] and [−0.026, +0.005], all ties;
  - E4B qpa: [−0.024, +0.005], [−0.027, +0.007] and [−0.033, −0.001]: two ties and one worse.
- **Deciders, mean of 3 seeds:**
  - 2B qpa − entropy is [−0.038, −0.006], worse; − nano-jev qpa is [−0.025, −0.005], worse;
  - E4B qpa − entropy is [−0.052, −0.019], worse.
- **Zero-shot deciders:** worse than entropy (qpa: 2B [−0.051, −0.026], E4B [−0.056, −0.024]).
- **Before their cost,** one fine-tuned qpa decider scores 0.661–0.666 (2B) and 0.659–0.669 (E4B), against 0.658–0.662
  for nano-jev qpa. The decider routes no better here, and pays 18 to 37 times more per pass.
- **Operating point:** every router's operating point is close to always using the 70B (0.748 at 96 units). The judge
  qpa gives 0.744 at 100, and entropy 0.742 at 91.

**Clean subset** (438 questions): the same order on both rulers.
- Lenient: fixed mix 0.715, nano-jev qpa 0.710, qp 0.707, LR before 0.698, LR after 0.697, entropy 0.697.
- Graded: entropy 0.672, LR after 0.671, nano-jev qp 0.661, qpa 0.658.

**Training** (router_train labels, cheapest correct option):
- **Lenient:** 79% of questions are labelled "3B, one answer", which includes those no option answered.
- **Graded:** 74% are labelled that way, and 6% "70B".

## Caveats

- **500 test questions instead of 7,405.**
  - The intervals are about four times wider than in the original experiment 7. That is why nano-jev's original
    +0.011 over entropy no longer shows.
  - "Tie" here means "not detectable on 500 questions", not "equal".
- **Test-set overlap with the judges' training.** The clean-subset check is for this: it drops the 62 test questions
  nano-jev saw, and the order holds.
- **Lenient labels make "the large model helps" rarer:** the oracle escalates 12.8% (lenient) against 15.8% (F1).
  The F1 check gives the same order.
- **Seeds.** One seed's AIQ varies little next to the gaps between routers (mean ± sd, before cost): E4B qpa
  0.734 ± 0.006, 2B qpa 0.729 ± 0.002, nano-jev qp 0.694 ± 0.002, nano-jev qpa 0.696 ± 0.004.
- **The cost unit counts parameter FLOPs only.** It leaves out attention, memory and latency, as AgentTTS's unit does.
- **Experiment 8 trains on 1,000 questions,** and 74–79% of its labels are the cheapest option. More training data
  could change the judges' order there.
- **"AgentTTS units" are not AgentTTS's budget numbers.** Ours (`stair_common`) are FLOPs over one 3B pass of 192
  tokens. AgentTTS's Tables 5–9 count equivalent 3B samples, 3u − 2 a call, so a number taken from those tables is
  about three times ours. Convert before comparing.
- **The AgentTTS reference cost (302) is a lower bound.** It uses AgentTTS's own token lengths, not measured ones.

## Where the numbers come from

All paths are on OMEGA under `~/Custom-LLM-Router`; the job IDs are logged in `runs/common-jobs.txt`. Copies are in
`results/common`, `common-judges`, `common-judges-f1` and `exp8-common` ([README](../../results/common/README.md)).

| What | Job(s) | File |
|---|---|---|
| Splits, pools, expert answers | 42803, 42804 / 42811 | `runs/common/{calib,test}.parquet`, `config.toml` |
| Zero-shot judges | 42812 | `runs/common-judges/zeroshot/` |
| nano-jev judges (experiment 7) | 42813 | `runs/common-judges/judges/*nanojev*/seed{0,1,2}/` |
| 2B / E4B decider judges (experiment 9) | 42814, 42815 | `runs/common-judges/judges/*strands-decider*/` |
| F1-label judges | 42816 | `runs/common-judges-f1/` |
| Evaluation (the seed mean pays every pass, plus each seed on its own), team-layout export, loss curves | 42847 (replaces 42817, 42830, 42831) | `runs/common-judges/eval/A/report.json`, `report-clean.json`; `results/common/` |
| Experiment 8 answers | 42818–42823, 42828, 42829 | `runs/exp8-common/answers/` |
| Experiment 8 grader check, Eq. 8 recost, re-scoring | 42841, 42842, 42843 | `runs/exp8-common/rescore/`, `answers/summary.csv` |
| Experiment 8 judges: lenient, graded, deciders | 42844, 42845, 42846 | `runs/exp8-common/judge-*/` |
| Experiment 8 evaluation (baselines to the most accurate option, per-seed cost; then AgentTTS's reference costs) | 42848, 42849 | `runs/exp8-common/judge-*/eval/report{,-clean}.json` |
