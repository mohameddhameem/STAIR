# Results

One line per run; the full write-up of each is in [docs/experiments/](docs/experiments/), and the index with
published reference points in [docs/experiments.md](docs/experiments.md). Report files are in [results/](results/).
Experiments 1–7 and 9 score answer F1 on the 7,405 HotpotQA dev questions; AIQ is comparable only within one model
pair.

## Experiments 1–6: experts and feature routers (2026-10-05/06)

| Run | What | Main result |
|---|---|---|
| 1 | Pair A: Qwen2.5-1.5B (top 2 passages) / 7B (all 10); logistic regression routers | Always small F1 61.3 at 1,051 GFLOPs; always large 66.9 at 22,762; best router AIQ 0.665 (entropy 0.662, random 0.641, oracle 0.755) |
| 2 | Cascade routers (decide after the small answer) | AIQ 0.672, the only routers significantly above entropy |
| 3 | 7B reading nano-jev's top 3 / top 5 | Better and cheaper: F1 69.2 at 7,292 / 70.0 at 11,598 GFLOPs |
| 4 | Large = Qwen2.5-14B (pair B) | F1 71.9; router AIQ 0.697 |
| 5 | Small = Qwen2.5-3B (pair C) | Router AIQ 0.681 |
| 6 | Three-stage cascade 1.5B → 7B (top 3) → 14B | F1 72.6 at 14,243 GFLOPs (15% reach the 14B) |

## Experiment 7: the System One judge (2026-10-07)

Fine-tuned nano-jev (33M) on pair A; 3 seeds, scores averaged. ([write-up](docs/experiments/2026-10-07-system-one-router.md),
[report](results/system-one/eval/A/report.json))

| Policy | AIQ | Judge cost |
|---|---|---|
| Entropy threshold | 0.662 | |
| Best logistic regression before the answer | 0.665 | |
| Zero-shot nano-jev ("is the answer correct?") | 0.664 | 33 GFLOPs |
| **Judge qp** (question + top-2 passages) | **0.673** | 32 GFLOPs |
| Judge qpa (adds the small answer) | 0.673 | 33 GFLOPs |

qp minus the best logistic regression before the answer: [+0.004, +0.011]. On pair B the best judge edges the best cascade router ([+0.000, +0.006]); on pair C (a different small model) the gain is small and not clearly above entropy.
No router reaches always-large's F1 at its operating point.

## Experiment 8: the AgentTTS setting (2026-10-14/15)

AgentTTS's 500 HotpotQA test questions, five answer options, cost in AgentTTS units (strong-everything = 311 units;
the 929 quoted here before was the same thing on Table 5's price scale).
Re-scored with an LLM grader (Qwen2.5-14B) on each model's short committed answer.
([write-up](docs/experiments/2026-10-15-experiment-8-rescored.md), [report](results/exp8-rescore/))

| Option | Lenient | **Graded** | Units |
|---|---|---|---|
| Llama 3.2 3B, one answer | 0.608 | 0.524 | 7.5 |
| 3B, 10 samples fused | 0.726 | 0.588 | 64.9 |
| 3B, 40 samples fused | 0.756 | 0.632 | 220.0 |
| **Llama 3.1 70B (4-bit), one answer** | 0.680 | **0.748** | 96.0 |
| Oracle (cheapest correct option) | | 0.816 | 30.3 |

**Correction (2026-10-23):** these units count the prompt once per sample. Under AgentTTS's Eq. 8 the samples
share it, so the fused rows cost less: ×10 30, ×40 71 units. See the
[common-standard write-up](docs/experiments/2026-10-18-common-redo.md).

The judge picks the 70B for 500 of 500 (qp) and 499 of 500 (qpa) questions; its AIQ ties the fixed-policy hull
(qp 0.701) or slightly beats it (qpa 0.713). AgentTTS's own 74.6 is lenient and not comparable until its answers
are graded the same way.

## Experiment 9: Amazon's Strands Decider as the judge (2026-10-16/17)

2B (Qwen3.5 torso) and 4B-active (Gemma 4 E4B) deciders, zero-shot and fine-tuned (LoRA + pointer head, 3 seeds).
([write-up](docs/experiments/2026-10-17-experiment-9.md), [results](results/exp9/))

| Setting | Judge | AIQ | With judge cost | Minus nano-jev qp |
|---|---|---|---|---|
| Exp. 7, pair A | 2B fine-tuned, qpa | 0.708 | **0.702** | [+0.030, +0.039] |
| | 4B fine-tuned, qpa | **0.713** | 0.700 | [+0.035, +0.045] |
| | 4B zero-shot | 0.693 | 0.683 | [+0.016, +0.024] |
| | 2B fine-tuned, qp | 0.680 | 0.676 | [+0.004, +0.011] |
| | nano-jev qp | 0.673 | 0.673 | |
| Exp. 8, graded | best decider (2B zero-shot, qpa) | 0.700 (cost included) | | hull 0.701; nano-jev qpa 0.713 |

The decider gains mostly from reading the small model's answer; on experiment 8 (3,000 training questions, few
examples for the expensive options) no decider beats the hull.

## Experiments 7–9 on the common standard (2026-10-18)

Rerun on the team's first common standard ([write-up](docs/experiments/2026-10-18-common-redo.md),
[results](results/common-judges/)):
- the test set is AgentTTS's 500 questions (dev [500:1000]), with calib dev [0:500];
- scoring is lenient;
- costs are AgentTTS units, with each judge's own cost.

It is kept as the comparison with AgentTTS on its own questions.

| Experiment | Result |
|---|---|
| 7 / 9 | One decider qpa judge (2B or E4B, one seed), with its cost: AIQ **0.724**. It beats entropy (0.697) and the nano-jev judges (~0.70), and ties the best LR cascade router (0.706) |
| 8 | 3B ×40 fused: lenient **0.746** (AgentTTS's re-scored 0.746) at 71 units, below the 70B (0.670 at 94). AgentTTS's own setting costs 302 units (97% of strong-everything's 311) |
| 8, option judge | nano-jev qpa 0.711: ties the best fixed mix (0.716), beats entropy [+0.003, +0.039] |

These figures include two later fixes, both re-evaluated without retraining:
- a qpa judge pays for the cheap answer it reads (2026-10-21);
- a policy that runs two options pays nano-jev's ranking once (2026-10-22).

## The golden set (2026-10-21 to 23): Jonathan's SLR setup

**Setup** ([write-up](docs/experiments/2026-10-21-slr-standard.md)):
- the test split is `slr-test` (1,481 questions);
- answers are scored with the standard lenient ruler;
- costs are AgentTTS Table 5 prices per call, weak-first, with strong-everything = 929. Jonathan's routers are not
  charged; ours are reported both ways.

**Verification** ([results](results/slr/)):
- his ruler, split and three certified rows reproduce exactly;
- thresholds are re-selected on `slr-val` by his own rule.

**Menu A paths on `slr-test`:**

| Path | Lenient | Price |
|---|---|---|
| WW | 50.30 | 80 |
| WS | 64.01 | 192 |
| SW | 61.78 | 817 |
| SS (always-strong) | 76.77 | 929 |
| Oracle | 81.5 | 205.5 |

**Experiment 10: the judge routing menu A** ([results](results/exp10-slr/)). AIQ is over weak-first prices 80–1,009,
with the paired bootstrap.

| Router | AIQ | With its own price | At his operating rule |
|---|---|---|---|
| Decider E4B qpa | **0.736** | 0.730 | 76.70 at 737.5 (3 seeds) |
| Decider 2B qpa | 0.731 | 0.730 | all-SS |
| SLR SFT (label) + reward (Jonathan) | 0.726 | (uncharged) | 75.96 at 664.7 |
| SLR SFT (label) (Jonathan) | 0.724 | (uncharged) | 76.70 at 834.8 |
| nano-jev qpa | 0.719 | 0.719 | all-SS |

- **E4B − his routers:** before its price [+0.004, +0.021] and [+0.003, +0.019]; with it, ties.
- **2B − his label router:** before its price [+0.001, +0.014]; with it, a tie.

**Experiments 7 / 9 on pair A** ([results](results/slr-judges/)). Prices: Qwen 1.5B 0.5, Qwen 7B 10.

| Router | Before its price | With it |
|---|---|---|
| E4B decider, fine-tuned, qpa | **0.711** | one judge 0.655 (4.4 a call) |
| 2B decider, fine-tuned, qpa | 0.696 | one judge **0.677–0.682** (1.3 a call) |
| LR cascade | 0.676 | |
| nano-jev qpa | 0.677 | 0.677 |
| Entropy | 0.665 | |

One priced 2B qpa judge beats entropy for all three seeds, and ties the LR cascade.

**Experiment 8: cheap retrieval and sampled answers** ([results](results/exp8-slr/)). Every option reads nano-jev's
top 5.

| Option | Lenient | Price |
|---|---|---|
| Llama 3B, one answer | 61.44 | 9.6 |
| 3B, 10 samples fused | 74.34 | 21.6 |
| **3B, 20 samples fused** | **75.08** | **31.6** |
| 3B, 40 samples fused | 74.88 | 51.6 |
| Llama 70B, one answer | 68.33 | 121.6 |

- **×20 fused − always-strong:** −1.69 [−4.19, +0.68], a tie, at 3.4% of the price.
- **The option judge** ties the fixed options and the baselines.

**H2: why nano-jev's evidence lags.** Our 70B with Jonathan's prompt:
- on his 72B's selections, it scores 76.23 (his always-strong is 76.77);
- on nano-jev's top 5, it scores 69.82: −6.4 points for retrieval;
- the prompt itself is worth 1.5, which is not significant.

**Published judges (2026-10-25):** nano-jev qp and qpa, retrained with their weights saved. Their AIQ is 0.674 /
0.674 (3-seed ensembles). The qpa run above (0.677) is the same recipe without saved weights. Links and numbers are
in [MODELS.md](MODELS.md).

**Published judges (2026-10-26):** the fine-tuned Strands Decider 2B and E4B, qp and qpa, on pair A (experiment 9) and
menu A (experiment 10).
- They too are retrains with their weights saved, and land within 0.002 of the experiment runs' seed means.
- On menu A, the E4B qpa ensemble scores 0.739, and 0.732 with its price. With its price it narrowly beats both of
  jonathan's routers, where the experiment run tied them, so the two draws together read "ties or narrowly beats".
- Their reports are in [results/decider-release/](results/decider-release/), and links and numbers in
  [MODELS.md](MODELS.md).

## The Jev novelty programme (2026-10-23 to 25)

**What it tested:** what nano-jev's typed-decision pretraining adds to routing on pair A. The full write-up is
[here](docs/experiments/2026-10-25-jev-novelty.md).
- **Pre-registration:** [here](docs/experiments/2026-10-24-jev-preregistration.md). It was committed before any
  MuSiQue or 2Wiki answers existed. Its dates read 2026-10-24, but it was written on 2026-10-23 (deviation 8).
- **Datasets:** HotpotQA is exploratory. MuSiQue and 2WikiMultiHopQA are confirmatory, so a claim is *confirmed*
  when it holds on both.
- **Tests:** δ = 0.010 AIQ, Holm within each family. Report files are in [results/jev/](results/jev/).
- **2Wiki split:** the team's 2Wiki val and test parts have no gold answers, so this programme uses 2Wiki's
  official dev set in the team's order (deviation 1).

| Claim | Status | HotpotQA / MuSiQue / 2Wiki |
|---|---|---|
| H1a: starting from nano-jev beats plain MiniLM (label-curve area, qp) | **Confirmed** | +0.021 / +0.010 / +0.019 |
| H1a, qpa | Replicated | +0.022 / +0.008 / +0.003 |
| H1b: the advantage is largest with few labels | Fails | the gap grows with labels |
| H1d: the typed head beats a plain head | Fails | +0.004 / +0.004 / +0.005, none after Holm |
| H2a: passages beat the question alone | **Confirmed** | MuSiQue +0.011, 2Wiki +0.025 |
| H2b: question-only routes at chance | Fails | it beats random by +0.010 / +0.030 |
| H2c: non-inferior to entropy and the LR routers | **Confirmed** | better only for qp on 2Wiki |
| H2d: a HotpotQA judge keeps 75% of its gain elsewhere | Not shown | 68–108%, wide intervals |
| H3a: non-inferior to every cheap weak-model signal (MSP, log-prob, margin, P(True), NLI, hidden-state probe) | **Confirmed** | closest is the probe, +0.009 / +0.005 / +0.008 |
| H3b: beats self-consistency and semantic entropy (10 samples) | Confirmed, priced | with the samples free: +0.015 on HotpotQA, +0.036 on 2Wiki, a tie on MuSiQue |
| H4a: one untrained typed question closes half the random-to-trained gap | **Confirmed** | 77% / 86% / 83% |
| H4b: 50 labels close 75% | Fails | 48–65% |
| H5a: a frozen judge transfers to other model pairs | Exploratory | 2 of 3 pairs, including Llama 3B / 8B |
| H6: risk-controlled escalation saves cost | Trivial | it keeps 0–0.5% of questions |
| E6: one 4-option question on menu A | Exploratory | +0.004 [+0.002, +0.007] over two stage decisions; against Jonathan's 1.5B router [−0.009, +0.002] |
| E7: nano-jev's ranking with 20 fused samples | Descriptive | +0.057 over MS MARCO, +0.122 over BM25 |
| E9: latency per decision | Measured | CPU 123 ms (1.5B router 4.8 s, 39×); L40S 6.4 ms (11.5 ms, 1.8×) |

**Not run** (the write-up lists every item): the dollar re-pricing, the BERT and kNN question routers, and H3 on
2Wiki's 2,000-question secondary test.
