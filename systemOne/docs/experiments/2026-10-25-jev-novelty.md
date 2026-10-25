# The Jev novelty programme: what nano-jev's typed decisions add to routing (results)

**Done 2026-10-25.** Every experiment in the [pre-registration](2026-10-24-jev-preregistration.md) ran, except the
items under [Not done](#not-done).
- **Job sheet:** [gpu-jobs-jev.md](../gpu-jobs-jev.md). Job ids are in
  [results/jev/jev-jobs.txt](../../results/jev/jev-jobs.txt).
- **Outputs:** every report, confirmatory file and job log is in [results/jev/](../../results/jev/), named as in
  `runs/`. No model weights or hidden states are included.

**How to read the numbers:**
- **Status of each dataset:** HotpotQA (the golden set) is *exploratory*; MuSiQue and 2WikiMultiHopQA are
  *confirmatory*.
  - A claim is **confirmed** when it holds on both confirmatory datasets, and **replicated** when it holds on at least
    2 of the 3.
- **Margin:** δ = 0.010 AIQ. Holm is applied within each family.
- **Intervals:**
  - **Family tests (H2–H5):** 90% intervals, the one-sided 95% bounds the tests use.
  - **H1's label-curve report and the descriptive comparisons (E6, E7):** 95%.
- **Judge AIQ:** the mean over seeds of each seed's AIQ, i.e. one deployed judge. Menu A (E1's exploratory arm and
  E6) is the exception: it reports the 3-seed ensemble, as experiment 10 always has.
- **Pair A** throughout: Qwen2.5-1.5B on nano-jev's top 2 passages, then Qwen2.5-7B on all of them. Table 5 prices
  (0.5 and 10 per call); nano-jev's own calls floor at 0.
- **Dates:** the pre-registration's dates read 2026-10-24, but the work was done on 2026-10-23 (deviation 8).

## What the evidence supports

| Claim | Status | Numbers (HotpotQA / MuSiQue / 2Wiki) |
|---|---|---|
| **Typed-decision pretraining is a better start than plain MiniLM** (H1a, qp) | **Confirmed** | LE-AUC +0.021 / +0.010 / +0.019 |
| The same, for qpa | Replicated, not confirmed | +0.022 / +0.008 / +0.003 (n.s.) |
| The advantage is largest with few labels (H1b) | **Fails everywhere** | The gap grows with labels |
| The typed head beats a plain head on nano-jev (H1d) | Fails | +0.004 / +0.004 / +0.005, none after Holm |
| **The passages, not the question, decide escalation** (H2a) | **Confirmed** | qp − q: +0.011 / +0.025 (+0.020 on 2Wiki's 2,000) |
| **nano-jev is non-inferior to entropy and the LR routers** (H2c) | **Confirmed**, qp and qpa | Non-inferior, not better (one exception below) |
| A question-only judge routes at chance (H2b) | **Fails on both** | It beats random by +0.010 / +0.030 |
| A HotpotQA judge keeps 75% of its gain on a new dataset (H2d) | Not shown | Points 68–108%, intervals too wide |
| **Non-inferior to every cheap weak-model signal, including a hidden-state probe** (H3a) | **Confirmed** | Closest: the probe, +0.009 / +0.005 / +0.008; on MuSiQue mean log-prob ties (−0.000) |
| Beats self-consistency and semantic entropy, priced (H3b) | Confirmed | Priced, both fall below random. Free, nano-jev still wins on HotpotQA and 2Wiki (+0.015 / +0.036) and ties on MuSiQue |
| **One untrained typed question closes ≥ 50% of the random-to-trained gap** (H4a) | **Confirmed** | 77% / 86% / 83% |
| 50 labels close ≥ 75% (H4b) | Fails everywhere | 48–65% |
| The zero-label pre-answer score beats priced entropy (H4c) | 2Wiki only | |
| **A frozen judge transfers across model pairs** (H5a, HotpotQA) | Exploratory: 2 of 3 pairs | Including Llama 3B / 8B |
| The risk guarantee buys savings (H6) | Holds only trivially | It keeps 0–0.5% of questions at α ≤ 0.2 |
| **One 4-option question vs two stage decisions** (E6, menu A) | Exploratory | +0.004 [+0.002, +0.007]; the first nano-jev judge not significantly below the 1.5B router |
| **nano-jev's ranking makes cheap evidence + sampling work** (E7) | Descriptive | vs MS MARCO +0.057, vs BM25 +0.122, at 20 samples |
| **Its speed case is CPU, not GPU** (E9) | Descriptive | GPU 1.8× faster than the 1.5B router; CPU 39× |

**In short.**
- **What holds:**
  - Fine-tuning a router from nano-jev, rather than from the same 33M MiniLM without decision training, gives a
    better router on all three datasets.
  - The resulting judge reads only text, so it needs no access to the weak model. Even so, it is non-inferior to
    every cheap signal from the weak model, including a probe on its hidden states.
  - With no routing labels at all, one typed question recovers 77–86% of a trained judge's gain.
- **What does not hold:**
  - It is not a few-label effect.
  - The typed head itself adds nothing.
  - It does not beat entropy or logistic regression, only matches them.
  - On menu A it still trails Jonathan's 1.5B router, though the joint question closes most of the gap.

## H1 (E1): decision pretraining × label budget

**Arms:**
- **Five starting points**, all MiniLM-L12 with 33M parameters: plain, MS MARCO, nano-jev v1.0, our retrain of v1.0's
  recipe (repro) and noHotpot.
- **Two heads**, typed and plain.
- **Budgets:** 50–4,814 labels on HotpotQA, with 10 seeds up to 500 and 5 above. On the confirmatory datasets the
  budgets are 100, 500 and 4,814, each with 5 seeds (deviation 8).
- **Hyperparameters:** identical for every arm: nano-jev v1.0's fine-tuning settings, with at least 150 steps.

**Gate:** our retrain reaches v1.0's accuracies on nano-jev's own test split
([results/jev/nanojev-pretrain](../../results/jev/nanojev-pretrain/)).

| Decision | v1.0 | repro | noHotpot (deviation 2) |
|---|---|---|---|
| grounded | 0.844 | 0.858 | 0.844 |
| relevance | 0.815 | 0.812 | **0.301** |
| sufficient | 0.845 | 0.854 | 0.685 |

**LE-AUC** (the area under AIQ vs log2 budget; typed head, qp; HotpotQA is the 1,303 clean questions):

| Start | HotpotQA | MuSiQue | 2Wiki |
|---|---|---|---|
| nano-jev v1.0 | **0.6602** | **0.2834** | **0.4797** |
| nano-jev repro | 0.6591 | 0.2845 | 0.4782 |
| nano-jev noHotpot | 0.6468 | 0.2778 | 0.4649 |
| MS MARCO MiniLM | 0.6439 | 0.2766 | 0.4629 |
| plain MiniLM | 0.6391 | 0.2731 | 0.4607 |
| *random routing* | *0.6358* | *0.2689* | *0.4430* |

**Family H1** (95% intervals, Holm within qp and within qpa):

| Test | HotpotQA (exploratory) | MuSiQue | 2Wiki | Status |
|---|---|---|---|---|
| H1a qp: v1.0 > plain | +0.021 [+0.014, +0.029], Holm 0.000 | +0.010 [+0.005, +0.015], 0.000 | +0.019 [+0.008, +0.032], 0.008 | **Confirmed** |
| H1a qpa | +0.022 [+0.015, +0.030], 0.000 | +0.008 [+0.002, +0.012], 0.010 | +0.003 [−0.007, +0.015], 1.000 | Replicated |
| H1b qp: gap at 100 > gap at full | −0.023 [−0.033, −0.012] | −0.009 [−0.017, +0.000] | −0.008 [−0.035, +0.014] | Fails |
| H1b qpa | −0.015 [−0.027, −0.004] | −0.024 [−0.032, −0.017] | −0.019 [−0.040, +0.002] | Fails |
| H1c qp: noHotpot keeps half | −0.003 [−0.007, +0.001] | −0.001 [−0.004, +0.003] | −0.005 [−0.014, +0.004] | Fails (caveat 2) |
| H1c qpa | −0.001 [−0.006, +0.003] | +0.000 [−0.003, +0.004] | −0.012 [−0.020, −0.002] | Fails |
| H1d qp: typed > plain head | +0.004 [+0.000, +0.008], Holm 0.096 | +0.004 [−0.001, +0.009], 0.33 | +0.005 [−0.006, +0.016], 1.0 | Fails |
| H1d qpa | +0.005 [−0.000, +0.009], 0.14 | −0.003 [−0.008, +0.002], 1.0 | −0.006 [−0.015, +0.003], 1.0 | Fails |

**Reading:**
- **Pretraining helps, but not with fewer labels.** Here is AIQ at 100 labels and at all 4,814 (typed, qp):

  | Start | HotpotQA 100 → 4,814 | MuSiQue | 2Wiki |
  |---|---|---|---|
  | nano-jev v1.0 | 0.651 → 0.671 | 0.279 → 0.287 | 0.468 → 0.496 |
  | MS MARCO | 0.639 → 0.652 | 0.273 → 0.283 | 0.450 → 0.486 |
  | plain MiniLM | 0.640 → 0.637 | 0.272 → 0.271 | 0.456 → 0.476 |

  - Under the shared settings, plain MiniLM barely learns on HotpotQA and MuSiQue. It sits at random even with
    every label. So the gap grows with labels (H1b), and "nano-jev at 100 labels beats plain at 4,814" is true
    there but says more about plain than about nano-jev.
  - On 2Wiki plain does learn, and nano-jev at 100 labels does not reach it.
  - Settings tuned for nano-jev may handicap the plain start (caveat 3). But the MS MARCO start shares the
    architecture and trains normally, and nano-jev beats it in every cell above.
- **It is the recipe, not a lucky checkpoint.** Our retrain's LE-AUC is within 0.0016 of v1.0's in 5 of 6
  dataset × input cells; the sixth is +0.007 (MuSiQue qpa).
- **The typed interface is not where the gain comes from.** Typed and plain heads tie on the same nano-jev weights
  (H1d).

**Exploratory: menu A at each budget.**
- The test is nano-jev qpa against Jonathan's 1.5B SFT label router, both priced, as the 3-seed ensemble.
- **It is significantly below his router at every budget:**

  | Budget | nano-jev | Difference |
  |---|---|---|
  | 100 | 0.7076 | [−0.023, −0.010] |
  | 500 | 0.7136 | [−0.017, −0.005] |
  | 1,000 | 0.7134 | [−0.017, −0.005] |
  | 4,814 | 0.7171 | [−0.013, −0.002] |

  So it is equivalent at none.
- **Context:**
  - SLR scores 0.7242, and the best fixed path 0.6932.
  - At every budget nano-jev stays above MS MARCO (0.7002 → 0.7136), plain MiniLM (0.6874 → 0.7013) and noHotpot
    (0.6942 → 0.7161).

## H2 (E2): evidence decides escalation on new datasets (confirmatory)

**Data:**
- MuSiQue-Ans dev (2,417) and 2Wiki's test (500); pair A as everywhere.
- 2Wiki's split had to be rebuilt from the answered dev set (deviation 1).
- **The gold pair is rarely in the weak model's top 2 passages:** 16% on MuSiQue, 37% on 2Wiki.
- **F1 of the two experts:**

  | | Always-weak | Always-strong | Oracle |
  |---|---|---|---|
  | MuSiQue | 0.205 | 0.333 | 0.421 |
  | 2Wiki | 0.352 | 0.534 | 0.646 |

**AIQ** (one deployed judge; [e2-musique](../../results/jev/e2-musique/confirm/H2.json),
[e2-2wikimultihopqa](../../results/jev/e2-2wikimultihopqa/confirm/)):

| | MuSiQue | 2Wiki (500) | 2Wiki secondary (2,000) |
|---|---|---|---|
| random | 0.2689 | 0.4430 | 0.4363 |
| q judge (question only) | 0.2789 | 0.4727 | 0.4631 |
| **qp judge** | **0.2895** | **0.4981** | **0.4833** |
| qpa judge | 0.2860 | 0.4947 | 0.4799 |
| LR question+evidence (the stronger "before" opponent, chosen on calib) | 0.2834 | 0.4878 | 0.4727 |
| LR question+evidence+small (the stronger cascade opponent) | 0.2844 | 0.4839 | 0.4762 |
| entropy (post) | 0.2881 | 0.4628 | |
| HotpotQA-trained qp, no retraining | 0.2885 | 0.4877 | 0.4706 |
| HotpotQA-trained qpa | 0.2874 | 0.4780 | 0.4687 |

**Family H2** (90% intervals):

| Test | MuSiQue | 2Wiki (500) | 2Wiki (2,000) | Status |
|---|---|---|---|---|
| H2a qp > q | +0.011 [+0.005, +0.016], Holm 0.006 | +0.025 [+0.013, +0.037], 0.002 | +0.020 [+0.014, +0.027], 0.000 | **Confirmed** |
| H2b q within δ of random | +0.010 [+0.007, +0.016], 0.67 | +0.030 [+0.021, +0.044], 1.0 | +0.027 [+0.022, +0.034], 1.0 | **Fails**: q beats random |
| H2c qp ≥ stronger before − δ | +0.006 [+0.000, +0.011], 0.000 | +0.010 [+0.001, +0.021], 0.004 | +0.011 [+0.006, +0.015], 0.000 | **Confirmed** |
| H2c qpa ≥ stronger cascade − δ | +0.002 [−0.004, +0.008], 0.002 | +0.011 [−0.002, +0.021], 0.010 | +0.004 [−0.003, +0.010], 0.001 | **Confirmed** |
| H2d qp keeps ≥ 75% | +0.004 [−0.001, +0.009], 0.14 | +0.003 [−0.008, +0.016], 0.88 | −0.001 [−0.007, +0.005], 1.0 | Not shown |
| H2d qpa keeps ≥ 75% | +0.006 [+0.001, +0.011], 0.08 | −0.004 [−0.015, +0.010], 1.0 | −0.000 [−0.008, +0.008], 1.0 | Not shown |

**Reading:**
- **Non-inferior, mostly not better.** The one exception is qp on 2Wiki, where the lower bound clears zero on both
  test sets.
  - On MuSiQue the calib-chosen opponent was the LR router, but entropy is stronger there on test (0.2881).
  - On test, qp is +0.001 above entropy and qpa −0.002 below it.
- **The question alone carries signal on these datasets.**
  - On HotpotQA it routed at chance. Here it gets about half of qp's gain over random, on both datasets.
  - Earlier reports' "question-only routing is at chance" is a HotpotQA result only.
- **Transfer from HotpotQA is partial.** The HotpotQA judges keep 95% / 108% of the in-domain gain on MuSiQue (qp /
  qpa), and 81% / 68% on 2Wiki. The intervals are too wide to show 75%.

## H3 (E3): the stronger baselines (confirmatory on the new datasets)

**Signals:**
- From the weak model: MSP, mean log-prob, first-token margin, P(True), a logistic probe on its last hidden layer,
  self-consistency (k = 10), semantic entropy (k = 10) and self-verification (k = 8).
- From the passages: an NLI cross-encoder, and gradient-boosted trees on retrieval scores.
- **Pricing:** every signal that calls the weak model again pays for it under Eq. 8 (one extra pass for P(True), 10
  samples for the sampled signals).

**AIQ** (nano-jev qpa, one deployed judge):

| | HotpotQA (1,303) | MuSiQue | 2Wiki (500) |
|---|---|---|---|
| random | 0.6358 | 0.2689 | 0.4430 |
| **nano-jev qpa** | **0.6704** | **0.2860** | **0.4947** |
| hidden-state probe | 0.6614 | 0.2813 | 0.4865 |
| mean log-prob | 0.6585 | 0.2860 | 0.4549 |
| MSP | 0.6576 | 0.2850 | 0.4549 |
| NLI check | 0.6573 | 0.2803 | 0.4730 |
| first-token margin | 0.6537 | 0.2785 | 0.4495 |
| P(True) (its one extra pass prices to 0) | 0.6432 | 0.2722 | 0.4641 |
| retrieval-score trees | 0.6430 | 0.2700 | 0.4504 |
| semantic entropy, priced | 0.6481 | 0.2680 | 0.4361 |
| self-consistency, priced | 0.6480 | 0.2652 | 0.4346 |

**Family H3** (nano-jev qpa minus each signal, 90% intervals; every test is supported after Holm):

| Signal | HotpotQA (exploratory) | MuSiQue | 2Wiki |
|---|---|---|---|
| H3a ≥ hidden-state probe − δ | +0.009 [−0.000, +0.020] | +0.005 [−0.001, +0.011] | +0.008 [−0.003, +0.018] |
| H3a ≥ mean log-prob − δ | +0.012 [+0.001, +0.023] | −0.000 [−0.007, +0.007] | +0.040 [+0.025, +0.051] |
| H3a ≥ MSP − δ | +0.013 [+0.002, +0.023] | +0.001 [−0.006, +0.008] | +0.040 [+0.025, +0.052] |
| H3a ≥ NLI − δ | +0.013 [+0.004, +0.023] | +0.006 [−0.001, +0.012] | +0.022 [+0.007, +0.035] |
| H3a ≥ first-token margin − δ | +0.017 [+0.006, +0.027] | +0.008 [+0.001, +0.014] | +0.045 [+0.028, +0.057] |
| H3a ≥ P(True) − δ | +0.027 [+0.017, +0.036] | +0.014 [+0.007, +0.020] | +0.031 [+0.012, +0.045] |
| H3a ≥ retrieval trees − δ | +0.027 [+0.017, +0.036] | +0.016 [+0.010, +0.021] | +0.044 [+0.028, +0.057] |
| H3b > self-consistency | +0.023 [+0.012, +0.032] | +0.021 [+0.013, +0.027] | +0.060 [+0.044, +0.077] |
| H3b > semantic entropy | +0.022 [+0.012, +0.032] | +0.018 [+0.011, +0.025] | +0.059 [+0.043, +0.076] |

**With the extra calls free** (descriptive; [H3-free.json](../../results/jev/e2-musique/confirm/H3-free.json) per
dataset; nano-jev qpa minus the free signal, 90% intervals):

| Signal, extra calls free | HotpotQA | MuSiQue | 2Wiki |
|---|---|---|---|
| self-consistency (AIQ) | 0.6552 | 0.2811 | 0.4574 |
| nano-jev − it | **+0.015 [+0.005, +0.025]**, Holm 0.023 | +0.005 [−0.002, +0.011], 0.44 | **+0.037 [+0.022, +0.053]**, 0.002 |
| semantic entropy (AIQ) | 0.6554 | 0.2839 | 0.4589 |
| nano-jev − it | **+0.015 [+0.004, +0.025]**, 0.023 | +0.002 [−0.005, +0.009], 0.61 | **+0.036 [+0.021, +0.052]**, 0.002 |
| self-verification, k = 8 (AIQ) | 0.6613 | 0.2877 | 0.4798 |
| nano-jev − it | +0.009 [−0.001, +0.019], 0.064 | −0.002 [−0.008, +0.005], 0.64 | +0.015 [−0.000, +0.030], 0.053 |

P(True) is missing from this table because its free AIQ equals its priced one on every dataset: one extra pass with
a one-token answer prices to 0 under Table 5's floor. So the H3a row for P(True) above is effectively free too.

**Reading:**
- **H3a is the programme's strongest confirmatory result.** nano-jev reads only the passages and the weak answer's
  text. Yet it is non-inferior on all three datasets to a probe on the weak model's own hidden states, which needs
  the weak model's weights and the labels to fit it.
  - The pre-registration's fallback reading applies only partly: the probe does not beat nano-jev, it ties.
  - On MuSiQue, the cheapest white-box signals (mean log-prob, MSP) also tie.
- **H3b is mostly, but not only, about price.**
  - Priced, self-consistency and semantic entropy fall below random on MuSiQue and 2Wiki.
  - Free, they still lose to nano-jev on HotpotQA (by 0.015) and 2Wiki (by 0.036). On MuSiQue the difference is not
    significant.
  - AutoMix-style self-verification, free, is not significantly different from nano-jev anywhere (2Wiki's lower
    bound is −0.000).
  - So on MuSiQue only their price separates the sampled signals from nano-jev.

## H4 (E4): routing from typed decisions with no or few labels

**What is compared:**
- **Scores:** nano-jev v1.0's untrained typed decisions: sufficient (top 2), relevance (top 1 and top 2), grounded
  and "is the proposed answer correct?".
- **Compositions:** each score on its own, the product and maximum of the pre-answer scores, and logistic combiners
  on 10, 25 or 50 calib labels (20 draws each).
- **The gap:** from random to the full-budget trained judge, closed on test.

| | HotpotQA (1,303) | MuSiQue | 2Wiki (500) |
|---|---|---|---|
| Score chosen on calib | not-grounded | insufficient | insufficient |
| **Its share of the gap (H4a ≥ 50%)** | **77%**, Holm 0.013 | **86%**, 0.003 | **83%**, 0.002 |
| Best single score on test | not-correct, 85% | not-grounded, 108% | not-grounded, 87% |
| 50-label combiner (H4b ≥ 75%) | 65%, fails | 50%, fails | 48%, fails |
| Pre-answer score − priced entropy (H4c) | −0.000 [−0.009, +0.012] | −0.002 [−0.008, +0.006] | **+0.026 [+0.011, +0.043]** |

**Reading:**
- **H4a is confirmed.** One typed question that nano-jev was never trained to route with recovers most of a trained
  judge's gain on every dataset.
  - On MuSiQue the sufficiency question alone (0.2866) is within 0.003 of the trained qp judge (0.2895).
- **Few labels hurt rather than help.** On every dataset the 50-label combiner scores below the single zero-label
  score chosen on calib. Fifty examples are too few to fit even a small combiner.
- **H4c holds on 2Wiki only** (1 of 3), so the zero-label score saving the weak call is not a general win.

## H5 (E5): transfer across model pairs (HotpotQA, exploratory)

**Setup:**
- Pair A's judges, frozen, are scored on three other pairs against judges retrained on each pair.
- **Pair D is new:** Llama-3.2-3B on the top 2 passages (EM 0.476) and Llama-3.1-8B on all of them (EM 0.552), over
  all 7,405 dev questions.

| Pair | Frozen qp | Retrained qp | H5a: frozen − retrained (90%) | Holm | Transfer ratio qp / qpa |
|---|---|---|---|---|---|
| B: 1.5B / 14B | 0.7002 | 0.7013 | −0.001 [−0.004, +0.002] | 0.000 | 0.97 / – |
| C: 3B / 7B | 0.6755 | 0.6807 | −0.005 [−0.009, +0.001] | 0.10 | 0.85 / 0.81 |
| D: Llama 3B / 8B | 0.7046 | 0.7077 | −0.003 [−0.006, +0.001] | 0.002 | 0.91 / 0.93 |

**Reading:**
- **H5a holds on B and D** (non-inferior), including a different model family. **On C it is not shown:** the lower
  bound is −0.0095, just past −δ.
- **H5b fails** on C and D: there is no sign that qpa, which reads the weak model's answer, transfers worse than qp
  when the weak model changes.

## H6 (E8): risk-controlled escalation

**Method:** Learn-then-Test thresholds (deviation 4) bound the error among the questions kept on the weak model, at
α = 0.05 / 0.10 / 0.20, over 1,000 resplits. α = 0.30 / 0.40 are post-hoc (deviation 6).

| | HotpotQA | MuSiQue | 2Wiki |
|---|---|---|---|
| Weak model's error rate | 39.5% | 79.1% | 64.2% |
| Kept on the weak model, α = 0.05 / 0.10 / 0.20 | 0% / 0% / 0.5% | 0% | 0% |
| H6a (risk ≤ α, violated in ≤ 10% of resplits) | holds | holds | holds |
| Post hoc α = 0.30 | keeps 42%, but violated in 14% of resplits | 0% | 0% |
| Post hoc α = 0.40 | keeps 94% | 0% | 0% |

**Reading:**
- H6a holds only because the rule escalates nearly everything. The weak model is wrong too often for an error bound
  of 20% or less to keep any of its answers.
- **H6b, the expected failure under shift, does not show at the pre-registered α.** Calibrated on HotpotQA and
  applied to MuSiQue, the rule keeps nothing, so nothing is violated.
  - Post hoc it fails as expected: at α = 0.30 and 0.40 the realised error is 0.54 and 0.76.
- At α = 0.30 the HotpotQA bound is violated in 14% of resplits, above the nominal 10%. This is post hoc, but it
  means the guarantee is not tight here.

## E6: menu A extensions (HotpotQA, descriptive)

Menu A is Jonathan's two-stage menu: Qwen2.5-7B or 72B to retrieve, then Llama-3.2-3B or 3.1-70B to answer. The
judges are 3-seed ensembles; SLR's label router scores 0.7242 and the best fixed path 0.6932.

| Judge | AIQ | vs the two-question judge (0.7171) | vs SLR label router (both priced) |
|---|---|---|---|
| **One 4-option question, with the WW answer (qpa)** | **0.7210** | **+, [+0.002, +0.007]** | **[−0.009, +0.002]** |
| One 4-option question, passages only (qp) | 0.7177 | tie, [−0.001, +0.002] | [−0.013, −0.001] |
| Trained on the retrieval stage only, used for both | 0.7162 | tie, [−0.002, +0.001] | [−0.014, −0.003] |
| Trained on the answer stage only, used for both | 0.6997 | −, [−0.023, −0.012] | [−0.032, −0.018] |

**Reading:**
- **The joint question is the best nano-jev judge on menu A,** and the first whose interval against SLR includes
  zero.
  - It is still not shown equivalent.
  - Against his reward router the interval is [−0.010, +0.000].
  - Its single seeds score 0.7176–0.7205, against SLR's 0.7242.
- **The passages-only joint question equals the two-question judge** with one call made before any answer exists.
- **Retrieval decisions transfer to the answer stage; answer decisions do not transfer back.**
- **Operating points:** with his selection rules on val, every nano-jev judge, including the joint one, puts its peak
  at "escalate everything" (76.77 at 1,009, weak-first). Its parity point is within 15 of that. His label router
  saves money there: 76.70 at 835. A higher AIQ has not yet bought a cheaper operating point.

## E7: which ranker makes cheap evidence plus sampling work (descriptive)

**Setup:** Llama-3.2-3B reads 5 pool passages chosen by each ranker, with 1 sample or 20 fused. The test set is
`slr-test` (1,481). Accuracy is the lenient ruler; cost is AgentTTS units, each cell paying its own ranker
([compare-test.json](../../results/jev/e7-evidence/answers/compare-test.json)).

| Ranker | All gold in its top 5 | 1 sample | 20 fused | Gain from fusing |
|---|---|---|---|---|
| gold oracle | 1.000 | 0.714 | **0.860** | +0.146 |
| **nano-jev** | 0.758 | 0.614 | **0.751** | +0.136 |
| MS MARCO MiniLM | 0.652 | 0.556 | 0.693 | +0.137 |
| BM25 | 0.438 | 0.490 | 0.629 | +0.140 |
| *72B retrieval → 70B (experiment 8)* | | *0.762* | | |

**Paired against nano-jev with 20 fused samples** (95%):

| Cell | Difference |
|---|---|
| MS MARCO, 20 fused | −0.057 [−0.079, −0.036] |
| BM25, 20 fused | −0.122 [−0.147, −0.097] |
| gold, 20 fused | +0.109 [+0.088, +0.132] |
| gold, 1 sample | −0.037 [−0.061, −0.011] |
| 72B → 70B | +0.011 [−0.014, +0.036] |

**Reading:**
- **The ranker and sampling contribute roughly additively.** Fusing 20 samples adds about +0.14 whichever ranker
  chose the passages, and the rankers keep their order at both sample counts.
- **nano-jev is the best non-oracle ranker by a wide margin,** and the gold oracle shows another +0.11 to gain from
  better ranking.
- **Twenty samples on nano-jev's passages beat one sample on the gold passages.**
- **Cost:** in AgentTTS units, nano-jev with 20 samples costs 44.7 a question against 37.1 for the 72B → 70B
  pipeline. The "3.4% of the price" in [the golden-set write-up](2026-10-21-slr-standard.md) is at Table 5 prices
  (31.6 of 929), where the 72B retrieval call dominates.
- **Scoring:** the 3B cells answer in sentences (EM 0), so they are scored by lenient containment only. The ranker
  comparisons share one prompt; the comparison with the 70B crosses answer styles.

## E9: latency

The test times p50 per decision on real `slr-test` inputs ([results/jev/e9-latency](../../results/jev/e9-latency/)):

| Model | L40S, batch 1 | L40S, batch 32 (per decision) | 8 CPU threads, batch 1 |
|---|---|---|---|
| **nano-jev judge (33M)** | **6.4 ms** | **2.9 ms** | **123 ms** |
| MS MARCO MiniLM (33M) | 6.3 ms | 2.1 ms | 123 ms |
| NLI xsmall (71M) | 7.8 ms | 4.0 ms | 131 ms |
| Qwen2.5-1.5B router (his forward pass) | 11.5 ms | 16.3 ms | 4,787 ms |
| Strands Decider 2B | 38.6 ms | 51.5 ms | needs a GPU (Triton) |
| Strands Decider E4B | 47.7 ms | 121.3 ms | needs a GPU |

**Reading:**
- **On a GPU the gap is small:** nano-jev is 1.8× faster than the 1.5B router at batch 1 and 5.6× at batch 32.
- **On a CPU it is 39×**, and the 2B / 4B deciders cannot run there at all.
- So nano-jev's deployment case is CPU or edge hardware, not GPU latency.
- On the CPU, batch 32 is slower per decision than batch 1 (505 ms), probably because the batch pads every pair to
  its longest input (up to 512 tokens).

## Not done

These pre-registered items did not run, and nothing above depends on them.
- **E9's dollar re-pricing** of menu A with AgentTTS v2 Table 10 prices was not run.
- **H3's question-side baselines:**
  - The RouteLLM-style BERT question router and kNN on MiniLM embeddings were not run. The question-only judge (q) is
    the nearest stand-in: a 33M BERT-family model on the question alone.
  - HHEM was dropped before E3 ran (deviation 5).
  - Perplexity was not reported separately. It is exp(−mean log-prob), a monotone transform, so its AIQ equals mean
    log-prob's.
- **The secondary alias score** (MuSiQue's `answer_aliases`) was not computed. Every number uses the gold answer.
- **H3 on 2Wiki's 2,000-question test:** only H2 ran there.

## Caveats

1. **Dates.** The pre-registration says 2026-10-24 throughout, but it was committed at 12:13:37 SGT on 2026-10-23
   (`9767be5`), and its split hashes at 12:20:37 (`493bc07`). The first answers on either new dataset came later:
   the 50-question pilots (43207 / 43208) were submitted at 12:20:41 and the full runs (43212 / 43213) at 12:23. The
   order the pre-registration claims holds; only the date is wrong (deviation 8).
2. **H1c cannot isolate HotpotQA exposure.** noHotpot's SQuAD-built relevance decision treats "a same-article,
   on-topic paragraph" as partially relevant. HotpotQA's treats "a needed supporting paragraph" that way. So noHotpot
   scores 0.30 on v1.0's relevance test, and its arm differs from v1.0 in more than HotpotQA.
3. **Plain MiniLM barely trains under nano-jev's settings** on HotpotQA and MuSiQue. H1a's gap may partly reflect
   settings that suit a decision-trained start. The MS MARCO arm, which trains normally, is the fairer comparison,
   and nano-jev also beats it.
4. **H4a on HotpotQA** uses the calib-chosen score (not-grounded, 77%), as pre-registered, not the test's best
   (not-correct, 85%).
5. **Bootstrap replicates.** The confirmatory H1 curves first ran with 1,000 replicates (43235 / 43236), against the
   5,000 pre-registered. The rerun at 5,000 (43266 / 43267) gave the same conclusions. Both are kept
   (`label-curve-boot1000/` and `label-curve/`).
6. **The confirmatory E1 cells have 5 seeds at every budget,** not the pre-registered 10 at budgets up to 500
   (deviation 8).
7. **Menu A compares ensembles.** The nano-jev judges are 3-seed ensembles against one SLR router, though the
   ensemble still costs well under 1% of a 1.5B call.
8. **2Wiki's primary test is 500 questions,** so its intervals are wide. The 2,000-question secondary test agrees on
   every H2 test.
9. **Contamination:**
   - MuSiQue is clean for nano-jev: it appears only in nano-jev's held-out evaluation.
   - The Strands deciders were trained on data that includes MuSiQue, and no decider number enters a confirmatory
     test.

## What this means for the release and the central question

**Claims the evidence supports, for the model card:**
- **A better starting point for a small router.** Fine-tuned from nano-jev, a 33M router beats the same architecture
  fine-tuned from plain MiniLM or from MS MARCO. This is confirmed on MuSiQue and 2Wiki for qp.
- **Text-only routing that matches white-box signals.** The fine-tuned judge is non-inferior to the weak model's
  probability, margin, P(True), an NLI check and a probe on its hidden states, without access to its weights.
  Confirmed.
- **Zero-label routing.** One untrained typed question recovers 77–86% of a trained judge's gain over random.
  Confirmed.
- **Transfer across model pairs** without retraining, including a Llama pair: exploratory, 2 of 3 pairs.
- **CPU deployment:** 123 ms a decision on 8 threads, 39× faster than a 1.5B router.

**Claims to avoid:**
- "better than entropy / logistic regression" (it ties);
- "fewer labels" (H1b fails);
- "the typed head matters" (H1d fails);
- "question-only routing is at chance" (HotpotQA only);
- "risk-controlled escalation" (trivial here);
- "beats the 1.5B SLR router" (E6 comes closest, and is still not equivalent).

**The published weights stay 43118's.** No starting point beat nano-jev v1.0. The repro equals it, and the E6
joint judge is a new judge set on menu A, not new weights for the qp / qpa release.
