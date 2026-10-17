# Experiment 6: multi-stage cascades

The next step from [experiments 3–5](2026-10-06-experiments-3-5.md). It chains the cached experts of
experiments 1–4 into cascades of two to four stages, on CPU only, and compares them on the same
7,405 test questions.

**Result:** the three-stage cascade **1.5B (top 2) → 7B (top 3) → 14B (all 10)** scores **F1
0.726 / EM 0.584 at 14,243 GFLOPs**. That is above always-14B (0.719 / 0.571, p = 0.02) at **32%
of its cost**. Its AIQ is 0.721, against 0.695 for the best two-stage cascade into the 14B
(difference CI [+0.021, +0.030]). A fourth stage adds nothing. In all 8 cascades the learned
routers significantly beat the entropy threshold.

## Setup

**Experts.** All come from earlier runs, which share questions, splits and nano-jev evidence (as
checked in the experiments 3–5 validation).

| Name | Model, paragraphs | Run | Test F1 | EM | GFLOPs |
|---|---|---|---|---|---|
| `small` | Qwen2.5-1.5B, top 2 | `results/gpu` (exp. 1) | 0.613 | 0.483 | 1,051 |
| `mid3` | Qwen2.5-7B, top 3 | `results/gpu-large-top3` (3a) | 0.692 | 0.555 | 7,292 |
| `mid5` | Qwen2.5-7B, top 5 | `results/gpu-large-top5` (3b) | 0.700 | 0.558 | 11,598 |
| `large7` | Qwen2.5-7B, all 10 | `results/gpu` (exp. 1) | 0.669 | 0.530 | 22,762 |
| `large14` | Qwen2.5-14B, all 10 | `results/gpu-large-14b` (4) | 0.719 | 0.571 | 43,906 |

The 3B from experiment 5 is left out. A cascade must start at the cheapest expert for AIQ to be
comparable (below), and the 3B on its own was barely better than the 1.5B.

**Cascade.** Every question is answered by the first stage. After each stage, a router decides
whether to escalate to the next. A question pays for every stage it reaches (no shared cache
between calls) and keeps the last answer.

**Routers.** Router *i* is a logistic regression with C tuned by 5-fold CV, as before. Its
features are:

- the 5 nano-jev evidence features;
- the answer signals of every stage so far: mean, max and first-token entropy, answer length,
  and yes/no;
- the **agreement** between consecutive answers (token F1 between them).

Its label is *a later stage helps*: stage *i*'s F1 < τ = 0.8 and some later stage's F1 ≥ τ. The
`entropy` variant uses each stage's mean token entropy instead. It is the multi-stage form of the
small-entropy baseline.

**No tuning on test.**

1. Each router's candidate thresholds are its calib-score quantiles at escalation rates 0, 0.05,
   …, 1. That gives up to 21^(stages−1) combinations.
2. Every combination is scored on calib.
3. Only the calib cost–F1 frontier is evaluated on test.
4. AIQ is computed over [1,051, 43,906] GFLOPs, the cheapest to the most expensive expert, the
   same range for every cascade. So every cascade must start with `small`, or it would get credit
   below its own starting cost.
5. The operating point is the cheapest calib-frontier policy whose calib F1 is within 1% of the
   cascade's strongest single expert on calib.

**Sanity check.** The two-stage `small → large7` cascade reproduces experiment 2:

- its router has the same CV loss (0.42905) as experiment 2's `evidence+small / large-helps`;
- its entropy AIQ is 0.6620, against 0.6619 in experiment 2;
- its router AIQ is 0.6704, slightly below the 0.6718 of the test-swept curve. That is expected,
  because thresholds now come from calib.

(The table below reports 0.6767 for the same cascade, because AIQ there is taken over the wider
1,051–43,906 range.)

## Execution

| | |
|---|---|
| Command | `cascade` with 5 `--expert` and 8 `--cascade` options (in `results/cascade/provenance.jsonl`), then the same with `--exclude nanojev_validation_ids.txt` |
| Hardware | Local CPU, no GPU. 23 s including 1,000 bootstrap resamples |
| Output | `results/cascade/`: `report.json`, `report-clean.json`, `curves.csv`, `curves-clean.csv`, `run.log` |

## Results

Test set, 7,405 questions. The 95% intervals come from a 1,000-sample paired bootstrap over
questions. "vs best single" is operating-point F1 minus the F1 of the cascade's strongest
single expert.

| Cascade | Router AIQ [95% CI] | Entropy AIQ | Op. F1 | EM | GFLOPs | Stops at each stage | vs best single [95% CI] |
|---|---|---|---|---|---|---|---|
| small → large14 | 0.695 [0.687, 0.704] | 0.687 | 0.723 | 0.578 | 35,202 | 22 / 78% | +0.004 [+0.001, +0.008] |
| small → mid5 → large14 | 0.717 [0.708, 0.725] | 0.710 | 0.723 | 0.580 | 16,531 | 21 / 65 / 14% | +0.005 [−0.002, +0.012] |
| **small → mid3 → large14** | **0.721** [0.713, 0.729] | 0.713 | **0.726** | **0.584** | **14,243** | 11 / 74 / 15% | **+0.007 [+0.001, +0.015]** |
| small → mid3 → mid5 → large14 | 0.721 [0.713, 0.729] | 0.713 | 0.720 | 0.578 | 13,313 | 0 / 75 / 20 / 5% | +0.002 [−0.006, +0.010] |
| small → mid5 | 0.697 [0.688, 0.706] | 0.693 | 0.700 | 0.562 | 8,098 | 39 / 61% | −0.001 [−0.005, +0.004] |
| small → mid3 | 0.689 [0.680, 0.698] | 0.687 | 0.688 | 0.555 | 5,686 | 36 / 64% | −0.003 [−0.008, +0.001] |
| small → large7 | 0.677 [0.668, 0.685] | 0.668 | 0.670 | 0.534 | 8,044 | 69 / 31% | 0.000 [−0.008, +0.008] |
| small → mid5 → large7 | 0.703 [0.695, 0.711] | 0.697 | 0.703 | 0.565 | 8,420 | 44 / 52 / 4% | +0.003 [−0.003, +0.009] |

The oracle, which sends each question to the cheapest expert with the best F1, reaches 0.842 F1 at
5,343 GFLOPs. It would answer 70% of questions with `small`.

**Router minus entropy (paired AIQ):** the router wins in all 8 cascades, with every lower CI
bound above zero. The gain ranges from [+0.001, +0.004] for small → mid3 to [+0.005, +0.012]
for small → large7.

**Against the two-stage cascade into the 14B (small → large14, paired AIQ):**

| Cascade | AIQ difference [95% CI] | p(≤0) |
|---|---|---|
| small → mid3 → large14 | [+0.021, +0.030] | < 0.001 |
| small → mid3 → mid5 → large14 | [+0.021, +0.031] | < 0.001 |
| small → mid5 → large14 | [+0.017, +0.026] | < 0.001 |
| small → mid5 → large7 | [+0.002, +0.014] | 0.001 |
| small → mid5 | [−0.005, +0.008] | 0.286 |

Without nano-jev's 1,000 validation questions (`report-clean.json`, 6,405 questions), every AIQ
moves by 0.003 or less.

## Findings

1. **A middle stage is worth more than a bigger last stage.** Adding the 7B on the top 3 between
   the 1.5B and the 14B raises AIQ by 0.025. At the operating point it beats always-14B on F1 and
   EM at 32% of its cost, and 60% less than the two-stage cascade's operating point (35,202
   GFLOPs) at slightly higher F1. Most questions stop at the 7B (74%), and only 15% need the 14B.
2. **The cheaper middle expert is the better one.** `mid3` and `mid5` give similar AIQ (0.721 and
   0.717). `mid3` reaches the higher operating point at lower cost. The 14B covers most of the
   cases where the top 3 misses a gold paragraph.
3. **A fourth stage adds nothing.** `small → mid3 → mid5 → large14` ties the three-stage cascade
   on AIQ. Its operating point skips `small` entirely and lands lower on F1.
4. **Learned routers beat entropy at every stage.** The multi-stage routers also see the answer
   agreement between stages. They beat the per-stage entropy threshold in every cascade, by 0.002
   to 0.009 AIQ.
5. **Large headroom remains.** The best cascade (0.726) is 0.12 F1 below the five-expert oracle
   (0.842). The oracle shows the 1.5B already has the best answer for 70% of questions, but the
   best router stops there for only 11%.

## Caveats

- **The thresholds are chosen from up to 9,261 combinations on 2,000 calib questions.** That is
  honest with respect to test, but the bootstrap covers only test-set sampling. Calib selection
  and router retraining would add variance.
- **Each stage is a separate call.** The 7B on the top 3 and the 14B on all 10 both pay full
  prompt cost. A real system could reuse prompt prefixes, so cost is a conservative upper bound.
- **Scorer cost (about 400 GFLOPs) is left out,** as before. Every cascade here starts with
  `small`, which needs it, so the comparison between cascades is unaffected.
- **These are offline cascades.** Answers come from caches computed with the same prompt and
  greedy decoding. A deployed cascade would give the same answers, but latency adds up across
  stages.

## Reproducing

```bash
uv run cascade \
  --expert small=results/gpu:small --expert mid3=results/gpu-large-top3:large \
  --expert mid5=results/gpu-large-top5:large --expert large7=results/gpu:large \
  --expert large14=results/gpu-large-14b:large \
  --cascade small,large14 --cascade small,mid5,large14 --cascade small,mid3,large14 \
  --cascade small,mid3,mid5,large14 --cascade small,mid5 --cascade small,mid3 \
  --cascade small,large7 --cascade small,mid5,large7 --out results/cascade
# the same with --exclude data/hotpotqa/nanojev_validation_ids.txt for report-clean.json
```
