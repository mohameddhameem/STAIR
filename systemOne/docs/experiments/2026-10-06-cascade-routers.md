# Experiment 2: cascade routers, significance and τ sensitivity

A CPU-only follow-up to [experiment 1](2026-10-05-omega-full-run.md). It reuses that run's
cached expert outputs (`runs/gpu`), so no model was rerun. It does three things:

1. Adds **cascade routers**: they decide after the small expert has answered and also see its
   answer signals.
2. Adds **paired bootstrap confidence intervals** to every AIQ and operating point.
3. Retrains every router at **τ from 0.5 to 1.0**.

**Result:** the cascade routers are the first to beat the small-model entropy baseline
significantly: AIQ 0.672 against 0.662, 95% CI of the difference [+0.006, +0.015]. One of
them is also the first operating point to **beat always-large on F1** (0.676 against 0.669,
p = 0.009) at 56% of its cost. The pre-generation routers from experiment 1 do not differ
significantly from the entropy baseline. τ barely matters for the `large-helps` routers.

## Setup

Data, experts, splits, costs and the routers' model and tuning are those of experiment 1. Only
the following changed.

**Cascade routers.** Two new input sets, each trained on both labels, make four new routers.
Each runs after the small expert, so an escalated question pays for both experts, exactly
like the entropy baseline.

| Input set | Features |
|---|---|
| `evidence+small` | the 5 evidence features, plus 5 from the small expert's answer |
| `question+evidence+small` | the same, plus question text and the 5 question cues |

The 5 small-expert features:

- mean token entropy;
- maximum token entropy;
- entropy of the first token;
- number of generated tokens;
- whether the answer is "yes" or "no".

The last one is the small expert's *prediction*, not the gold answer. It flags yes/no questions
almost perfectly: 431 of the 447 questions it answers yes/no have a yes/no gold answer. On
those questions the large expert gains 0.10 F1 (0.803 against 0.698).

**Bootstrap.** `eval-routing --bootstrap 1000 --seed 0`, a paired bootstrap over the 7,405
test questions. Router scores are fixed: each resample re-ranks them and recomputes every curve.
The intervals therefore cover test-set sampling, not router retraining. `p(≤0)` is the share of
resamples in which a difference is ≤ 0.

**τ sweep.** `tau-sweep --run runs/gpu-cascade` retrains all 10 routers at τ = 0.5, 0.6, 0.7,
0.8, 0.9 and 1.0. τ changes only the routers' training labels: the experts, the baselines and
the oracle stay fixed. At τ = 0.8 the sweep reproduces `report.json` exactly.

## Execution

| | |
|---|---|
| Run directory | `runs/gpu-cascade`, a copy of `runs/gpu`'s caches and config. `runs/gpu` is untouched |
| Hardware | Local CPU, 20 threads, no GPU |
| Commands | `train-router --run runs/gpu-cascade --tau 0.8` (4 min), `eval-routing` with and without `--exclude` (12 s each), `tau-sweep` (23 min) |
| Check | With experiment 1's `routers.pkl`, the new evaluation code reproduces experiment 1's `report.json`, `report-clean.json` and `curves.csv` to within 1e-10 |

Cross-validation for the C grid now runs the fits in parallel (`n_jobs=-1`). Training all
10 routers takes 4 minutes, against 23 minutes for 6 routers before. The chosen C values do not
change.

## Results

Test set, 7,405 questions, cost in GFLOPs per question. The 95% intervals come from the
bootstrap. Experiment 1's rows are unchanged.

| Method | AIQ [95% CI] | CPT 80% | Operating point: F1 | EM | GFLOPs | Sent to large |
|---|---|---|---|---|---|---|
| Always small | | | 0.613 | 0.483 | 1,051 | 0% |
| Always large | | | 0.669 | 0.530 | 22,762 | 100% |
| Random | 0.641 [0.633, 0.649] | 18,419 | | | | |
| Small-entropy threshold (post) | 0.662 [0.653, 0.671] | 7,861 | | | | |
| evidence / large-helps | 0.663 [0.654, 0.672] | **5,466** | 0.667 | 0.529 | 8,503 | 34% |
| question+evidence / large-helps | 0.665 [0.657, 0.675] | 6,454 | 0.660 | 0.525 | 5,933 | 23% |
| **evidence+small / small-fails** | 0.666 [0.658, 0.675] | 6,824 | **0.676** | **0.541** | 12,657 | 50% |
| question+evidence+small / small-fails | 0.666 [0.658, 0.675] | 6,808 | 0.670 | 0.534 | 9,886 | 39% |
| **evidence+small / large-helps** | 0.672 [0.663, 0.680] | 5,647 | 0.667 | 0.531 | 7,135 | 27% |
| **question+evidence+small / large-helps** | **0.672** [0.665, 0.682] | 5,619 | 0.665 | 0.529 | 6,340 | 23% |
| Oracle | 0.755 [0.746, 0.763] | 5,638 | 0.772 | 0.631 | 5,638 | 21% |

On the clean test set (6,405 questions, without nano-jev's validation questions), every AIQ
moves by 0.002 or less. For example, question+evidence+small / large-helps scores 0.6725 and
small-entropy 0.6624.

### AIQ against the small-entropy baseline (paired)

| Router | AIQ difference, 95% CI | p(≤0) |
|---|---|---|
| question / small-fails | [−0.024, −0.017] | 1.000 |
| question / large-helps | [−0.019, −0.009] | 1.000 |
| question+evidence / small-fails | [−0.010, −0.002] | 0.997 |
| evidence / small-fails | [−0.006, +0.003] | 0.768 |
| evidence / large-helps | [−0.004, +0.005] | 0.438 |
| question+evidence / large-helps | [−0.002, +0.009] | 0.098 |
| question+evidence+small / small-fails | [+0.001, +0.007] | 0.005 |
| evidence+small / small-fails | [+0.002, +0.007] | 0.001 |
| evidence+small / large-helps | [+0.007, +0.013] | < 0.001 |
| question+evidence+small / large-helps | [+0.006, +0.015] | < 0.001 |

### Operating point F1 minus always-large F1 (paired)

| Router | 95% CI | p(≤0) |
|---|---|---|
| **evidence+small / small-fails** | **[+0.001, +0.014]** | **0.009** |
| question+evidence+small / small-fails | [−0.007, +0.008] | 0.476 |
| evidence / large-helps | [−0.010, +0.006] | 0.738 |
| evidence+small / large-helps | [−0.011, +0.006] | 0.751 |
| question+evidence+small / large-helps | [−0.013, +0.004] | 0.866 |
| question+evidence / large-helps | [−0.018, −0.001] | 0.980 |

All other routers are significantly below always-large at their operating point.

### By question type at the operating point (F1, share sent to large)

| Router | bridge | comparison | span | yes/no |
|---|---|---|---|---|
| Always large | 0.664 | 0.689 | 0.659 | 0.823 |
| evidence / large-helps (exp. 1) | 0.679, 40% | 0.617, 11% | 0.665, 36% | 0.697, 12% |
| evidence+small / small-fails | 0.683, 55% | 0.650, 31% | 0.670, 52% | 0.766, 22% |
| evidence+small / large-helps | 0.674, 28% | 0.639, 20% | 0.661, 26% | 0.755, 38% |

### τ sensitivity (AIQ)

| Router | τ 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| question+evidence+small / large-helps | 0.6727 | 0.6721 | 0.6723 | 0.6724 | 0.6721 | 0.6725 |
| evidence+small / large-helps | 0.6713 | 0.6709 | 0.6715 | 0.6718 | 0.6711 | 0.6708 |
| question+evidence+small / small-fails | 0.6714 | 0.6690 | 0.6662 | 0.6659 | 0.6647 | 0.6644 |
| evidence+small / small-fails | 0.6695 | 0.6682 | 0.6669 | 0.6664 | 0.6654 | 0.6649 |
| question+evidence / large-helps | 0.6652 | 0.6653 | 0.6658 | 0.6653 | 0.6660 | 0.6659 |
| evidence / large-helps | 0.6623 | 0.6629 | 0.6624 | 0.6626 | 0.6627 | 0.6625 |
| question+evidence / small-fails | 0.6619 | 0.6594 | 0.6564 | 0.6559 | 0.6559 | 0.6563 |
| evidence / small-fails | 0.6608 | 0.6607 | 0.6606 | 0.6606 | 0.6605 | 0.6605 |
| question / large-helps | 0.6469 | 0.6466 | 0.6475 | 0.6478 | 0.6487 | 0.6490 |
| question / small-fails | 0.6436 | 0.6428 | 0.6413 | 0.6413 | 0.6413 | 0.6413 |

The baselines do not depend on τ: random 0.6413, small-entropy 0.6619, oracle 0.7548. Every
change across τ is smaller than the bootstrap interval widths (about ±0.008).

## Findings

1. **The small expert's own answer is the strongest routing signal.** Adding it lifts AIQ from
   0.663–0.665 to 0.672. That beats the entropy baseline significantly for all four cascade
   routers. Without it, no router is significantly better than the entropy baseline: experiment
   1's +0.003 is noise (CI [−0.002, +0.009]).
2. **A cascade can beat always-large.** evidence+small / small-fails scores F1 0.676 and EM 0.541
   at 56% of always-large's cost, significantly above always-large (p = 0.009). It escalates
   50%, keeping the small expert's answer whenever the router predicts it is right. Those are
   the questions where reading only the top-2 paragraphs beats reading all 10.
3. **Pre-generation routing is still cheapest at moderate quality.** evidence / large-helps
   recovers 80% of the F1 gap at 5,466 GFLOPs, against about 5,620 for the best cascades. A
   cascade always pays for the small expert. Pre-generation wins at low budgets, cascades win
   on the curve as a whole (AIQ).
4. **Yes/no questions improve most.** Yes/no F1 at the operating point rises from 0.697
   (evidence / large-helps) to 0.755–0.766, mainly through the small expert's yes/no answer.
   Comparison questions are still below always-large (0.639–0.650 against 0.689).
5. **τ is not a sensitive choice.** Across τ = 0.5–1.0, the evidence-based `large-helps` routers
   vary by at most 0.001 AIQ, and question / large-helps by 0.002. `small-fails` routers prefer a low τ (0.5), because a high τ also escalates
   near-misses the large expert does not fix. That is the same reason `large-helps` wins in
   experiment 1.
6. **The oracle gap is still large:** 0.672 against 0.755.

## Caveats

- **AIQ against random is biased upward.** AIQ takes a curve's upper hull, so even a random
  ranking scores at or above the random baseline's straight line. That is why question /
  small-fails ties random exactly. Differences between two ranked policies, such as a router
  and small-entropy, are not affected.
- **The scorer's cost is still left out of the curves.** It is about 400 GFLOPs per question.
  It applies equally to the routers and the small-entropy baseline, so those comparisons hold.
  Always-large does not need it.
- **The bootstrap does not cover router retraining.** Retraining with other CV seeds or
  router_train samples would add variance.
- **The operating point is chosen on calib, within 1% of always-large.** For the cascade
  routers it lands above always-large on test. That threshold is not tuned on test.

## Reproducing

```bash
cp runs/gpu/{router_train,calib,test}.parquet runs/gpu/config.toml runs/gpu-cascade/
uv run train-router --run runs/gpu-cascade --tau 0.8
uv run eval-routing --run runs/gpu-cascade                     # report.json, with --bootstrap 1000
uv run eval-routing --run runs/gpu-cascade --exclude data/hotpotqa/nanojev_validation_ids.txt
uv run tau-sweep --run runs/gpu-cascade                        # tau-sweep.csv / .json
```

No GPU needed: all steps read the cached expert outputs. On omega,
`RUN=runs/gpu bash scripts/run_cluster.sh eval` retrains and re-evaluates in place, overwriting
experiment 1's `routers.pkl` and reports.
