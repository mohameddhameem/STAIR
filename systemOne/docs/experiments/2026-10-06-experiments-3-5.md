# Experiments 3–5: changing one expert at a time

These are the four omega jobs planned in [gpu-jobs.md](../gpu-jobs.md). Each changes one expert of
[experiment 1](2026-10-05-omega-full-run.md) and reuses everything else. Routers, cascade
features, the bootstrap and the operating-point rule are as in
[experiment 2](2026-10-06-cascade-routers.md). The runs took place on 2026-10-06.

**Results:**
- **The 7B reads better with fewer paragraphs.** On nano-jev's top 3 it scores 0.692 F1 at a
  third of the cost of reading all 10 (0.669). On the top 5 it scores 0.700 at half the cost.
  The distractor paragraphs actively hurt it.
- **A 14B large expert widens the gap** (0.719 F1) and gives routing the most room.
- **A 3B small expert hardly helps on its own** (+0.005 F1 for twice the cost), though routers
  built on its outputs work well.
- **In every setting, the cascade routers significantly beat the small-entropy baseline.**

## Runs

| Exp. | Run directory | Change from experiment 1 | Job | GPU time | Stage that ran |
|---|---|---|---|---|---|
| 3a | `runs/gpu-large-top3` | 7B reads nano-jev's top 3 paragraphs | 42217 | 36 min | large |
| 3b | `runs/gpu-large-top5` | 7B reads the top 5 | 42219 | 52 min | large |
| 4 | `runs/gpu-large-14b` | Qwen2.5-14B-Instruct (revision `cf98f3b3`) replaces the 7B, all 10 paragraphs | 42224 | 2 h 58 min | large |
| 5 | `runs/gpu-small-3b` | Qwen2.5-3B-Instruct (revision `aa8e7253`) replaces the 1.5B, top 2 | 42221 | 17 min | small |

- **Submission:** every job was `qsub -v STEP=full,CONFIG=...,RUN=...,REUSE=runs/gpu scripts/omega.pbs`,
  with a backup job (`afternotok`).
- **Outcome:** all four exited 0 on the first attempt, and no backup job ran. The logs show no
  errors.
- **Reuse:** each log shows `copied` for both unchanged stages in all three splits (24 + 4 + 15
  chunks each) and `recomputing` only for the changed expert. The copied expert's F1 matches
  experiment 1 exactly.
- **Code:** commit `f9de95e`. Provenance marks the runs as having uncommitted changes
  (`git_dirty`) because of the then-untracked `scripts/export_results.sh`. No code the runs used
  had changed.

Mean time per question and prompt tokens for the new experts:

| Expert | Paragraphs | Prompt + output tokens | Seconds/question |
|---|---|---|---|
| 7B (exp. 1) | 10 | 1,494 | 0.248 |
| 7B (3a) | top 3 | 479 | 0.086 |
| 7B (3b) | top 5 | 761 | 0.131 |
| 14B (4) | 10 | 1,493 | 0.483 |
| 1.5B (exp. 1) | top 2 | 341 | 0.020 |
| 3B (5) | top 2 | 341 | 0.032 |

## Results

These are on the test set (7,405 questions), cost in GFLOPs per question. 95% intervals are from
a 1,000-sample paired bootstrap.

| Exp. | Always small: F1 / GFLOPs | Always large: F1 / GFLOPs | Oracle: F1 / GFLOPs | AIQ: random / entropy / best router / oracle |
|---|---|---|---|---|
| 1, 2 | 0.613 / 1,051 | 0.669 / 22,762 | 0.772 / 5,638 | 0.641 / 0.662 / 0.672 / 0.755 |
| 3a | 0.613 / 1,051 | **0.692 / 7,292** | 0.759 / 2,277 | 0.653 / 0.665 / 0.669 / 0.745 |
| 3b | 0.613 / 1,051 | **0.700 / 11,598** | 0.773 / 3,294 | 0.657 / 0.675 / 0.682 / 0.756 |
| 4 | 0.613 / 1,051 | 0.719 / 43,906 | 0.794 / 11,174 | 0.666 / 0.687 / 0.697 / 0.772 |
| 5 | 0.618 / 2,105 | 0.669 / 22,762 | 0.769 / 6,320 | 0.644 / 0.671 / 0.681 / 0.753 |

Best router per experiment, by AIQ, and the gain over the entropy baseline (paired):

| Exp. | Best router | AIQ [95% CI] | Gain over entropy [95% CI] | Operating point: F1 / EM / GFLOPs / sent to large |
|---|---|---|---|---|
| 2 | question+evidence+small / large-helps | 0.672 [0.664, 0.682] | [+0.006, +0.015] | 0.665 / 0.529 / 6,340 / 23% |
| 3a | question+evidence+small / large-helps | 0.669 [0.660, 0.678] | [+0.000, +0.006] | 0.682 / 0.548 / 5,007 / 54% |
| 3b | question+evidence+small / large-helps | 0.682 [0.673, 0.690] | [+0.004, +0.010] | 0.681 / 0.545 / 4,958 / 34% |
| 4 | question+evidence+small / large-helps | 0.697 [0.689, 0.705] | [+0.007, +0.013] | 0.713 / 0.572 / 23,043 / 50% |
| 5 | evidence+small / large-helps | 0.681 [0.673, 0.690] | [+0.006, +0.013] | 0.671 / 0.536 / 6,384 / 19% |

The best operating-point F1 in each run is 3a 0.688 (evidence+small / large-helps, 5,646
GFLOPs), 3b 0.697 (same router, 7,692), 4 0.723 (same router, 34,259) and 5 0.674
(evidence / small-fails, 7,221).

Without nano-jev's 1,000 validation questions (`report-clean.json`, 6,405 questions), every best
router's AIQ moves by 0.002 or less.

**Comparing AIQ across experiments:** AIQ is the area under the curve between each run's own
two experts, so it is only comparable *within* a row. Compare F1 and GFLOPs across experiments.

## Findings

1. **Distractors hurt the 7B.** With 3 paragraphs selected by nano-jev, it gains 0.023 F1 and
   costs 68% less than with all 10. With 5 paragraphs it gains 0.031 F1 at 51% of the cost. This
   happens even though the gold pair is missing from the top 3 for 12% of validation questions.
   The top-5 7B (0.700 F1, 11,598 GFLOPs) beats experiment 2's best operating point (0.676 F1,
   12,657 GFLOPs) at a similar cost, without any routing.
2. **A better large expert leaves the routing problem harder.** In 3a and 3b, the oracle
   escalates the same ~20% of questions as before, but always-large is now so cheap and strong
   that the routers' operating points fall short of it: 0.682 against 0.692 in 3a. The gain over
   entropy also shrinks: in 3a its interval only just excludes zero (p = 0.017).
3. **The 14B has the most routing headroom.** It scores 0.719 F1 at twice the 7B's cost, and
   `large-helps` rises to 17.8% of questions. Its best router reaches 0.713 F1 at 52% of
   always-14B's cost, with a gain over entropy (+0.010 AIQ) as large as experiment 2's.
4. **Doubling the small expert buys almost nothing directly.** The 3B scores 0.618 F1 against
   0.613 for the 1.5B, at twice the GFLOPs. But its entropy is a better signal: the entropy
   baseline's AIQ rises from 0.662 to 0.671, and the best router sends only 19% of questions to
   the large expert.
5. **Cascade routers are the strongest family in all five settings.** In every run, the best
   router uses the small expert's own output signals (`+small`), and it significantly beats the
   entropy threshold.

## Next steps

The three-expert cascade is done in [experiment 6](2026-10-06-multi-stage-cascade.md).

- **Combine the best pieces:** the 1.5B or 3B on the top 2, then the 7B on the top 5, then the 14B
  on the top 5. Everything except a 14B-on-top-5 run is already cached, so a three-expert
  cascade can be tested on CPU.
- **Run the 14B on the top 5 paragraphs** (one GPU job, about 1.5 h). If finding 1 holds for the
  14B too, it is the strongest and cheapest large expert.

## Files

The results for each run are in `results/<run directory name>/` (see
[results/README.md](../../results/README.md)): reports, curves, per-question caches, config,
provenance and the job log.
