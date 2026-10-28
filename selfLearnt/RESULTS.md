# Results so far (cluster run, seed 0)

HotpotQA, the team test set (AgentTTS's 500 questions, dev [500:1000]); accuracy = lenient (team standard, AgentTTS's
ruler); cost = AgentTTS units per question, counted on the tokens actually processed (`stair_common.cost`). Seed 0 only:
with 500 questions an accuracy carries about ±2 points of sampling noise. The per-question tables (`results/common/`
layout) are in [`results/common/`](../results/common/) on this branch, with the fine-tuning loss curves in
[`results/common/history/`](../results/common/history/).

These are on AgentTTS's 500 test questions (`test` = dev[500:1000]), which the team keeps for comparing with the paper.
Since 21 Oct (`common` 0.3.0) the team's HotpotQA headline is Jonathan's golden set (`slr-test`, 1,481 questions,
priced at AgentTTS's Table 5). The golden-set runs have their own experiment IDs, sheets and files, so the two are never
mixed: AL-07 = AL-01 (AgentTTS), AL-08 = AL-02 (always-strong, Jonathan's rows, already in:
[`jonathan-ss-hotpotqa-slr-test.csv`](../results/common/jonathan-ss-hotpotqa-slr-test.csv), 0.7677), AL-09 / AL-10 /
AL-11 = AL-04 / AL-05 / AL-06 (results `AL-09`..`AL-11-hotpotqa-slr-test-seed0.csv`, curves
`history/selfLearnt-golden-*`). They are running; results expected early November. AL-03 is not repeated.

| Experiment | Policy | Accuracy | Units / question |
|---|---|---|---|
| AL-01 AgentTTS | **paper-best** (their reported setting: Qwen2.5 72B x1 -> Llama 3.2 3B x40 + fusion) | **0.744** | 1,984 |
| AL-01 AgentTTS | searched (their search re-run, Qwen2.5 72B as planner instead of o3-mini) | 0.638 | 3,790 |
| AL-02 | always-strong, the team's shared reference (Qwen2.5 72B -> Llama 3.1 70B, one greedy pass each, AWQ 4-bit, AgentTTS's prompts, no fuser; Jonathan's stored run, `jonathan-ss-hotpotqa-test.csv`) | 0.776 | 1,829 |
| AL-03 | fixed: Qwen 7B -> Llama 3B (untrained, our prompts, one pass) | 0.400 | 181 |
| AL-03 | fixed: Qwen 7B -> Llama 8B | 0.494 | 190 |
| AL-03 | fixed: Qwen 7B -> Llama 70B | 0.584 | 276 |
| AL-03 | fixed: Qwen 32B -> Llama 3B / 8B / 70B | 0.564 / 0.606 / 0.678 | 805 / 815 / 913 |
| AL-03 | fixed: Qwen 72B -> Llama 3B / 8B / 70B | 0.614 / 0.690 / **0.730** | 1,808 / 1,814 / 1,875 |
| AL-03 | rulebook, starting rules (hidden-state probes) | 0.512 | 531 |
| AL-03 | rulebook, learnt rules | 0.566 | 609 |
| AL-03 | oracle within the rulebook's choices | 0.646 | 243 |
| AL-04 | fixed: Qwen 7B -> Llama 3B (both fine-tuned) | 0.818 | 175 |
| AL-04 | fixed: Qwen 7B -> Llama 8B (fine-tuned) | **0.848** | 178 |
| AL-04 | fixed: Qwen 7B -> Llama 70B (fine-tuned) | **0.854** | 219 |
| AL-04 | fixed: Qwen 32B -> Llama 3B / 8B / 70B (fine-tuned) | 0.804 / 0.834 / 0.840 | 793 / 796 / 837 |
| AL-04 | fixed: Qwen 72B -> Llama 3B / 8B / 70B (fine-tuned) | 0.800 / 0.816 / 0.820 | 1,782 / 1,785 / 1,826 |
| AL-04 | rulebook, starting rules = learnt rules (no change was accepted) | 0.830 | 437 |
| AL-04 | oracle within the rulebook's choices | 0.894 | 183 |
| AL-05 | fixed: nano-jev 33M -> Llama 1B / 3B / 8B (fine-tuned answerers) | 0.616 / 0.698 / 0.722 | 4 / 6 / 12 |
| AL-05 | fixed: Qwen2.5 0.5B -> Llama 3B / 8B (fine-tuned) | 0.730 / 0.766 | 14 / 18 |
| AL-05 | fixed: Qwen3.5 0.8B -> Llama 3B / 8B / 70B (fine-tuned) | 0.796 / 0.826 / 0.842 | 22 / 25 / 65 |
| AL-05 | fixed: Qwen 7B -> Llama 8B / 70B (fine-tuned; also in AL-04) | 0.848 / 0.854 | 178 / 219 |
| AL-05 | rulebook, starting rules (8 retrievers from nano-jev, 4 answerers from Llama 1B) | 0.672 | 12 |
| AL-05 | rulebook, learnt rules | 0.710 | 13 |
| AL-05 | oracle within the rulebook's choices | 0.794 | 36 |
| AL-06 | fixed: Qwen2.5 0.5B -> Llama 1B / 3B / 8B (fine-tuned) | 0.672 / 0.730 / 0.766 | 13 / 14 / 18 |
| AL-06 | fixed: Qwen3.5 0.8B -> Llama 3B / 8B / 70B (fine-tuned) | 0.796 / 0.826 / 0.842 | 22 / 25 / 65 |
| AL-06 | rulebook, starting rules (7 retrievers from Qwen 0.5B, 4 answerers from Llama 1B) | 0.724 | 21 |
| AL-06 | rulebook, learnt rules | 0.756 | 22 |
| AL-06 | oracle within the rulebook's choices | 0.830 | 28 |

### Each experiment's own method and its best fixed pair (cost relative to AgentTTS's paper-best = 100%)

AL-03 .. AL-06 each test a learnt rulebook (the method) and also score every fixed retriever -> answerer pair on their
ladders as reference points. AL-05's and AL-06's ladders contain AL-04's fine-tuned models, so the same best pair tops
all three; what differs is the rulebook.

| Experiment | Method | Accuracy | Relative cost | Best fixed pair (oracles excluded) | Accuracy | Relative cost |
|---|---|---|---|---|---|---|
| AL-01 | AgentTTS paper-best | 0.744 | 100% | - | | |
| AL-02 | always-strong, Qwen 72B -> Llama 70B (team reference) | 0.776 | 92% | - | | |
| AL-03 | learnt rules, untrained models | 0.566 | 31% | Qwen 72B -> Llama 70B (untrained) | 0.730 | 95% |
| AL-04 | learnt rules, fine-tuned 3 + 3 models | **0.830** | 22% | Qwen 7B -> Llama 70B (fine-tuned) | **0.854** | 11% |
| AL-05 | learnt rules, ladders from nano-jev 33M | 0.710 | **0.6%** | the same pair | 0.854 | 11% |
| AL-06 | learnt rules, ladders from Qwen 0.5B | 0.756 | 1.1% | the same pair | 0.854 | 11% |

AL-05's full table (35 policies) is in [`AL-05-hotpotqa-test-seed0.csv`](../results/common/AL-05-hotpotqa-test-seed0.csv).

### Accuracy or savings: the cost-accuracy curve (AL-05)

AL-05 has the longest ladders (8 fine-tuned retrievers from nano-jev 33M to Qwen 72B, 4 answerers from Llama 1B to
70B), so its fixed pairs trace the whole trade-off. These are the pairs no other pair beats on both accuracy and cost
(the efficient frontier); every other pair is both less accurate and no cheaper than one of them.

| Pair (fine-tuned) | Accuracy | Units / question | vs AgentTTS cost | vs always-strong |
|---|---|---|---|---|
| nano-jev 33M -> Llama 1B | 0.616 | 4 | 0.2% | -16.0 pt |
| nano-jev 33M -> Llama 3B | 0.698 | 6 | 0.3% | -7.8 pt |
| nano-jev 33M -> Llama 8B | 0.722 | 12 | 0.6% | -5.4 pt |
| Qwen2.5 0.5B -> Llama 3B | 0.730 | 14 | 0.7% | -4.6 pt |
| Qwen2.5 0.5B -> Llama 8B | 0.766 | 18 | 0.9% | -1.0 pt |
| Qwen3.5 0.8B -> Llama 3B | 0.796 | 22 | 1.1% | +2.0 pt |
| **Qwen3.5 0.8B -> Llama 8B** | **0.826** | **25** | **1.3%** | **+5.0 pt** |
| **Qwen3.5 0.8B -> Llama 70B** | **0.842** | **65** | **3.3%** | **+6.6 pt** |
| Qwen 7B -> Llama 8B | 0.848 | 178 | 9.0% | +7.2 pt |
| Qwen 7B -> Llama 70B | 0.854 | 219 | 11.0% | +7.8 pt |

For reference: AgentTTS paper-best 0.744 at 1,984 units (100%); always-strong 0.776 at 1,829 (92%).

- **Which point to use depends on the budget.** Going down the curve saves a lot of cost for a modest loss: from the
  most accurate pair to Qwen 0.8B -> Llama 70B costs 70% less for 1.2 points; Qwen 0.8B -> Llama 8B costs 89% less
  for 2.8 points and still beats always-strong by 5 points at 1.3% of its cost.
- **What the extra small models add over AL-04.** At AL-04's accuracy (0.848-0.854) nothing is cheaper: the same Qwen
  7B pairs are the cheapest. The saving is below it: AL-04's smallest retriever is Qwen 7B, so its cheapest pair costs
  175 units, while AL-05 reaches 0.826-0.842 for 25-65 units.
- **The learnt rules are not on the curve.** They reach 0.710 at 13 units, about where the frontier is (nano-jev ->
  Llama 8B, 0.722 at 12; paired on the same questions 38 right only for the rules, 44 only for the pair, so no
  better). The rules stop at the bottom rungs on almost every question instead of escalating where it pays.
- **Switching has to beat the curve to earn its place.** Per-question switching is worth it only if it reaches a point
  above the frontier, e.g. about 0.84 for well under 65 units by using the large models only on hard questions. Not
  shown yet: the oracle row (0.794 at 36 units) only switches answerers on nano-jev's shortlist, and it lands on the
  frontier rather than above it (Qwen 0.8B -> Llama 3B: 0.796 at 22). Whether switching retrievers per question can
  beat the curve is the open question.
- **Caveat.** These pairs were read off the test results. A pair claimed as a result must be chosen on validation
  first (e.g. the cheapest pair within 1 point of the best on val) and then reported once on test; the golden-set run
  (slr-val / slr-test) will do that. nano-jev saw 62 of these test questions in its own calibration; without them
  (438 questions) the picture is the same (nano-jev -> Llama 8B 0.715, learnt rules 0.708, Qwen 0.8B -> Llama 70B
  0.831, Qwen 7B -> Llama 70B 0.845).

### AL-06: the same ladders from Qwen 0.5B (no nano-jev)

| | AL-05 (from nano-jev 33M) | AL-06 (from Qwen2.5 0.5B) |
|---|---|---|
| Starting rules | 0.672 at 12 units | 0.724 at 21 |
| Learnt rules | 0.710 at 13 | **0.756 at 22** |
| Gain from learning | +3.8 pt | +3.2 pt |
| Oracle within the rulebook's choices | 0.794 at 36 | 0.830 at 28 |

- **A stronger first rung helps the rules:** +4.6 points over AL-05's learnt rules for 9 more units (about 1% of
  AgentTTS's cost). Learning again improves on the starting rules by about 3 points.
- **Still below the fixed-pair curve.** Fixed Qwen 0.8B -> Llama 3B reaches 0.796 at 21.8 units: 4 points more than
  the learnt rules at slightly lower cost (paired: 41 questions right only for the pair, 21 only for the rules). The
  rules start at the weakest retriever (Qwen 0.5B) and escalate too rarely to the much better Qwen 0.8B one rung up.
- AL-06's fixed pairs are a subset of AL-05's, so the curve above Qwen 0.5B is the same (best: Qwen 7B -> Llama 70B
  0.854; best cheap: Qwen 0.8B -> Llama 8B 0.826 at 25 units). Its oracle (0.830 at 28) is only just above that curve,
  so at this end there is little room for per-question switching to win.

### Against the two baselines (AgentTTS's 500 test questions, seed 0)

Cost = measured AgentTTS units per question. Differences in bold are clear when the two policies are paired on the same
questions; "n.s." = within the noise of 500 questions.

| Setting | Accuracy | Units | vs AgentTTS cost | vs AgentTTS (0.744) | vs always-strong (0.776) |
|---|---|---|---|---|---|
| AgentTTS paper-best (AL-01) | 0.744 | 1,984 | 100% | - | -3.2 |
| Always-strong, team reference (AL-02) | 0.776 | 1,829 | 92% | +3.2 | - |
| AL-04 learnt rules | 0.830 | 437 | 22% | **+8.6** | **+5.4** |
| AL-06 learnt rules | 0.756 | 22 | 1.1% | +1.2 (n.s.) | -2.0 (n.s.) |
| AL-05 learnt rules | 0.710 | 13 | 0.6% | -3.4 (n.s.) | **-6.6** |
| Fixed Qwen 7B -> Llama 70B (fine-tuned) | 0.854 | 219 | 11% | **+11.0** | **+7.8** |
| Fixed Qwen 0.8B -> Llama 70B | 0.842 | 65 | 3.3% | **+9.8** | **+6.6** |
| Fixed Qwen 0.8B -> Llama 8B | 0.826 | 25 | 1.3% | **+8.2** | **+5.0** |
| Fixed Qwen 0.8B -> Llama 3B | 0.796 | 22 | 1.1% | **+5.2** | +2.0 (n.s.) |

Under AgentTTS's Table 5 prices (the golden set's convention) settings with Llama 70B look less cheap: fixed Qwen 7B ->
Llama 70B is 192 against 929 for always-strong (21%). The golden-set rows report both.

### Why AL-05 and AL-06's rules lose to AL-04's (same fine-tuned models)

The models are fine-tuned identically; what differs is the ladder the rules choose from. The rules start every question
at the smallest model and escalate only on a doubtful signal, and they escalate rarely in all three experiments. The
retriever each question ended on (learnt rules, test):

| Experiment | Ladder starts at | Stayed on the first retriever | The rest ended on |
|---|---|---|---|
| AL-04 | Qwen 7B (fine-tuned) | 414 of 500 | Qwen 32B / 72B |
| AL-05 | nano-jev 33M | 411 | Qwen 0.5B / 0.8B; never the 7B |
| AL-06 | Qwen 0.5B | 430 | Qwen 0.8B / 1.5B / 3B; never the 7B |

- In AL-04 staying put means the fine-tuned 7B, a strong retriever; in AL-05/06 it means a 33M or 0.5B retriever that
  misses more of the needed passages. Answers follow the same pattern (AL-05: 299 of 500 answered by Llama 1B).
- The rules move one or two rungs at a time and run at most 3 retrievers per question, so on an 8-rung ladder the 7B
  (5 rungs above nano-jev) is out of reach.
- Pairwise against AL-04's rules: AL-05 15 questions right only for it vs 75 only for AL-04; AL-06 13 vs 50. The long
  ladders bought 20-30x lower cost (3-5% of AL-04's) for 7-12 points, not the same accuracy for less.
- What this suggests: let the rules jump up the ladder (e.g. straight to Qwen 0.8B or 7B when the first retriever is
  unsure), or start the long ladders at Qwen 0.8B, the efficient point of the fixed-pair curve.

## Fine-tuning so far (early stopping on validation loss, `results/common/history/`)

| Model | Role | Epochs run | Best epoch (kept) | Val loss |
|---|---|---|---|---|
| Llama 3.2 1B / 3.2 3B / 3.1 8B | answerer | 2 | 1 | 0.354 / 0.270 / 0.273 |
| Llama 3.1 70B (QLoRA, 2,000 questions) | answerer | 2 | 1 | 0.258 |
| Qwen2.5 0.5B | retriever | 3 | 2.5 | 0.127 |
| Qwen3.5 0.8B | retriever | 2 | 1 | 0.044 |
| Qwen2.5 1.5B / 3B / 7B | retriever | 2 | 2 | 0.058 / 0.034 / 0.024 |
| Qwen2.5 32B (QLoRA) | retriever | 2 | 1.5 | 0.035 |

Answerers stop improving after one epoch (validation loss rises in the second, so the epoch-1 checkpoint is kept);
retrievers keep improving through epoch 2 (Qwen2.5 32B: best at 1.5). Qwen2.5 72B (QLoRA) has finished training; its
curve is added with the AL-05/06 results.

AL-05 and AL-06 are done (seed 0, AgentTTS's 500).

## Reading so far

- **Reproduction.** AgentTTS's reported best setting scores 0.744 here (paper: 0.742), so the pipeline, data and ruler
  reproduce their setup. Their search re-run with an open planner and a 50-question score table picks a worse, costlier
  setting (0.638 at 3,790 units).
- **Always-strong is the team's shared run.** AL-02 now uses Jonathan's stored greedy always-strong outputs on the same
  500 test questions (0.776 at 1,829 units; 0.788 on val), so every HotpotQA track compares against one baseline. Our
  own first AL-02 run (0.692 at 5,142) sampled at temperature 0.9 / 0.7 and ran AgentTTS's fuser on a single answer,
  which only rewrote it and cost 3,290 of its units; it is withdrawn. Untrained, our short-answer prompts do worse than
  AgentTTS's with the same two models (0.730 vs 0.776, AL-03 fixed Qwen 72B -> Llama 70B).
- **AL-03's rulebook does not yet beat fixed pipelines.** The learnt rules (0.566 at 609 units) are dominated by the fixed
  Qwen 7B -> Llama 70B pipeline (0.584 at 276). They escalate too rarely: 41% of test questions stop at the cheapest pair
  (Qwen 7B -> Llama 3B, 0.400 when used alone), because the untrained models' probe signals flag few wrong answers.
  The "oracle" row is answer-only switching on Qwen 7B's shortlist (the cheapest answerer if right, else Llama 70B), not
  an upper bound on retrieval escalation.
- **Escalation with merged shortlists works.** Where the retrievers disagree (about 430-460 of 500 questions), merging
  Qwen 7B's shortlist with a larger retriever's (the current rule: identical or majority shortlist, else the union)
  answers as well as the larger retriever's own list or better: Llama 70B on 7B+32B merged 0.700 vs 32B alone 0.677
  (test; val 0.698 vs 0.667); 7B+72B 0.754 vs 0.735; 7B+32B+72B 0.743 vs 0.735. A passage-level vote adds nothing.
  The rule is kept as designed for AL-04 .. AL-06.
- **Fine-tuning is the big step (AL-04).** Every fine-tuned pair beats every untrained one, and the fine-tuned
  Qwen 7B -> Llama 8B reaches 0.848 at 178 units: +10 points over AgentTTS's best setting at 9% of its cost, and +7
  over always-strong at 10% of its cost (paired on the same 500 questions: 53 right only for ours, 17 only for
  always-strong). Our recipe changes the prompts as well as the models (short direct answers, fine-tuned), so the
  comparison is our whole recipe against the shared baseline. Larger
  fine-tuned retrievers add nothing (Qwen 7B 0.848, 32B 0.834, 72B 0.816 with Llama 8B): after fine-tuning the small
  retriever's shortlist is enough.
- **AL-04's rulebook learnt nothing.** Over 4 rounds all 72 candidate changes (64 threshold steps, 8 planner
  proposals) were rejected, mostly as "not better on both validation halves"; the starting rules (0.830 at 437 units)
  escalate more than they gain and are dominated by the fixed Qwen 7B -> Llama 8B pipeline. The candidates only nudge
  thresholds (at most 0.05 per round) and none removes a rule, so the learner cannot reach "escalate rarely".
- **Headroom for switching.** Picking, per question, the cheapest fixed pipeline that answers correctly would reach 0.866
  at 366 units (AL-03 models, test): the target the learnt rules approach as their signals improve (fine-tuned models,
  AL-04 .. AL-06).
