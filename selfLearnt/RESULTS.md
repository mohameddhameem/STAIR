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
AL-11 = AL-04 / AL-05 / AL-06. AL-03 is not repeated. AL-07 is in (AgentTTS on slr-test, our own passages): paper-best
0.736 at 1,971 units, searched 0.646 at 3,518. AL-09 .. AL-11 were stopped once AL-13 .. AL-15 (below) were set up: the
same fine-tuned models, on the passages the team's always-strong reference uses.

Because the team's always-strong reference uses AgentTTS's own passage file (tts_test.json) and our runs rebuilt the
passages themselves, the golden set is also run on AgentTTS's passages, in its own folder and IDs, reusing golden_v1's
fine-tuned models: AL-12 = AgentTTS, AL-13 / AL-14 / AL-15 = AL-04 / AL-05 / AL-06 (`configs/golden_tts.yaml`). Checked
first: Jonathan's always-strong answer step re-run on our stack with his passages, retrievals and prompt gives 77.31 on
slr-test against his 76.77 (87% identical answers, 28 vs 20 questions right for only one side).

## Summary: first run vs golden set

Accuracy (lenient) @ measured AgentTTS units per question, seed 0. Old = AgentTTS's 500 test questions; golden = the
team's 1,481 `slr-test` questions with AgentTTS's own passages (the inputs of the team's always-strong reference). The
golden IDs AL-09 .. AL-11 (the same on our own passages) were stopped in favour of AL-13 .. AL-15.

| Experiment | Old (500 questions) | Golden (1,481 questions) |
|---|---|---|
| AgentTTS (paper's best setting) | AL-01: 0.744 @ 1,984 | AL-12: 0.762 @ 1,955 |
| Always-strong (team reference) | AL-02: 0.776 @ 1,829 | AL-08: 0.768 @ 1,821 |
| Untrained models + rules | AL-03: 0.566 @ 609 | not repeated |
| Fine-tuned 3+3 models + rules | AL-04: 0.830 @ 437 | AL-13: 0.796 @ 464 |
| Long ladders from nano-jev + rules | AL-05: 0.710 @ 13 | AL-14: 0.674 @ 12 |
| Ladders from Qwen 0.5B + rules | AL-06: 0.756 @ 22 | AL-15: 0.713 @ 26 |

On the golden set our experiments score 3-4 points lower (the fixed fine-tuned pairs too: Qwen 7B -> Llama 70B 0.854 ->
0.813) while AgentTTS rises, so the lead over always-strong narrows from about +8 to about +4.5 points but stays clear.
**The drop comes from the questions, not from the passages or the retrained models.** Cross-check (2 Nov,
`configs/xcheck_old500.yaml`): the golden models with AgentTTS's passages on the old 500 questions score what the first
run scored there, and lose the 3-4 points only on the golden questions:

| Fixed pair (fine-tuned) | First run (first-run models, our passages, old 500) | Golden models, AgentTTS's passages, old 500 | Golden set (slr-test) |
|---|---|---|---|
| Qwen3.5 0.8B -> Llama 8B | 0.826 @ 25 | 0.828 @ 25 | 0.808 @ 25 |
| Qwen2.5 7B -> Llama 8B | 0.848 @ 178 | 0.844 @ 177 | 0.812 @ 176 |

Why (`jobs/golden_diag.py`, 2 Nov): the question mix is the same (19-20% comparison questions in every set) and the
retriever finds both gold passages as often (89-94%). The difference is where the questions come from: on both of
AgentTTS's slices, dev [0:500] and [500:1000], our pairs score 0.83-0.85; on both golden sets, drawn from all 7,405 dev
questions, 0.81-0.82, even when both gold passages were picked (answer accuracy then 0.85-0.86 vs 0.83-0.84).
Always-strong also drops a little (0.788 / 0.776 vs 0.768 / 0.768). The first 1,000 dev questions suit our fine-tuned
answerers better than the dev set as a whole; the golden set, sampled across all of it, is the representative number,
and our lead over always-strong there is about +4.5 points.

## First run in detail (AgentTTS's 500 test questions)

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

## Golden set on AgentTTS's passages (AL-12 .. AL-15, slr-test, done 2 Nov)

The team's HotpotQA headline set: 1,481 test questions (`slr-test`), each with AgentTTS's own 100 candidate passages
(`tts_test.json`, the same inputs as the team's always-strong reference). All 11 models were fine-tuned again for this
run (`golden_v1`, 25-31 Oct), each checkpoint picked on `slr-val` (1,110 questions); the rules were learnt on `slr-val`
too. Seed 0. Accuracy = lenient; units = measured AgentTTS units per question; price = AgentTTS's Table 5 price (the
golden set's cost rule, always-strong = 929). "Paired" = questions right only for this setting / only for always-strong.
With 1,481 questions an accuracy carries about +-1.1 points of sampling noise.

| Setting | Accuracy | Units | Price | vs always-strong | Paired |
|---|---|---|---|---|---|
| AL-12 AgentTTS paper-best (Qwen2.5 72B x1 -> Llama 3.2 3B x40 + fusion) | 0.762 | 1,955 | 1,327 | -0.6 (n.s.) | 150 / 159 |
| AL-12 AgentTTS searched (its search, Qwen2.5 32B x10 -> Llama 3.1 8B x30 + fusion) | 0.749 | 1,360 | 1,877 | -1.9 (n.s.) | 152 / 179 |
| AL-08 always-strong, team reference (Jonathan's rows) | 0.768 | 1,821 | 929 | - | - |
| AL-13 learnt rules (no change accepted: = starting rules) | 0.796 | 464 | 218 | **+2.8** | 118 / 76 |
| AL-14 learnt rules (no change accepted) | 0.674 | 12 | 4 | **-9.4** | 107 / 246 |
| AL-15 learnt rules (starting rules 0.698 at 24) | 0.713 | 26 | 13 | **-5.5** | 111 / 192 |
| Fixed Qwen 7B -> Llama 70B (fine-tuned) | 0.813 | 217 | 192 | **+4.5** | 110 / 43 |
| Fixed Qwen 7B -> Llama 8B | 0.812 | 176 | 88 | **+4.4** | 129 / 63 |
| Fixed Qwen 3B -> Llama 8B | 0.812 | 78 | 43 | **+4.4** | 131 / 66 |
| Fixed Qwen 0.8B -> Llama 8B | 0.808 | 25 | 18 | **+4.0** | 129 / 70 |
| Fixed Qwen 0.8B -> Llama 3B | 0.785 | 22 | 10 | +1.7 (n.s.) | 129 / 104 |

Oracles within each rulebook's choices: AL-13 0.848 at 182 units, AL-14 0.771 at 41, AL-15 0.783 at 30. Full tables:
[`AL-12`](../results/common/AL-12-hotpotqa-slr-test-seed0.csv) ..
[`AL-15-hotpotqa-slr-test-seed0.csv`](../results/common/AL-15-hotpotqa-slr-test-seed0.csv).

- **The first run's findings hold on the team's set.** Fine-tuned pairs beat both baselines clearly: Qwen 0.8B ->
  Llama 8B is 4 points above always-strong at 1.4% of its measured cost (2% of its price), and 4.6 points above
  AgentTTS's paper setting.
- **The curve is flat at the top.** From 25 units (0.808) to 217 (0.813) accuracy gains half a point, within the noise:
  above Qwen 0.8B -> Llama 8B, larger models buy almost nothing here.
- **The learnt rules still do not beat a fixed pair.** AL-13's rules (0.796 at 464) are below Qwen 7B -> Llama 8B
  (0.812 at 176); AL-14's and AL-15's stay on the bottom rungs, as on the first run. In AL-13 and AL-14 the learner again
  accepted no change.
- **All numbers sit 3-4 points lower than on AgentTTS's 500** (e.g. Qwen 7B -> Llama 70B 0.854 -> 0.813), while
  AgentTTS rises (0.744 -> 0.762) and always-strong is flat (0.776 -> 0.768): the margin over always-strong narrows from
  about +8 to +4.5 points but stays clear.
- **Passages matter for AgentTTS.** On our own BM25 passages (AL-07) its paper setting scores 0.736; on its own
  passages 0.762.
- **Still to do:** the fixed pairs above are read off the test results. A pair claimed as a result must be chosen on
  `slr-val` first (e.g. the cheapest within 1 point of the best) and reported once on test.

## MedAgentBench: acceptance check of our harness (2 Nov)

Before running our models and rules on MedAgentBench (Jonathan's request), our cluster setup must reproduce his
always-weak and always-strong runs: the authors' harness (stanfordmlgroup/MedAgentBench @ 9926011, their grader
refsol.py), all 300 tasks, Qwen2.5 7B and Qwen2.5 72B AWQ, temperature 0, 8 rounds. The records server is the authors'
image unpacked without Docker (`tools/pull_image.py`). Code: `stair/medagent.py` (stage `mab`); per-task grades and the
comparison in [`results/medagentbench/selfLearnt-acceptance/`](../results/medagentbench/selfLearnt-acceptance/).

| Run | Ours | Jonathan's | Tasks that differ | Units / task (ours / his) |
|---|---|---|---|---|
| Qwen 7B (always-weak) | 12.00% | 11.67% | 1 of 300 | 289 / 295 |
| Qwen 72B (always-strong), first run | 52.33% | 56.00% | 27 | 1,276 / 1,255 |
| Qwen 72B, plain greedy (no repetition penalty) | 53.33% | 56.00% | 26 | 1,218 / 1,255 |
| **Qwen 72B, records server on UTC** | **56.67%** | 56.00% | **6** | 1,180 / 1,255 |

- **Accepted.** On the corrected server the 72B matches Jonathan's run (his own two runs: 56.67 and 56.00) and the 7B
  matches too, so later MedAgentBench rows use this setup (`-utc` files).
- **What was wrong: the records server's clock.** The first 72B run lost on the "last 24 hours" task types (4: 77 vs
  100%, 6: 43 vs 73%). The patient database was written inside the authors' container, which runs on UTC, and its
  timestamps carry no time zone; our server ran on the cluster's clock (UTC+8), so date searches were 8 hours off and
  missed readings near the window's start (task4_3: a magnesium reading at 13:35 UTC was missing from
  `date=ge2023-11-12T10:15`), while the grader, which reads the records without a date search, expected it. The server
  now runs on UTC (`-Duser.timezone=UTC`, `jobs/fhir_test.sbatch` checks both clocks).
- The decoding setting (the model's default repetition penalty 1.05 vs plain greedy) moved the score by 1 point; the
  default is kept, as an OpenAI-style vLLM server applies it.

## Fine-tuning (early stopping on validation loss, `results/common/history/`)

Each model was fine-tuned twice: for the first run (checkpoints picked on dev [0:500], AgentTTS's validation questions;
`selfLearnt-<model>-...csv`) and again for the golden set (picked on `slr-val`, 1,110 questions;
`selfLearnt-golden-<model>-...csv`). Same training questions and settings; the validation losses are on different
questions, so they are not comparable across the two runs.

| Model | Role | First run: epochs run / best epoch / val loss | Golden set: epochs run / best epoch / val loss |
|---|---|---|---|
| Llama 3.2 1B | answerer | 2 / 1 / 0.354 | 2 / 1 / 0.388 |
| Llama 3.2 3B | answerer | 2 / 1 / 0.270 | 1.5 / 0.5 / 0.329 |
| Llama 3.1 8B | answerer | 2 / 1 / 0.273 | 2 / 1 / 0.286 |
| Llama 3.1 70B (QLoRA, 2,000 questions) | answerer | 2 / 1 / 0.258 | 2 / 1 / 0.214 |
| Qwen2.5 0.5B | retriever | 3 / 2.5 / 0.127 | 3 / 2 / 0.125 |
| Qwen3.5 0.8B | retriever | 2 / 1 / 0.044 | 3 / 2 / 0.038 |
| Qwen2.5 1.5B | retriever | 2 / 2 / 0.058 | 2 / 1.5 / 0.051 |
| Qwen2.5 3B | retriever | 2 / 2 / 0.034 | 2 / 2 / 0.048 |
| Qwen2.5 7B | retriever | 2 / 2 / 0.024 | 2 / 1.5 / 0.029 |
| Qwen2.5 32B (QLoRA) | retriever | 2 / 1.5 / 0.035 | 2 / 1 / 0.060 |
| Qwen2.5 72B (QLoRA) | retriever | 2 / 2 / 0.020 | 2 / 2 / 0.028 |

Answerers stop improving after about one epoch (validation loss rises after it, so the earlier checkpoint is kept);
retrievers mostly keep improving through epoch 2. The golden curves here were written from the golden run's training records
(2 Nov): on the cluster, 9 of the `selfLearnt-golden-*` files had been overwritten with the first run's curves by a file
renaming slip. Only the curve files were affected, not the models or the results tables.

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
