# HotpotQA on the golden set: Jonathan's Stage-Level Routing setup

**Done (2026-10-23).**
- the golden set is imported and verified;
- experiment 10 with nano-jev judges and the 2B and E4B deciders;
- experiments 7 and 9 with nano-jev, MS MARCO, zero-shot and fine-tuned decider judges;
- experiment 8 and the H2 diagnosis.

See [the job sheet](../gpu-jobs-slr.md).

**Decision (2026-10-21):** on HotpotQA the team follows Jonathan's Stage-Level Routing (SLR) setup. It is on STAIR
branch `jonathan` at `5b69cce` and is now in `stair_common` 0.3.0 (STAIR PR #8). Our numbers now report against his,
on the same terms:

| | Before (common standard, [2026-10-18](2026-10-18-common-redo.md)) | Now (golden set) |
|---|---|---|
| Test | dev[500:1000], AgentTTS's 500 | `slr-test`, 1,481 of the 7,405 dev questions |
| Train / tune | HotpotQA train draw / dev[0:500] | `slr-train` (4,814) / `slr-val` (1,110), both from dev |
| Ruler | `lenient` (`stair_common.score`) | unchanged |
| Cost | AgentTTS units from measured tokens | AgentTTS Table 5 prices per call, weak-first, every call that ran counted. Strong-everything = 929. Measured units are kept beside them |
| Router cost | the judge's own cost, always | not charged in the golden set's rows. We also report ours with the judge's cost, which our central question needs |

The common-standard write-up stays as the AgentTTS-slice history: it is still the comparison with the paper's own
500 questions.

## The golden set, verified

`slr all` (`src/stair/slr.py`, [configs/slr.toml](../../configs/slr.toml)) reads his files at the pinned commit,
sha256-checked. Output: [results/slr](../../results/slr/report.json).

- **Ruler:** `stair_common.score.lenient` gives his verdict on all 29,620 question × path cells (7,405 × 4).
- **Split:** his `splits.json`, the grid's split column and `stair_common`'s `slr-*` parts agree on every question.
- **His certified rows reproduce exactly.** For each router, the thresholds are re-selected on `slr-val` from his val
  scores with his own rule, and then replayed on `slr-test`:

| His row | His report | Recomputed | Thresholds re-selected on val (his rule) |
|---|---|---|---|
| Always-strong | 76.77 at 929 | 76.77 at 929.0 | |
| SLR SFT (label) | 76.70 at 835 | 76.70 at 834.8 | 0.95 / 1.00 / 1.00 (peak) |
| SLR SFT (label) + reward | 75.96 at 665 | 75.96 at 664.7 | 0.80 / 0.55 / 0.60 (parity) |

**His selection rules** (`configb_pair_sweep.py`):
- Each stage's threshold takes 21 values (0, 0.05, …, 1), so 9,261 points are swept on val.
- **Peak** is the most accurate point. Ties go to the dearer point, as Python's `max` over his tuples orders them.
- **Parity** is the first cheapest point at or above always-strong's val accuracy.

Experiment 10 selects its operating points with the same two rules.

## Menu A on the golden set

Every path at Table 5 prices: a fixed policy runs only its own retrieval and answer. The oracle takes the cheapest
correct path, else the cheapest path. Measured units are AgentTTS units from his token counts, a different scale
(§3 of the `stair_common` README).

**`slr-test` (1,481 questions):**

| Path | Retrieval → answer | Lenient | Price | % of 929 | Measured units |
|---|---|---|---|---|---|
| WW | Qwen2.5-7B → Llama-3.2-3B | 0.503 | 80 | 8.6% | 175 |
| WS | Qwen2.5-7B → Llama-3.1-70B | 0.640 | 192 | 20.7% | 216 |
| SW | Qwen2.5-72B → Llama-3.2-3B | 0.618 | 817 | 87.9% | 1,779 |
| SS | Qwen2.5-72B → Llama-3.1-70B | **0.768** | 929 | 100% | 1,821 |
| Oracle | cheapest correct path | 0.815 | 205.5 | 22.1% | 408 |

- **On `slr-val` (1,110):** WW 0.523, WS 0.650, SW 0.632, SS 0.768, oracle 0.815 at 199.9.
- **Clean test (1,303, without the 178 questions nano-jev saw):** WW 0.500, WS 0.645, SW 0.619, SS 0.771, oracle
  0.821.
- **Where the room is:**
  - Always-strong is 0.768. The oracle reaches 0.815 at 22% of the price.
  - His label router stays level with always-strong (76.70) at 90% of the price. His reward router gives up 0.8
    points for 72%.
  - Experiment 10 asks whether a System One judge reaches further toward the oracle.
- **Retrieval is the price:**
  - 72B retrieval is 814 of strong-everything's 929, and measured, its prompt is the whole 100-passage pool
    (about 14k tokens).
  - The weak paths that skip it (WW, WS) cost 9–21%.

## Experiment 10: the System One judge on menu A, against Jonathan's routers

**Setup.** One typed-decision judge answers two yes/no questions per question, weak-first like his routers:
- **retrieval:** do the 7B's selected passages hold what the answer needs?
- **answer:** under the chosen retrieval, will the weak answer do? It reads that branch's passages (qp), or those plus
  the weak answer (qpa).

The labels are his (`ret_labels`, `qa_labels`). It trains on `slr-train`, holding out 10% by question for the best
epoch. Thresholds come from `slr-val` with his sweep and his two rules (peak, parity), and are reported once on
`slr-test`. Code: `src/stair/exp10_slr.py`, [configs/exp10-slr.toml](../../configs/exp10-slr.toml). Jobs 43081
(training, 11 minutes) and 43082 (eval). Output: [results/exp10-slr](../../results/exp10-slr/eval/report.json).

**`slr-test`, 1,481 questions.** AIQ is over weak-first prices from 80 (all WW) to 1,009 (all SS), with the paired
bootstrap. Judges are the mean of 3 seeds. The deciders are the Strands Deciders with LoRA, trained like experiment 9's:
the 2B in job 43098 (eval 43100), the E4B in 43114 (eval 43115).

| Router | AIQ [95% CI] | With its own price | One seed | Peak point (val rule → test) | Parity point |
|---|---|---|---|---|---|
| **Decider E4B qpa** | **0.736** [0.717, 0.758] | 0.730 (20.2 a question, 3 passes) | 0.731, 0.735, 0.729 | **76.70 at 737.5** | 76.70 at 737.5 |
| Decider 2B qpa | 0.731 [0.711, 0.753] | 0.730 (4.7 a question, 3 passes) | 0.733, 0.726, 0.726 | 76.77 at 1,009 (all SS) | 76.77 at 1,009 |
| SLR SFT (label) + reward (his) | 0.726 [0.705, 0.747] | (uncharged) | | 75.96 at 679.3 | 75.96 at 664.7 |
| SLR SFT (label) (his) | 0.724 [0.704, 0.745] | (uncharged) | | 76.70 at 834.8 | 75.35 at 670.3 |
| Decider E4B qp | 0.723 [0.702, 0.744] | 0.717 | 0.718, 0.721, 0.720 | 76.77 at 1,009 | 76.77 at 1,009 |
| Decider 2B qp | 0.722 [0.701, 0.742] | 0.720 | 0.720, 0.719, 0.721 | 76.77 at 1,009 | 76.77 at 1,009 |
| nano-jev qpa | 0.719 [0.698, 0.739] | 0.719 | 0.717, 0.717, 0.717 | 76.77 at 1,009 | 76.77 at 1,009 |
| nano-jev qp | 0.718 [0.697, 0.739] | 0.718 | 0.716, 0.715, 0.716 | 76.77 at 1,009 | 76.77 at 1,009 |
| Fixed paths (hull) | 0.693 [0.672, 0.714] | | | | |

**The 2B decider reading the weak answer is the first of our judges to match his routers.**
- **Before its price:**
  - qpa − his label router is **[+0.001, +0.014]**, better;
  - qpa − his reward router is [−0.000, +0.012], a tie.
- **With its price** (1.6 a pass, at its measured tokens): [−0.000, +0.012] and [−0.002, +0.011], ties with both.
- **One seed:** seed 0 beats both his routers ([+0.002, +0.016] and [+0.001, +0.014]); seeds 1 and 2 tie them.
- **Clean test** (1,303): 0.735 against his 0.727 and 0.729; − label [−0.000, +0.015], − reward [−0.002, +0.013].
- **The 2B qp decider,** which doesn't read the weak answer, ties his routers ([−0.008, +0.002], [−0.010, +0.001]).
- **At his operating rules, it still escalates everything,** like nano-jev: on val, no threshold point reaches
  always-strong's accuracy at a lower price, so both rules pick all-SS.
  - It is not short of operating points: it has 106 val-Pareto points, against his routers' 121 and 133.
  - So it ranks questions better on average (AIQ), but his routers own the top end: points at or above always-strong's
    accuracy for less.
  - One seed alone does find such points: seed 0's parity point gives 76.64 at 814.2, and seed 2's 76.50 at 824.3.

**The E4B decider reading the weak answer beats both his routers before its price, and ties them with it.**
- **Before its price:**
  - qpa − his label router is **[+0.004, +0.021]**;
  - qpa − his reward router is **[+0.003, +0.019]**;
  - it beats the fixed paths by [+0.035, +0.053].
- **With its price:** three passes cost 20.2 a question (one pass 6.5, both stages), and the difference to his routers
  is [−0.002, +0.015] and [−0.003, +0.013], ties.
- **One seed with its price** scores 0.730, 0.733 and 0.727. Seed 1 still beats his label router ([+0.001, +0.016])
  and ties his reward router; seeds 0 and 2 tie both.
- **Clean test** (1,303): 0.739 against his 0.727 and 0.729. Before its price it beats both ([+0.003, +0.021],
  [+0.001, +0.018]); with it, it ties both.
- **It gives the first real operating point under his rules.**
  - The mean of three seeds reaches **76.70 at 737.5**: his label router's accuracy (76.70 at 834.8) at 12% less.
  - It takes all four paths: WW 230, WS 149, SW 584, SS 518. It leans on SW (72B retrieval, 3B answer), which his
    label router never uses and his reward router uses for 187 questions.
  - Even with its own three passes (+20.2), it costs 757.7, still below his 834.8.
  - One seed alone does not repeat this: seeds 0 and 1 escalate everything, and seed 2's parity point is 76.03 at 710.5.
    So the point needs the three-seed ensemble.
- **The E4B qp decider,** without the weak answer, ties his routers before its price ([−0.007, +0.003],
  [−0.009, +0.002]). With it, it is worse ([−0.013, −0.003], [−0.015, −0.004]).
- **The first E4B run** (43099) ran out of GPU memory at micro-batch 4. This one trained at 2, as experiment 9's E4B.

**nano-jev judges (33M):**

- **His routers win, narrowly but clearly:**
  - qpa − his label router is [−0.011, −0.0004];
  - qpa − his reward router is [−0.013, −0.002];
  - qp is worse than both too.
  - Both judges beat the fixed paths: qp by [+0.021, +0.029], qpa by [+0.022, +0.030].
- **At his operating points, our judges collapse to escalating everything.** On val, no threshold reaches always-strong's
  accuracy at a lower price, so peak and parity both pick all-SS: 76.77 at 1,009 (weak-first). His label router
  keeps that accuracy (76.70) at 834.8, and his reward router gives up 0.8 points for 664.7.
- **Clean test** (1,303 questions): the same order. The judges score 0.721 and 0.722, his routers 0.727 and 0.729.
- **The judge's own cost doesn't change this.** A nano-jev call prices at 0 under `budget()`, since it is cheaper than
  one 3B sample.
- **Why, most likely:** his routers are Qwen2.5-1.5B with LoRA, about 45 times nano-jev's size. The size-matched 2B
  decider (above) closes the gap and edges ahead on AIQ, and the E4B goes past them.

## Experiments 7 and 9 on the golden set (pair A)

**Setup.** Experiment 7's pair A is not menu A: Qwen2.5-1.5B reads the top 2 of the 10 distractor passages, and
Qwen2.5-7B reads all 10. Its caches are cut from experiment 1's full-dev run onto `slr-train` / `slr-val` /
`slr-test`, with no model calls (job 43083). The judges and LR routers retrain on `slr-train` (43084, 43085), and are
evaluated on `slr-test` (43097). The fine-tuned deciders trained in 43101 (2B) and 43102 (E4B), and 43103 re-evaluated
everything with them; the earlier rows are unchanged.

**Prices:**
- The 7B pays 10 (Table 5). The 1.5B pays 0.5: the same formula at its nominal size, as Table 5 doesn't list it.
- A router that reads the small answer pays both calls when it escalates.
- A judge's own calls are priced with `budget()` at their measured tokens: nano-jev 0, the 2B decider about 1.2–1.3,
  the E4B about 4.4–4.5 a call.

Output: [results/slr-judges](../../results/slr-judges/eval/A/report.json).

**`slr-test`, 1,481 questions** (always-small 0.609 at 0.5, always-large 0.666 at 10, oracle 0.772 at 2.05):

| Router | AIQ [95% CI] | With its own price |
|---|---|---|
| Oracle | 0.758 | |
| **Decider E4B fine-tuned, qpa** (mean of 3 seeds) | **0.711** [0.691, 0.731] | one judge: 0.655, 0.655, 0.654 (4.4 a call); mean of 3: 0.609* |
| Decider 2B fine-tuned, qpa (mean of 3 seeds) | 0.696 [0.676, 0.718] | one judge: **0.682, 0.681, 0.677** (1.3 a call); mean of 3: 0.658 |
| Decider E4B, zero-shot (`grounded`) | 0.696 [0.675, 0.717] | 0.650 |
| Decider E4B fine-tuned, qp | 0.684 [0.664, 0.706] | one judge: 0.647, 0.643, 0.644 |
| LR evidence + small answer | 0.681 [0.661, 0.702] | |
| Decider 2B fine-tuned, qp | 0.679 [0.658, 0.701] | one judge: 0.665, 0.670, 0.670 |
| Decider 2B, zero-shot (`correct`) | 0.678 [0.656, 0.698] | 0.667 |
| nano-jev qpa (mean of 3 seeds) | 0.677 [0.658, 0.699] | 0.677 (one seed: 0.669–0.678) |
| LR cascade (question + evidence + small answer) | 0.676 [0.655, 0.698] | |
| nano-jev qp | 0.674 [0.654, 0.698] | 0.674 |
| Entropy threshold | 0.665 [0.645, 0.686] | |
| MS MARCO cross-encoder, qp | 0.659 [0.639, 0.680] | |
| nano-jev qa | 0.646 | |
| nano-jev q | 0.641 | |
| Random | 0.638 | |

\* Three E4B passes cost 13.3 a question, more than the 7B's answer (10). The curve then lies wholly above the AIQ
price range, and its AIQ floors at always-small's 0.609, for qp and qpa alike.

**Paired differences:**

| Comparison | Before its price | With its price |
|---|---|---|
| nano-jev qpa − entropy | [−0.000, +0.025] | the same (its calls price at 0) |
| nano-jev qpa − LR cascade | [−0.007, +0.011] | the same |
| Zero-shot E4B − entropy | [+0.019, +0.043] | **[−0.030, −0.003]** |
| Zero-shot E4B − LR cascade | [+0.009, +0.030] | **[−0.039, −0.015]** |
| Zero-shot 2B − LR cascade | [−0.009, +0.012] | [−0.019, −0.0001] |
| Fine-tuned 2B qpa − entropy | [+0.020, +0.045] | one judge: **[+0.006, +0.028]**, [+0.005, +0.028], [+0.001, +0.024] |
| Fine-tuned 2B qpa − LR cascade | [+0.011, +0.032] | one judge: [−0.004, +0.016], [−0.005, +0.016], [−0.008, +0.011] |
| Fine-tuned 2B qpa − nano-jev qpa | [+0.009, +0.029] | mean of 3: [−0.031, −0.011] |
| Fine-tuned E4B qpa − entropy | [+0.033, +0.058] | one judge: [−0.023, +0.001], [−0.024, +0.001], [−0.024, +0.000] |
| Fine-tuned E4B qpa − LR cascade | [+0.024, +0.046] | one judge: [−0.033, −0.010], [−0.034, −0.011], [−0.035, −0.011] |
| Fine-tuned E4B qp − LR cascade | [+0.000, +0.019] | |
| MS MARCO qp − nano-jev qp | [−0.024, −0.007] | |
| MS MARCO qpa − nano-jev qpa | [−0.031, −0.014] | |

- **The research questions on the golden set:**
  - **RQ2:** the passages carry the signal. q (0.641) and qa (0.646) are far below qp (0.674) and qpa (0.677).
  - **RQ3:** the label doesn't matter here. large-helps and small-fails give 0.674 against 0.674 (qp), and 0.677
    against 0.675 (qpa).
  - **RQ4:** nano-jev's pretraining beats MS MARCO's, by +0.007 to +0.031.
- **Before cost, fine-tuning helps and size helps.** The fine-tuned E4B reading the small answer is the best router
  (0.711). It beats entropy, the LR cascade and nano-jev qpa by +0.02 to +0.06, and scores above its zero-shot
  version (0.696). The fine-tuned 2B is next (0.696).
- **With its price, one fine-tuned 2B qpa judge is the best deployable router.**
  - Its call costs 1.3, an eighth of the 7B's answer.
  - It scores 0.677–0.682: better than entropy for all three seeds, and level with the LR cascade.
  - This is the common standard's result again ([2026-10-18](2026-10-18-common-redo.md)), now on 1,481 questions.
- **The E4B pays for itself out of its lead.** At 4.4 a call (nearly half the 7B's answer), one fine-tuned E4B judge
  falls to 0.654–0.655, which ties entropy and is worse than the LR cascade. The zero-shot E4B does the same (0.650).
- **Averaging three seeds never pays here:** three passes cost 3.8 (2B) or 13.3 (E4B) a question.
- **The 33M nano-jev judge ties entropy and the LR cascade,** at no price.
- **Clean test** (1,303): the same order.
  - Before cost: E4B fine-tuned qpa 0.711, 2B fine-tuned qpa 0.695, E4B zero-shot 0.694, LR 0.676, nano-jev qpa
    0.672, entropy 0.661.
  - One 2B qpa judge with its price: 0.675–0.681.
- **Caveat on the prices:** Table 5 prices each answering call at the paper's nominal lengths (256 + 64 tokens), while
  the 7B here reads all 10 passages. A judge has no Table 5 entry, so it is priced at its measured tokens. Relative to
  the experts, judges are therefore dearer here than in measured AgentTTS units. There, the E4B zero-shot with its cost
  tied the LR cascade router ([2026-10-18](2026-10-18-common-redo.md)).

## Experiment 8 on the golden set: cheap retrieval, sampled answers

**Setup** ([configs/exp8-slr.toml](../../configs/exp8-slr.toml), job sheet rows 8.1–8.8):
- nano-jev ranks each question's 100-passage pool, and the answerer reads its top 5.
- The five answer options are experiment 8's: Llama-3.2-3B with one answer, or 10 / 20 / 40 samples fused, and
  Llama-3.1-70B (4-bit) with one answer. All use AgentTTS's answer prompt.
- **Split:**
  - test is all 1,481 `slr-test` questions;
  - the option judge trains on 1,000 `slr-train` questions and is calibrated on 500 `slr-val` ones.
  - Questions already answered on the common standard are drawn first, and their answers are reused (199 test,
    149 calib, 652 train; all byte-identical).
- **Prices** (Table 5, Eq. 8):
  - 3B 3; ×10 / ×20 / ×40 fused 12 / 22 / 42, plus a fusion call at 3; 70B 115.
  - nano-jev's ranking of the pool is priced with `budget()` at its measured tokens: 49k tokens, 6.6 a question.
- **Jobs:**
  - answers 43091–43095, plus 43104;
  - the option judge 43096, re-evaluated by 43116;
  - output in [results/exp8-slr](../../results/exp8-slr/answers/summary.csv).
- **Checks:** recomputing the standard ruler from the raw answers matches the summary exactly.

**The options against Jonathan's rows** (`slr-test`; the paired bootstrap is over questions, 1,000 resamples):

| Policy | Lenient | Price | % of 929 | − his always-strong, points [95% CI] |
|---|---|---|---|---|
| His always-strong (SS: 72B retrieval → 70B) | 76.77 | 929 | 100% | |
| His SLR SFT (label) | 76.70 | 834.8 | 90% | |
| His SLR SFT (label) + reward | 75.96 | 664.7 | 72% | |
| His WS (7B retrieval → 70B) | 64.01 | 192 | 21% | |
| nano-jev top 5 → 3B, 20 samples fused | **75.08** | **31.6** | **3.4%** | −1.69 [−4.19, +0.68] |
| nano-jev top 5 → 3B, 40 samples fused | 74.88 | 51.6 | 5.6% | −1.89 [−4.46, +0.74] |
| nano-jev top 5 → 3B, 10 samples fused | 74.34 | 21.6 | 2.3% | −2.43 [−4.93, −0.14] |
| nano-jev top 5 → 70B, one answer | 68.33 | 121.6 | 13% | −8.44 [−10.60, −6.01] |
| nano-jev top 5 → 3B, one answer | 61.44 | 9.6 | 1.0% | |
| Oracle over the five options | 85.42 | 16.0 | 1.7% | |

- **The ×20 fused answer ties always-strong and both of his routers, at 3–5% of their price.**
  - Against his label router: −1.62 [−4.05, +1.08]. Against his reward router: −0.88 [−3.45, +1.69].
  - It beats his WS path, the cheapest way to his 70B (192), by +11.07 [+8.44, +13.64].
  - ×20 and ×10 tie each other: +0.74 [−0.88, +2.30].
- **Clean test** (1,303): the same picture. ×20 scores 75.44 against always-strong's 77.05, a difference of −1.61
  [−4.22, +1.07].
- **Two caveats:**
  - A tie is not equality: the intervals allow always-strong to be up to 4 points better.
  - Table 5 prices each call at the paper's nominal lengths, while the 20 samples are priced under Eq. 8 (one prompt).
    The measured units tell the same story: 44.7 for ×20 fused against 1,821 for SS.

**The option judge** (nano-jev, 3 seeds; [report](../../results/exp8-slr/judge-5opt/eval/report.json)):
- It chooses one of the five options per question.
- It trains on the cheapest correct option (75% of training questions are labelled "3B, one answer").
- Its own calls price at 0 under `budget()`.
- The baselines escalate from the 3B answer to the most accurate option on calib, ×20 fused.
- AIQ covers 9.6 (always 3B) to 121.6 (always 70B).

| Router (`slr-test`) | AIQ [95% CI] | One seed |
|---|---|---|
| Oracle | 0.847 | |
| nano-jev qp | **0.746** [0.726, 0.768] | 0.741, 0.746, 0.746 |
| Best mix of fixed options | 0.743 [0.725, 0.765] | |
| nano-jev qpa | 0.742 [0.724, 0.764] | 0.744, 0.740, 0.745 |
| LR before the 3B answer | 0.738 [0.716, 0.759] | |
| Entropy threshold | 0.737 [0.716, 0.758] | |
| LR after the 3B answer | 0.737 [0.715, 0.758] | |

- **No router beats the fixed options here.** Every paired difference straddles 0:
  - qp − the fixed mix is [−0.007, +0.012];
  - qp − entropy is [−0.001, +0.020];
  - qp − LR before is [−0.002, +0.020];
  - qpa − entropy is [−0.004, +0.017].
- **Clean test (1,303):** the same order and the same ties. qp scores 0.750, the fixed mix 0.746, qpa 0.743, and entropy
  0.741.
- **Operating points (calib → test):**
  - the qpa judge at the 25% budget gives 74.27 at 25.8;
  - qp gives 73.67 at 26.9;
  - always ×20 gives 75.08 at 31.6, which the LR-before router picks.
- **Why routing adds little here:**
  - ×20 fused is already within 1.7 points of always-strong, at 3.4% of its price.
  - The oracle (85.4 at 16.0) shows that per-question choices could still gain about 10 points. Neither the judge nor
    the baselines find them from the question, the passages and the 3B answer.
- **Correction before publication:** the first evaluation (job 43096) charged nano-jev's ranking twice to every router
  that reads the 3B answer and escalates. That made qp look better than entropy ([+0.002, +0.023]) and LR after
  ([+0.002, +0.023]). Fixed in `dd5b76f` and re-evaluated by 43116; the common-standard reports moved by about 0.002
  ([correction](2026-10-18-common-redo.md)).

## H2 on the golden set: the gap is retrieval, not the answer prompt

H2 ([experiment 8 plan](../experiment-8-plan.md)) asked whether the 70B reading nano-jev's top passages comes within
2 points of 72B retrieval + 70B answering. On the golden set it misses by 8.4 points (68.33 against 76.77).

Two things differ between our 70B cell and his SS path:
- the evidence: nano-jev's top 5 raw passages, against his 72B's selection;
- the prompt: AgentTTS's long-form answer, against his "ONLY the exact answer phrase", greedy, at most 128 tokens.

The H2 job (43090) separates them: our 70B with his prompt, reading either nano-jev's top 5 or his 72B's selections.

| 70B cell (`slr-test`) | Lenient | Paired difference [95% CI] |
|---|---|---|
| His prompt, his 72B selections | 76.23 | − his SS: −0.54 [−1.82, +0.74]. They agree on 93.7% of questions, with identical answer strings on 84.8% |
| His prompt, nano-jev's top 5 | 69.82 | − the row above: **−6.41 [−8.24, −4.66]** (retrieval) |
| AgentTTS's prompt, nano-jev's top 5 | 68.33 | − the row above: −1.49 [−3.78, +0.81] (prompt) |

- **We reproduce his SS:** with his evidence and prompt, our 4-bit 70B scores 0.5 points below his AWQ 70B, within noise.
- **Retrieval is the gap.** Swapping his 72B's selections for nano-jev's top 5 costs 6.4 points. Swapping the prompt
  costs 1.5, which is not significant. So the earlier guess that the answer format explained H2 is wrong.
- **Why the 72B's evidence is better:**
  - It isn't a ranking. The 72B reads the whole pool and writes about 2 references per question.
  - 27.5% of the references on known paragraphs are rewritten rather than copied, as in "Shirley Temple portrayed
    Corliss Archer in the 1945 film 'Kiss and Tell'".
  - The gold answer already appears in its selected text for 85.4% of test questions.
  - nano-jev's top 5 holds both gold passages for 75.8% of questions (86.7% of gold passages).
- **The cheap fix is sampling, not a bigger reader.** On the same nano-jev passages, the 3B with 20 samples fused
  (75.08) beats the 70B (68.33) at a quarter of its price.
