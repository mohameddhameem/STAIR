# Experiment 8 (plan): cheap retrieval + System One routing on the AgentTTS setting

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

**Status:** phase 0 done on 2026-10-11 (CPU, local): **go**. Phase 1 (the GPU answer runs) done on
2026-10-12 to 2026-10-14; results are below and in [`results/exp8-phase1/`](../results/exp8-phase1/). Phases 2 and 3
(the option-choosing judge, [`configs/exp8-judge.toml`](../configs/exp8-judge.toml), and its evaluation) ran on
2026-10-14; see [the write-up](experiments/2026-10-14-experiment-8.md) and `results/exp8-phase2/`. **With three options H3 was
not met** (73.2 lenient at 6.6% of the cost, against 74.6 for AgentTTS, a gap inside the noise). **With 20- and 40-sample
fused options added (2026-10-15, [write-up](experiments/2026-10-15-experiment-8-more-samples.md), `results/exp8-phase2b/`)
H3 is met:** always-40 scores 75.6 at 23.7% and the judge 75.4 at 19.4%; the 40-sample gain appears on test only.

**Found while building phase 1:** AgentTTS's fusion call (Archon's fuser) repeats the whole user
message twice, and that message holds all ~100 candidate passages. So the call reads roughly 26k+
tokens plus the sampled answers, not the 3 units of one 3B pass. The AgentTTS recipe probably costs
well above the ~101% of strong-everything on the team's slide. Our fusion carries only the
answerer's prompt and the samples, and our costs use measured tokens. It extends the central research question ([system-one-router.md](system-one-router.md))
to a published benchmark, and joins three threads:

- our System One judge (experiment 7);
- AgentTTS (Wang et al., NeurIPS 2025, arXiv:2508.00890), a test-time compute allocator whose
  HotpotQA result is 0.74;
- the team's fixed-policy "ruler" on AgentTTS's models (`reference/Lenient Accuracy and Cost
  Comparison_V1.pdf`).

**Goal:** match or beat AgentTTS's HotpotQA accuracy (74.6 lenient, as re-scored by the team) on
their test questions, at a small fraction of its compute.

## Idea

In AgentTTS's two-stage setting, an LLM retriever selects passages from 100 candidates and an LLM
answers from them. Retrieval dominates the cost: at the paper's per-pass prices (its Table 5, as
verified by the team), one 72B retrieval pass costs **814** of the **929** units of "strong
everything" (72B retrieval + 70B answering). The unit is one 3B pass on the cheapest task.

Our retriever, nano-jev (33M), ranks 100 candidates for roughly **1 unit**. That is an estimate,
about 18k tokens × 2 × 33.4M parameters against ~1,150 GFLOPs per unit, and phase 0 measures it. If
nano-jev's ranking holds up on AgentTTS's pools, retrieval becomes nearly free. The System One judge
can then spend the budget per question where it matters: on answering.

## Research question and hypotheses

> **RQ8.** Can nano-jev retrieval plus a System One judge reach AgentTTS's accuracy on their own
> HotpotQA test questions at ≤ 25% of the strong-everything cost?

| Hypothesis | Measured by | Phase |
|---|---|---|
| **H1.** nano-jev's top-k contains both gold passages about as often as an LLM retriever: within 5 points of the 7B LLM retriever's recall | Recall@2/3/5 of the gold pair on the 100-candidate pools. BM25 order as the floor; the team's LLM retrieval outputs, once checked in | 0 |
| **H2.** Llama-3.1-70B reading nano-jev's top-k is within 2 points of 72B retrieval + 70B answering (72.2 lenient), at ~12% of its cost | Lenient accuracy and cost of the 70B on nano-jev evidence | 1 |
| **H3.** A judge choosing per question among answer options reaches ≥ 74.6 lenient at ≤ 25% of the strong-everything cost. Options: Llama-3.2-3B ×1, 3B ×10 + fusion, 70B ×1 | AIQ and the operating point against the fixed-policy hull and the AgentTTS recipe | 1–2 |

## Design

**Questions.**

| Split | Questions | Use |
|---|---|---|
| Test | HotpotQA dev (distractor) items **500–999** in file order: AgentTTS's test questions, as its `retrieval_QA_data_generator.py` selects them (`start = sample_number`). Pending the authors' confirmation | Final numbers |
| Judge training | 3,000 of our `router_train` questions | Fit the judge |
| Calibration | 1,000 of our `calib` questions | Temperature and operating point |

The training and calibration questions come from HotpotQA train: hard only, nano-jev's training
questions excluded, as in experiments 1–7. Test questions nano-jev saw in its own validation are
reported separately, as the clean subset.

**Candidate pools,** rebuilt as in AgentTTS:

1. The gold paragraphs (supporting facts), deduplicated by text.
2. Then the BM25 top 100 for the question.
3. Then a shuffle, with a **fixed seed** (AgentTTS's had none).

The BM25 corpus is the paragraphs of the question's own split: dev for test, train for training and
calibration. AgentTTS used Elasticsearch `multi_match` over title and text, with default BM25
(k1 = 1.2, b = 0.75) and the standard analyzer. We use the same formula over `title + text` with
lowercase word tokens. That is an approximation, since gold is always forced in and only the
distractors can differ.

**Retrieval options.**

- nano-jev relevance ranking, then top-k for k = 2, 3, 5;
- BM25 order, the floor;
- for comparison, the team's Qwen2.5-7B / 72B (AWQ) LLM retrieval, if its outputs can be reused.

**Answer options.**

- **Llama-3.2-3B-Instruct**, K = 1 (deterministic).
- **3B with K = 10 samples at temperature 0.9 + one fusion call by the same model:** AgentTTS's
  repeated sampling with fusion.
- **Llama-3.1-70B-Instruct (AWQ)**, K = 1.
- **Optional:** Qwen2.5 1.5B / 7B / 14B, for continuity with experiments 1–7.

**Policies compared.**

- the team's fixed ladder: every always-X policy and their upper hull, including random mixtures;
- the AgentTTS recipe (72B retrieval + 40 sampled 3B answers + fusion; 74.6 lenient, re-scored);
- the oracle;
- the small-model entropy threshold;
- logistic regression routers;
- **the System One judge**, a multi-option typed question over the answer options above.

**Metrics.**

- **Accuracy:** lenient accuracy as the primary metric, using the team's scorer, imported as shared
  code. Also strict EM and F1 (official normalisation).
- **Cost:** two scales. AgentTTS's per-pass prices (7B retrieval 77, 72B 814, 3B answer 3, 70B 115)
  for comparability, and measured token-based GFLOPs (2 × params × tokens). A fusion call is costed
  at its actual prompt length, because it reads every sampled answer.
- **Comparison:** AIQ against the fixed-policy hull, the operating point chosen on calib, and a
  paired bootstrap (1,000 resamples).

## Phases

| Phase | Where | What | Output | Effort |
|---|---|---|---|---|
| **0** | CPU, local (under a memory cap) | Build the pools for test and judge-training questions. nano-jev relevance on the 500 test pools. Recall@k (H1) and the measured nano-jev retrieval cost | `runs/exp8/pools/`, `runs/exp8/recall.json` | ~1 day; nano-jev scoring ~2 h on CPU |
| 1 | **GPU (omega)** | nano-jev scoring of the 4,000 training/calibration pools. Answer runs on nano-jev top-k evidence: 3B (K = 1; K = 10 + fusion), 70B-AWQ (K = 1). Questions: 500 test + 4,000 train/calib | Per-question caches as in `results/`, with lenient correctness and token counts | ~3–6 GPU-hours; the 70B may need 2 GPUs |
| 2 | **GPU (omega)**, short | Train multi-option System One judges (inputs qp / qpa) on the training caches | Judge scores on calib/test | < 1 h |
| 3 | CPU | Ladder, hull, AIQ, bootstrap, transfer; write-up and the final-report section | `docs/experiments/…-experiment-8.md` | ~1 day |

**Go/no-go after phase 0:** if nano-jev's recall@5 is more than 10 points below the 7B LLM retriever,
phase 1 switches to a two-step retrieval instead: nano-jev shortlists about 20 passages, and the 7B
LLM retriever picks from that shortlist.

## Phase 0 results (2026-10-11)

**Pools.**

| Pools | Questions | Corpus | Mean candidates | Gold per pool |
|---|---|---|---|---|
| test | 500 (dev items 500–999) | 66,635 distinct dev paragraphs | 100.1 | 2 in every pool |
| router_train | 3,000 | train paragraphs | 100.3 | 2 in every pool |
| calib | 1,000 | train paragraphs | 100.3 | 2 in every pool |

- **Seeded and rebuildable:** the pools are built with seed 0 and rebuild identically with
  `pools build`. At 132 MB they are not tracked.
- **Contamination:** 62 of the 500 test questions were in nano-jev's own validation set. The clean
  subset has 438.

**nano-jev ranking of the 500 test pools (CPU)** (`results/exp8-phase0/recall-test.json`).

Each cell shows the share of questions whose top k holds **both** gold paragraphs, then the mean
share of gold found:

| Ranker | top 2 | top 3 | top 5 | top 10 | top 20 |
|---|---|---|---|---|---|
| **nano-jev** | **0.544** / 0.740 | 0.652 / 0.806 | **0.772** / 0.874 | 0.860 / 0.927 | 0.926 / 0.963 |
| BM25 order (floor) | 0.244 / 0.570 | 0.338 / 0.631 | 0.460 / 0.707 | 0.602 / 0.795 | 0.734 / 0.863 |

- **By question type:** comparison questions are nearly solved (both gold in the top 2 for 0.948,
  in the top 5 for 1.000). Bridge questions are the weak spot (0.447 in the top 2, 0.717 in the
  top 5). Their second-hop paragraph often shares few words with the question.
- **Clean subset (438):** within 0.01 of the full set at every k.

**Against AgentTTS's LLM retrievers.** AgentTTS reports retrieval F1 (Ret_F1: precision/recall F1
of the selected set) for HotpotQA. Read off its Figure 10a, so approximate, and measured on its own
unseeded pools:

| Retriever | Ret_F1 | Cost (AgentTTS units per question) |
|---|---|---|
| Qwen2.5-7B | ~0.42 | 77 |
| Qwen2.5-32B | ~0.70 | |
| Qwen2.5-72B | ~0.89 | 814 |
| **nano-jev top 2** | **0.740** | **2.9** |

nano-jev's top 2 has Ret_F1 = 0.740, because with exactly 2 picked, precision = recall = the gold
share.

**Measured nano-jev cost.**

- **Tokens:** 49,649 per question. Relevance scores 3 options per candidate, so ~300 sequences.
  That is above the plan's estimate of ~1 unit.
- **Compute:** 3,317 GFLOPs = **2.9 AgentTTS units** per question, which is 283× cheaper than the
  72B retrieval pass and 27× cheaper than the 7B.
- **Time:** 12.4 s per question on a 20-thread CPU; minutes on the L40S.
- **Truncation:** 0.5% of pairs exceed 512 tokens.

**H1 verdict: supported, and stronger than hypothesised.** nano-jev does not just come within 5
points of the 7B LLM retriever. Its top-2 retrieval F1 (0.74) is far above the 7B's (~0.42) and
slightly above the 32B's (~0.70), at a fraction of their cost. Only the 72B (~0.89) is better, at
283× the cost. **Go for phase 1** with nano-jev top-k retrieval. The two-step fallback is not
needed.

**Implications for phase 1.**

- **Context size:** give the answerer the top 3 or top 5 rather than the top 2. The top 5 holds both
  gold paragraphs for 77%, and for 72% of bridge questions.
- **Cost:** nano-jev (2.9) + a 70B answer is about 13% of strong-everything (929).
- **Optional variant: hop-2 re-scoring.** Re-rank with the question + the top-1 passage, to recover
  bridge questions. It costs one more nano-jev pass, minutes on the GPU.

## Phase 1 results (2026-10-12 to 2026-10-14)

**Execution.** Seven PBS jobs, all exit 0 on the first attempt with no retries (the backups never ran). The 70B in 4-bit
did not fit one 46 GB L40S, so its jobs ran on 2 GPUs (`device_map="auto"`).

| Job | Cell(s) | GPUs | Wall time |
|---|---|---|---|
| 42567 | `llama70b-top5`, all splits | 2 | 4 h 40 min |
| 42569 | `llama70b-top3`, test only | 2 | 27 min |
| 42584 | `llama3b-top5`, all splits | 1 | 56 min |
| 42585 | `llama3b-top3`, all splits | 1 | 38 min |
| 42586 | `llama3b-s10fuse-top5`, test | 1 | 1 h 28 min |
| 42587 | `llama3b-s10fuse-top5`, calib | 1 | 2 h 52 min |
| 42588 | `llama3b-s10fuse-top5`, router_train | 1 | 8 h 44 min |

Before them, the nano-jev scores of all 4,500 pools (job 42496) reproduced phase 0's recall exactly, and a 20-question
pilot (jobs 42498, 42499) found the 2-GPU requirement. The logs have no errors, and every answer file has the expected
rows (500, 1,000, 3,000), unique ids and no empty answers.

**Test set (500 questions).** Lenient accuracy is the primary metric. Cost is in AgentTTS units and as a share of
strong-everything (929 units).

| Cell | Lenient | Units | % of 929 | s per question |
|---|---|---|---|---|
| 3B, top 3, once | 0.584 | 5.9 | 0.6% | 0.5 |
| 3B, top 5, once | 0.608 | 7.3 | 0.8% | 0.8 |
| **3B, top 5, 10 samples fused** | **0.726** | **64.7** | **7.0%** | 10.6 |
| 70B, top 3, once | 0.628 | 63.7 | 6.9% | 3.2 |
| 70B, top 5, once | 0.680 | 93.6 | 10.1% | 3.7 |
| Oracle (cheapest correct cell) | 0.814 | 14.2 | 1.5% | |
| *AgentTTS recipe, as re-scored by the team* | *0.746* | *~938* | *~101%* | |

The calib and router_train splits give the same picture (3B fused 0.722 and 0.718, 70B top 5 0.677 and 0.666, oracle
0.815 and 0.803).

**Reading the table.**

- **H2 is not met at the 2-point margin.** The 70B reading nano-jev's top 5 scores 68.0, which is 4.2 below the 72.2 of
  72B retrieval + 70B answering, at 10% of its cost. With 500 questions the sampling noise is about ±4 points, so this
  is not a clear shortfall either.
- **Ten fused samples of the 3B beat one 70B answer** (72.6 against 68.0) at 69% of its cost, and sit 2 points under
  AgentTTS's 74.6 at 7% of its cost. A paired test is part of phase 3; I do not claim significance yet.
- **More passages help both models:** top 5 beats top 3 by 2.4 points for the 3B and 5.2 for the 70B.
- **The oracle is the headroom for H3.** Always picking the cheapest correct cell reaches 81.4 lenient at 14 units. A
  judge needs 74.6 at no more than 232 units (25%); the fused 3B alone is already at 72.6 for 65 units.

**Caveats.**

- **Strict EM is about 0** (F1 0.02–0.13), because both models answer in full sentences. That is why lenient is the
  primary metric. The lenient scorer is a placeholder until the team's is shared. The answers are cached, so only
  scoring reruns.
- **10 samples, not AgentTTS's 40,** to keep the GPU time down; and the 70B is 4-bit NF4, while the team's reproduction
  used AWQ.
- **The 3B top-5 costs 7.3 units against AgentTTS's 3,** because it reads five full passages rather than a short
  prompt. Costs are measured tokens, so they are comparable only within our own cells.

## Dependencies

- **Team (shared code):**
  - the lenient scorer;
  - the per-cell outputs and token counts (`meta_*.jsonl`);
  - the exact question slices (the "first-500" slice and the 1,481-question test split);
  - how the fusion call is costed.
- **AgentTTS authors:** `evaluation_2wiki.py` (their EM definition) and `tts_test_2_samples500.json`
  (to confirm the test questions). The questions are listed in the email draft.

Neither blocks phase 0.

## Risks

- **Pool difficulty:** nano-jev was trained on sets of 2 gold + 8 distractor paragraphs, while 100
  BM25 candidates are harder. Phase 0 measures this directly.
- **Model size:** the 70B-AWQ on one L40S (46 GB) is tight. Two GPUs, the per-user limit, need a
  multi-device model load (a small change to `HFExpert`), or vLLM.
- **Statistical power:** 500 test questions give ±3.8 points of sampling noise at 74%. A clear
  accuracy win needs about 78.5. Equal accuracy at a much lower cost is the stronger and more likely
  claim.
- **Comparability:** AgentTTS's EM definition is unpublished. Lenient scoring follows the team's
  re-scoring (0.74 → 74.6), and both lenient and strict numbers are reported.
