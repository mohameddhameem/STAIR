# Pre-registration: what nano-jev's typed decisions add to routing (the Jev novelty programme)

**Status:** pre-registered on 2026-10-24.
- This file is committed before any E1 judge is trained and before any 2WikiMultiHopQA or MuSiQue answer is generated.
- Later changes go to a dated "Deviations" section at the end. Nothing above it is edited after results exist.

**Plan and code:**
- Job sheet: [gpu-jobs-jev.md](../gpu-jobs-jev.md).
- Branch: `feat/jev-novelty`.
- Background: the literature review (git-ignored, `reports/Typed decision router novelty.md`) found that "a small
  model can route" is not new. The open claims concern what typed-decision pretraining adds: labels, transfer and
  per-stage decisions.

## Common protocol

- **Data status:**
  - HotpotQA (the golden set, `slr-*` splits) is **exploratory**: its earlier results shaped these hypotheses.
  - 2WikiMultiHopQA and MuSiQue are **confirmatory**.
- **Pair A on every dataset:**
  - The weak model is Qwen2.5-1.5B-Instruct reading nano-jev's top 2 passages.
  - The strong model is Qwen2.5-7B-Instruct reading all of them.
  - Both run on the transformers backend, greedy, with experiment 1's prompt (`configs/gpu.toml`).
- **Scoring:** the standard lenient ruler (`stair_common.score.lenient`) against the dataset's gold answer. Aliases
  (2Wiki's alias file, MuSiQue's `answer_aliases`) give a secondary score only.
- **Cost:** Table 5 price per answering call (Qwen2.5-7B 10, Qwen2.5-1.5B 0.5). A router's own calls are priced with
  `stair_common.cost.budget` at its measured tokens. nano-jev's calls floor at 0.
- **Primary metric:** AIQ, the area under the accuracy-vs-cost hull between always-weak and always-strong
  (RouterBench). It is reported with and without the router's price. Also reported:
  - the fraction of the random-to-oracle gap closed;
  - CPT(50%) and CPT(80%);
  - the escalation rate at the val-chosen operating point.
- **One judge model:** the AIQ of a single deployed judge is the mean over seeds of each seed's AIQ. The ensemble
  (mean score over seeds) is reported beside it as before.
- **Statistics:**
  - **Bootstrap:** paired, resampling test questions with the same indices for every router. 1,000 replicates for
    reports, 5,000 for the confirmatory tests below. Where seeds vary, a two-level bootstrap (questions, then seeds
    within each arm).
  - **Non-inferiority / equivalence margin:** δ = 0.010 AIQ.
    - On HotpotQA, the bootstrap standard error of a judge-minus-LR difference is about 0.004, and judge-minus-entropy
      about 0.0065.
    - So δ is about 2 standard errors, and the 1,481 test questions give roughly 80% power at δ for the LR comparison.
  - **"Ties"** are claimed only by TOST at δ: both one-sided 95% bounds inside ±δ. A difference that is neither
    significant nor equivalent is reported as "inconclusive".
  - **Multiple comparisons:** Holm within each hypothesis family below, at familywise α = 0.05.
  - **Replication:** a claim is *confirmed* when it holds on both confirmatory datasets. It is *replicated* when it
    holds on at least 2 of the 3 datasets.
- **Test sets:**
  - **HotpotQA:** `slr-test` (1,481), with the **1,303 questions nano-jev never saw** as the primary HotpotQA
    population.
  - **2Wiki:** the team split's test (dev[500:1000], 500). A secondary test, dev[1000:3000] (2,000), gives tighter
    intervals; both are reported.
  - **MuSiQue:** the whole MuSiQue-Ans v1.0 dev set (2,417).
- **Training and val splits:**
  - 2Wiki: router_train is 4,814 questions from the official train set; val is the team's dev[0:500].
  - MuSiQue: val is 500 and router_train is 4,814, both from MuSiQue train.
  - The rule is the team's: sort ids, shuffle with `numpy.random.default_rng(20260729)`, cut.
  - `make-splits --dataset` writes them, and their sha256 is added under "Split hashes" before any answer exists.
- **Exclusions:** none, beyond the clean HotpotQA population above.
  - A prompt over the model's limit stops the run rather than dropping questions.
  - nano-jev's card used 1,000 MuSiQue dev questions for its own held-out *evaluation* (no training), so MuSiQue
    stays clean for nano-jev.
  - The Strands deciders were trained on data that includes MuSiQue. Any decider number on MuSiQue is labelled
    contaminated and is left out of every confirmatory test.

## Hypotheses

### H1 (E1): typed-decision pretraining, label efficiency and the typed head

**Arms:**
- **Starting points:** five MiniLM-L12 checkpoints of 33M each:
  - **plain** (`microsoft/MiniLM-L12-H384-uncased`);
  - **MS MARCO** (`cross-encoder/ms-marco-MiniLM-L12-v2`);
  - **nano-jev v1.0**;
  - **nano-jev-repro** (v1.0's recipe retrained by us);
  - **nano-jev-noHotpot** (the same recipe, HotpotQA replaced by SQuAD 2.0).
- **Heads:** typed and plain.
- **Inputs:** qp and qpa.

**Training:**
- Budgets are 50, 100, 250, 500, 1,000, 2,000 and the full 4,814 on HotpotQA, and 100, 500 and full on the
  confirmatory datasets.
- Seeds: 10 for budgets up to 500, 5 above.
- Labels are `large-helps`.
- Every arm gets identical hyperparameters: nano-jev v1.0's fine-tuning settings, plus a minimum of 150 optimizer
  steps.

**Summary statistic:** the area under the AIQ-versus-log2(budget) curve (trapezoid), called *LE-AUC*.

**Family H1 (Holm), for each of qp and qpa:**
- **H1a:** nano-jev v1.0 (typed) has a larger LE-AUC than plain (typed).
- **H1b:** the gap between v1.0 and plain at budget 100 is larger than at the full budget.
- **H1c:** nano-jev-noHotpot keeps at least half of v1.0's LE-AUC gain over plain. The one-sided test is on
  (noHotpot − plain) − 0.5 × (v1.0 − plain) > 0.
- **H1d:** on nano-jev v1.0, the typed head has a larger LE-AUC than the plain head.

**Gate on the pretraining:**
- nano-jev-repro must reach v1.0's test accuracies on nano-jev's own test split within 0.02 per decision.
- If it does not, H1c is reported as not tested.
- The repro − v1.0 LE-AUC difference estimates the noise added by retraining. It is reported, not tested.

**Exploratory, HotpotQA menu A only:** nano-jev qpa at each budget is compared with Jonathan's 1.5B SFT label router
by TOST at δ, with both routers priced. Budgets are 100, 500, 1,000 and full, with 3 seeds.

### H2 (E2): evidence decides escalation, on new datasets (confirmatory; pair A)

**Family H2 (Holm), per dataset:**
- **H2a:** the qp judge (nano-jev v1.0, full budget) has a higher AIQ than the question-only judge (q).
- **H2b:** the q judge is within δ of random (TOST).
- **H2c:** the qp and qpa judges are non-inferior at δ to the stronger of entropy and the best logistic-regression
  router at the same decision point.
  - qp is compared with the routers that decide before the weak answer, plus entropy.
  - qpa is compared with the cascade routers and entropy.
- **H2d:** the HotpotQA-trained judges (`runs/slr-judge-models`, 43118's), applied with no retraining, keep at least
  75% of the in-domain judge's gain over random. The one-sided test is on
  (AIQ_transfer − random) − 0.75 × (AIQ_in-domain − random) > 0.

### H3 (E3): the stronger baselines (pair A, all datasets; confirmatory on the new ones)

**Signals, from the weak model:**
- answer probability (MSP), mean log-prob, perplexity and first-token margin;
- P(True);
- a logistic probe on the last hidden layer (5-fold on router_train);
- self-consistency (k = 10, T = 0.7);
- discrete semantic entropy (k = 10);
- AutoMix-style self-verification (k = 8).

**Signals, from the evidence and the question:**
- an NLI entailment check (`nli-deberta-v3-xsmall`);
- HHEM-2.1-Open;
- gradient-boosted trees on retrieval-score features;
- a BERT question router;
- kNN on MiniLM embeddings.

**Pricing:** each signal is priced at its extra calls under Eq. 8, and reported free and priced.

**Family H3 (Holm), per dataset:**
- **H3a:** nano-jev qpa (priced) is non-inferior at δ to every signal that costs at most one extra weak-model pass,
  priced: MSP, margin, perplexity, P(True), the probe, NLI, HHEM and the retrieval features.
- **H3b:** nano-jev qpa (priced) has a higher AIQ than self-consistency and semantic entropy (priced).

**Pre-stated reading:** if the probe beats nano-jev, nano-jev's remaining case is that it needs no access to the weak
model's weights, and that it transfers across weak models (H5).

### H4 (E4): routing from typed decisions with no or few routing labels (CPU)

**Scores:** nano-jev v1.0's untrained typed decisions:
- `sufficient` over the top 2 passages;
- relevance of the top passage, and of the top 2;
- `grounded` of the weak answer;
- the "is the proposed answer correct?" question.

**Compositions:**
- every single score;
- the product and the minimum of the pre-answer scores;
- a logistic combiner fitted on n = 10, 25 or 50 labelled router_train questions, with 20 draws each.

**Family H4 (Holm), per dataset:**
- **H4a:** the best zero-label composition **chosen on val** closes at least 50% of the gap between random and the
  full-budget trained judge, on test.
- **H4b:** with 50 labels, the combiner closes at least 75% of that gap (the mean over draws).
- **H4c:** at zero labels, the pre-answer composition (no weak-model call) beats entropy once entropy is charged the
  weak answer it needs.

### H5 (E5): transfer across model pairs (HotpotQA)

**Pairs:**
- **B:** 1.5B / 14B.
- **C:** 3B / 7B.
- **D:** Llama-3.2-3B on the top 2 / Llama-3.1-8B on all 10, a new model family.

**Judges:** the pair-A judges (43118's weights), frozen and scored on each pair, against judges retrained on that pair
at the full budget.

**Family H5 (Holm):**
- **H5a:** on each pair, the frozen qp judge is non-inferior at δ to the retrained qp judge.
- **H5b:** the frozen qpa judge's transfer ratio, (frozen − random) / (retrained − random), is lower than qp's on C
  and D, where the weak model changes.

### H6 (E8): risk-controlled escalation (CPU)

**Method:**
- Temperature-scale P(weak correct) on val.
- Choose the split-conformal threshold for α = 0.05, 0.10 and 0.20, so that the error among questions that are not
  escalated is at most α.
- Repeat over 1,000 random val/test resplits.

**Hypotheses:**
- **H6a:** within each dataset, the realised error averaged over resplits is at most α, and it exceeds α in at most
  10% of resplits.
- **H6b (pre-stated as expected to fail):** calibrated on HotpotQA and applied to MuSiQue, the guarantee is violated.
  This shows the guarantee needs in-domain calibration labels.

### Descriptive only (no confirmatory claims)
- **E6, menu A:**
  - stage-to-stage transfer of the judges;
  - one 4-option typed question against the two binary stage decisions;
  - the best fixed path at each budget.
- **E7:** which rankers make cheap evidence plus sampling work.
- **E9:**
  - p50 / p95 latency per decision on an L40S and on 8 CPU threads;
  - menu A re-priced with AgentTTS v2 Table 10's dollar prices.

## Analysis code

These scripts compute the confirmatory numbers, from the run folders named in the job sheet:
- `stair.label_curve` (H1);
- `judge eval` plus `stair.confirm` (H2, H3, H5);
- `stair.zero_label` (H4);
- `stair.risk_control` (H6).

## Split hashes

Added before any 2Wiki or MuSiQue answer exists:

sha256 of the part's ids joined by newlines, as `make-splits --dataset` wrote them on OMEGA, 2026-10-24 (2Wiki after
deviation 1):

| Dataset | Part | n | sha256 |
|---|---|---|---|
| 2wikimultihopqa | router_train | 4814 | `6f8e3255247d5706197e4b65fa30cd0ffc1143c7652b25609314d6fb4a34f6f4` |
| 2wikimultihopqa | calib | 500 | `81a6fd77f8a242a94c3990bd3dbb56639435839cb413f788f9949d43b262c36e` |
| 2wikimultihopqa | test | 500 | `a310285491780f9a394869c4657112b72aefa382bb4fe85f6ea0b5d5d38620bc` |
| 2wikimultihopqa | test_ext | 2000 | `e367eb1bf25a727b9ebff9e3af683ec0372e214282dbb91308de81f03ab58972` |
| musique | router_train | 4814 | `a0d19e6e643d1119f63f2566ed2753a79bc4a19fbf9acd5cbe9b69fa78c4f567` |
| musique | calib | 500 | `aee3dd6beda2b5a2a8195ef1551a2338bc08bda99d4c18881231ca23a036457d` |
| musique | test | 2417 | `bad4f9b18a7abb8e475b7285db2c370540b9a4aa51770ce553056e9b0761ce46` |

## Deviations

1. **2026-10-24, before any 2WikiMultiHopQA answer existed: the 2Wiki split.**
   - **What was found:** the team's `stair_common` 2Wiki parts (val, test, train_extra) cannot be scored. Their
     source is AgentTTS's `tts_train.json` (pinned at `3f0c2df`, sha256 `4f60ba6a…`). That file holds the official
     2Wiki *test* set: all 12,576 questions have an empty answer and no supporting facts. A Hugging Face mirror
     (`framolfese/2WikiMultihopQA`) lists the same ids under "test", also without answers.
   - **New rule:**
     - 2Wiki's calib, test and test_ext are the first 500, the next 500 and the next 2,000 ids of the official dev
       set, which has answers (the mirror's "validation", 12,576), in the team's random order
       (`stair_common.splits.shuffled`).
     - router_train is unchanged.
     - `prepare-multihop` now refuses any part with questions that lack a gold answer.
   - **Aliases:** 2Wiki's alias file is not in the mirror, so the secondary alias score exists for MuSiQue only.
2. **2026-10-24, before any E1 judge was trained: the nano-jev-noHotpot arm.**
   - **What happened:** as pre-registered (HotpotQA replaced by 12,000 SQuAD 2.0 questions with nano-jev's own
     builders), the arm never trained. Job 43206 shows loss and dev NLL at 0.693 (ln 2) for all three epochs, and test
     accuracy 0.48–0.49.
   - **Why:** in nano-jev's recipe only HotpotQA produces the 3-option *relevance* decision. Without it every example
     is a yes/no choice, and the plain base never leaves the start where every option scores the same.
   - **The reproduction is unaffected:** it reaches v1.0's test accuracy within 0.02 (grounded 0.858 against 0.844,
     relevance 0.812 against 0.815, sufficient 0.854 against 0.845), so the gate passes.
   - **New rule:** HotpotQA is replaced by SQuAD 2.0 *decision for decision*, so the arm keeps v1.0's decision mix and
     volume and differs only in HotpotQA:
     - **relevance:** 6,000 SQuAD questions × 4 examples (`squad_relevance_examples`), against v1.0's 23,840
       HotpotQA ones. Per question:
       - its own paragraph, "directly answers";
       - another paragraph of the same article without the answer, "partially relevant";
       - paragraphs from two other articles, "irrelevant".
     - **sufficiency:** 18,000 of nano-jev's own SQuAD examples, against v1.0's 17,920.
     - **grounded:** MultiNLI's 8,000, unchanged.
   - **Calib and test** are built the same way, with no HotpotQA.
   - **The collapsed model** is kept as `runs/nanojev-pretrain/nohotpot-collapsed` and not used.
3. **2026-10-24, before any E4 analysis ran: where H4's few labels come from.**
   - **What was found:** `judge zeroshot` scores nano-jev's answer-based typed questions (grounded, correct) on calib
     and test only, so router_train has no such scores to fit a combiner on.
   - **New rule:**
     - The H4b combiners draw their 10, 25 or 50 labelled questions from **calib**.
     - H4a's "chosen on val" uses calib too.
     - Test is never used to fit or choose.
   - **Unchanged:** the trained-judge reference in each gap ratio.
     - A composition that reads only pre-answer scores is compared with the full-budget qp judge.
     - Any other composition is compared with the qpa judge.
     - This makes explicit which judge "the full-budget trained judge" means in H4a.
   - **Scores on different scales:** relevance scores are not probabilities, so compositions combine ranks
     (`zero_label.compositions`).
4. **2026-10-24, before any E8 analysis ran: H6's threshold rule.**
   - **Wording vs intent:** H6 says "split-conformal", but its claim is about the error rate *among the questions not
     escalated*. Split-conformal risk control bounds a joint loss (kept *and* wrong), not that conditional rate.
   - **New rule:** the threshold is chosen by Learn-then-Test with fixed-sequence testing, as `stair.risk_control`
     implements it.
     - Thresholds keeping 5%, 10%, …, 100% of the calibration half are tested in that order.
     - Each test uses a Clopper-Pearson upper bound at δ = 0.1, and the scan stops at the first failure.
     - This bounds the conditional error with probability ≥ 0.9, which is exactly H6a's "exceeds α in at most 10%".
   - **No temperature scaling:** only the score's order matters to this rule, so scaling cannot change it.
5. **2026-10-24, before E3 and E9 ran: HHEM-2.1-Open is dropped.**
   - **Why:** it loads only with `trust_remote_code`, which would run code from its Hugging Face repository on the
     cluster.
   - **Coverage kept:** H3a's evidence-verifier family keeps the NLI cross-encoder (`nli-deberta-v3-xsmall`, 71M, no
     remote code) and the retrieval-score trees. E9 times the NLI model in its place.
6. **2026-10-24, after the HotpotQA E8 result (post-hoc, labelled exploratory): two more α values.**
   - **What happened:** the weak model is wrong on 39.5% of HotpotQA questions. At the pre-registered α (0.05,
     0.10, 0.20), the threshold rule keeps 0–6% of questions on it. So H6a holds, but only trivially.
   - **Addition:** `risk-control` also reports α = 0.30 and 0.40, marked `posthoc`, to show where the guarantee starts
     to buy savings.
   - **Unchanged:** H6a and H6b are still judged at the pre-registered α only.
7. **2026-10-24, before E6's joint judge was trained and before any E7 answer: their designs (both stay descriptive).**
   The stage-transfer judges of E6 had already run; these two parts had not.
   - **E6, the one 4-option question:**
     - "Which is the cheapest pipeline that answers this question correctly?" with options WW / WS / SW / SS.
     - It is asked once, before anything escalates, so it reads only the weak retriever's selection (qp), plus path
       WW's answer for qpa.
     - Its label is his two targets as one path. Its four probabilities become the three scores his threshold rule
       sweeps: P(weak retrieval) = P(WW) + P(WS), then P(weak answer) under each retrieval.
     - It is compared with the two-stage judge (`e1-nanojev qpa b4814`) by paired bootstrap; so are the stage-transfer
       judges.
   - **E7, the arms:** Llama-3.2-3B on 5 pool passages, with 1 sample or 20 fused.
     - **Four rankers:**
       - nano-jev: experiment 8's cells, reused;
       - MS MARCO MiniLM-L12: the same 33M architecture;
       - BM25: no neural ranker;
       - the gold oracle: both gold paragraphs, then nano-jev's next 3.
     - **BM25 and the 1-sample arms are additions to the plan.** They separate the ranker's share from sampling's.
     - **Reported:** the team's lenient accuracy and Eq. 8 cost, with each cell paying its own ranker; paired bootstrap
       against nano-jev with 20 fused samples; recall@k per ranker.
8. **2026-10-25, after every result: corrections and omissions.** No test or margin changes.
   - **Dates:** the status line, the file name and deviations 1–6 say 2026-10-24, but all of them were written on
     2026-10-23. The git commits show it: `9767be5` at 12:13, `52ab885` at 12:18, `493bc07` at 12:20,
     `3646444`–`0b833e6` by 13:55, all +08:00. The order stated holds: this file, then the split hashes, then the
     first 2Wiki / MuSiQue answers (pilots 43207 / 43208 submitted at 12:20:41). Deviation 7's date is correct.
   - **Seeds:** the confirmatory E1 cells (MuSiQue, 2Wiki) used 5 seeds at every budget, as the plan's "100 / 500 /
     full × 5 seeds" said. "Training" above says 10 seeds up to a budget of 500.
   - **Not run:**
     - H3's question-side signals: the BERT question router and kNN on MiniLM embeddings.
     - Perplexity as a separate signal. It is a monotone transform of mean log-prob, so its AIQ is the same.
     - The MuSiQue alias score.
     - H3 on 2Wiki's secondary test (H2 only).
     - E9's dollar re-pricing of menu A.
   - **"Free and priced":** the free numbers were computed after the confirmatory results, by
     `confirm --family H3 --free` (`H3-free.json`). They are descriptive and outside the family.
   - **The write-up** is [2026-10-25-jev-novelty.md](2026-10-25-jev-novelty.md).
