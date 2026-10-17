# Experiment 7: a System One router

**Status:** done on 2026-10-07; results in [2026-10-07-system-one-router.md](experiments/2026-10-07-system-one-router.md). The GPU job sheet is [gpu-jobs-system-one.md](gpu-jobs-system-one.md). It also
has a reduced CPU preview, [`configs/system-one-cpu.toml`](../configs/system-one-cpu.toml), which covers
RQ1 and RQ2 with 1 seed. This
is the **central research question for the final report** (see
[final-report-plan.md](final-report-plan.md)). Experiments 1–6 provide its baselines and its
motivation.

## Idea

The project routes each HotpotQA question to a **small** or a **large** LLM: a binary choice,
decided by a calibrated probability and a threshold. Today's best routers are logistic
regressions on 5–15 hand-made numbers: nano-jev's relevance and sufficiency scores, and the small
model's token entropy ([experiment 2](experiments/2026-10-06-cascade-routers.md)).

The idea comes from the **"System One" decision model** described in the Jev AI guide
([Hugging Face blog](https://huggingface.co/blog/sora-2/what-is-jev-ai-a-practical-guide-to-system-one-and)).
A fast model receives a *state* (text or passages) and a *typed question*: a choice between
options, a score on a scale, or a yes/no. It returns calibrated probabilities, and the
application's own code applies a threshold. When the judge is uncertain, the application
escalates to a stronger model.

That guide is a product description, not a paper: it reports no architecture, training data or
benchmarks. We use it only as the **design pattern**. The open model that follows the pattern is
**nano-jev**, a 33M MiniLM cross-encoder. It scores each (question + option, state) pair and
softmaxes over the options, with calibrated temperatures. We already use nano-jev as the paragraph
scorer.

The step this experiment takes: **make the router itself a System One judge.** The judge reads the
text (question, passages, and optionally the small model's answer) and answers one typed
question: *"Which model should answer this question?"*, with the options {small, large}.

## Research question

> **RQ.** Can a small typed-decision judge (System One), which reads the question, the evidence
> and optionally the small model's answer, route between a small and a large LLM better than
> feature-based routers and uncertainty thresholds, once the judge's own cost is counted?

| Sub-question | What is varied | Hypothesis |
|---|---|---|
| **RQ1. Zero-shot** | nano-jev's built-in typed questions, with no routing training: `sufficient` (before the small model runs), `grounded` on the small model's answer, and a custom "is the proposed answer correct?" question (both after it runs) | H1: the `grounded` and `correct` judges beat random and come close to the small-entropy baseline |
| **RQ2. What the judge reads** (the project's original matched control) | Same judge, training and label, four inputs. **Before the small model:** Q (question only, RouteLLM-style) and QP (question + the small model's passages). **After:** QPA (+ the small model's answer) and QA (question + answer) | H2: QPA > QP > Q, and QPA matches or beats the best logistic regression cascade router (AIQ 0.672) |
| **RQ3. Label** | `large-helps` vs `small-fails` | H3: `large-helps` wins, as it did for logistic regression in experiments 1–2 |
| **RQ4. Decision pretraining** | Start from nano-jev, or from the same architecture pretrained only for MS MARCO relevance (`cross-encoder/ms-marco-MiniLM-L12-v2`, MiniLM-L12-H384) | H4: nano-jev's typed-decision pretraining (relevance, sufficiency, grounding) helps, especially for QP |
| **RQ5. Transfer** | Train on the 1.5B / 7B pair, test on the 1.5B / 14B and 3B / 7B pairs. Compare with judges trained on those pairs | H5: transfer holds when the small model is unchanged (1.5B / 14B) and degrades when it changes (3B) |
| **RQ6. Cost** | The judge's GFLOPs and ms per question, added to the cost–F1 curves | H6: the judge costs < 5% of the small model, so the ranking of methods does not change |

## Design

**Model pairs (binary routing).** All three use the existing caches, so no LLM is rerun.

| Pair | Small | Large | Cache | Baselines already measured (AIQ) |
|---|---|---|---|---|
| A (main) | Qwen2.5-1.5B, top-2 nano-jev paragraphs | Qwen2.5-7B, all 10 | `results/gpu` | entropy 0.662, best pre-generation LR 0.665, best cascade LR 0.672, oracle 0.755 |
| B | 1.5B, top 2 | Qwen2.5-14B, all 10 | `results/gpu-large-14b` | entropy 0.687, best LR 0.697 |
| C | Qwen2.5-3B, top 2 | 7B, all 10 | `results/gpu-small-3b` | entropy 0.671, best LR 0.681 |

### Zero-shot judges (RQ1)

These use nano-jev v1.0's calibrated heads, with no training. "Passages" are the top-2
paragraphs the small model read.

| Policy | nano-jev question | Escalation score | Decides |
|---|---|---|---|
| `zs-sufficient` | built-in `sufficient` over the passages (already the feature `ev_sufficiency`) | P(no) | before the small model |
| `zs-grounded` | built-in `grounded`, claim: *The answer to "{question}" is {answer}.* | P(no) | after the small model |
| `zs-correct` | custom typed question: *Is the proposed answer correct? Question: {question} Proposed answer: {answer}*, options yes/no | P(no) | after the small model |

### Fine-tuned judges (RQ2–RQ5)

nano-jev's grouped-softmax cross-encoder, fine-tuned on one typed question.

- **Typed question:** `Which model should answer this question: {question}`, with options
  `small model` and `large model`. Each option is scored as a (question + option, state) pair and
  the two scores are softmaxed together, exactly as nano-jev scores its own decisions. The router
  probability is P(`large model`).
- **State by input variant:**
  - Q: the question;
  - QP: the passages in nano-jev's `[i] title: text` layout;
  - QPA: `Proposed answer: {answer}` followed by the passages;
  - QA: `Proposed answer: {answer}`.
- **Training data:** pair A's 12,000 router-train questions, labelled from the cached F1 at τ = 0.8.
  - 10% are held out, stratified, to pick the best epoch by dev NLL.
  - The calib split stays untouched for temperature scaling and the operating point, as for every
    router so far.
- **Hyperparameters**, nano-jev's own: learning rate 5e-5, weight decay 0.01, 6% warmup, 16
  questions per batch, bf16, max length 512, up to 3 epochs.
- **Seeds:** 3 per configuration. The policy's score is the mean probability over the 3 seeds,
  and the spread of per-seed AIQ is reported.

**Configurations.** Each has 3 seeds, 36 trainings in total.

| # | Inputs | Label | Start from | Trained on | Answers |
|---|---|---|---|---|---|
| 1–4 | Q, QP, QPA, QA | large-helps | nano-jev | pair A | RQ2, and RQ5 by scoring pairs B and C |
| 5–6 | QP, QPA | small-fails | nano-jev | pair A | RQ3 |
| 7–8 | QP, QPA | large-helps | MS MARCO MiniLM-L12 | pair A | RQ4 |
| 9–12 | QP, QPA | large-helps | nano-jev | pairs B and C | RQ5: in-pair reference for the transfer gap |

### Evaluation

The same protocol and code as experiments 1–6 (`eval-routing`):

- **Test set:** all 7,405 dev questions, plus the clean 6,405 without nano-jev's validation
  questions.
- **Metrics:**
  - F1-vs-GFLOPs curves and AIQ;
  - CPT 50% and 80%;
  - the operating point chosen on calib (within 1% of always-large);
  - ECE before and after temperature scaling;
  - breakdowns by bridge/comparison and yes/no vs span.
- **Paired bootstrap** (1,000 resamples) for:
  - each policy's AIQ;
  - each judge minus the small-entropy baseline, random, and the best logistic regression router
    of the same type (before or after the small model).
- **Judge cost:**
  - the judge's tokens × 2 × 33.4M parameters, in GFLOPs per question;
  - measured ms per question on the L40S;
  - AIQ recomputed with the judge's cost added to every point of its curve.
- **Comparison set:**
  - always-small, always-large, random and oracle;
  - the small-entropy threshold;
  - the 10 logistic regression routers of experiment 2;
  - the zero-shot judges;
  - the fine-tuned judges.

**Why not nano-jev's plain base for RQ4.** `microsoft/MiniLM-L12-H384-uncased` was checked on
2026-10-07: it cannot learn this typed question. Its two option inputs ("small model" / "large
model") give final hidden states that differ by only ~0.007, so their scores start almost equal
and move together. Its loss stayed at exactly ln 2 while memorising 32 questions for 15 epochs,
at learning rates 5e-5, 2e-4 and 5e-4. nano-jev learned the same 32 questions (0.70 → 0.36),
and so did the MS MARCO cross-encoder (0.68 → 0.61), which already separates its inputs. The
MS MARCO model is the closer fair comparison: same size and architecture, relevance-trained,
no typed decisions.

## What would count as a result

- **Positive:** a fine-tuned QPA or QP judge significantly beats both the entropy baseline and
  the best logistic regression router of its type (CI of the AIQ difference above 0) at under 5%
  extra cost. The headline is then "a 33M System One judge routes a 1.5B/7B pair better than
  hand-built features".
- **Informative negative:** the judge only ties logistic regression on nano-jev's features. Then
  the features already carry what the judge can read, and System One routing is a cheaper way to
  get the same result without feature engineering. That is still reportable, with the input
  ablation explaining why.
- **Transfer:** whether the judge must be retrained when the large model changes (pair B) or the
  small one does (pair C). This decides whether System One routing is practical.

## Threats to validity

- **Contamination:** nano-jev saw 6,000 HotpotQA train questions and 1,000 validation questions.
  Router-train and calib exclude its train questions. The clean test set excludes its validation
  questions.
- **Label noise:** F1 ≥ 0.8 is a proxy for "correct". τ was insensitive for logistic regression
  (experiment 2), and the sweep can be repeated for the best judge.
- **Selection:** epoch choice uses a router-train hold-out. Temperature and threshold use calib.
  Test is used once.
- **The source:** the Jev AI guide is not peer-reviewed. It motivates the design, and claims rest
  on our own measurements.
- **Scope:** HotpotQA only. Transfer to MuSiQue or 2Wiki is future work (see
  [research-report.md](research-report.md)).
