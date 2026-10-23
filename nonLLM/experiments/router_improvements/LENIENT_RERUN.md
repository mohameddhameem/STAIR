# Completed-family rerun with review-validated reference lenient scoring

The target is lenient matching, not normalized exact match. The original
`mint_labels.py` could not be recovered from the supplied archives, local
directories or the six public AgentTTS commits. Its missing implementation is
not silently replaced with official HotpotQA EM.

## Frozen working scorer

`reference_lenient.py` implements the reconstruction supported by original
answer-review evidence:

1. Lowercase; remove Unicode punctuation; remove English articles; collapse
   whitespace.
2. Accept equality or a complete gold word sequence contained in the prediction.
3. Reject contradictory predictions containing both `yes` and `no` for a
   boolean gold answer.
4. Do not use reverse containment. Do not blanket-reject quantifier prefixes:
   original examples accept `more than 27 languages` and `over 9,000 employees`.

The frozen implementation agrees with **4,267/4,267** unique untruncated
original review judgments: 3,476 train and 791 validation records. Three
potentially truncated review records are excluded. This is a selected subset,
not an independent test certification or proof of the missing source's behavior
on every unobserved case. The protocol and report retain this limitation.

The old strict scorer differs on 278 reviewed answers. The prior local guarded
bidirectional rule differs on 218. These old experiments are kept separately.

## Cohorts

* **Original reference:** supplied `em_lenient` outcomes are authoritative.
  SS validation = 586/1,110 = 52.79%; SS test = 793/1,481 = 53.54%.
  Original-grid non-LLM replays remain a separate benchmark with the original
  flat cost proxy. No reconstructed scorer is needed to replay these labels.
* **Regenerated Instruct:** all 29,620 greedy QA answers are regraded with the
  same frozen reconstructed lenient rule. SS validation = 583/1,110 = 52.52%;
  SS test = 784/1,481 = 52.94%. Original and regenerated answers are different.
  This cohort uses the existing six mean parameter-weighted token call costs.

Regenerated weak answers must never be joined to original answers' outcome
labels. All 73 method / operating-point records in this rerun belong to the
regenerated cohort, with fresh matching labels and lenient rewards.

## Full training scope

* Original-style majority, TF-IDF logistic regression / LinearSVM /
  multinomial NB, and numerical-feature random forest.
* October-style full weak-first cascade and missing-weak-answer ablation.
* Initial 27 comparisons: richer features, outcome targets, ExtraTrees gain
  regression, histogram boosting, cost-sensitive supervision, confidence,
  four-sample augmentation, five-answer consensus, exact expected reward and
  finite-action GRPO.
* Extended exact-reward / GRPO: four feature modes, three seeds, 120 epochs,
  groups 8/16/32, three cost penalties, 144 seed-specific training streams.
  Checkpoint/configuration selection uses validation; 24 family/objective
  operating points are replayed on test.
* Twelve architecture operating points and three combined objectives:
  pre-retrieval direct routing, pre-retrieval + QA review, fixed strong
  retrieval + QA review, paid post-answer choice, and their combination.

All score-dependent labels, sampled-answer targets, reward arrays, model
weights, validation checkpoints and thresholds are rebuilt. Only target-free
feature arrays over identical input observations are reused. Training samples
have equal total weight per question. Mode `base` means the feature ablation
without confidence; it does not mean a Base language model.

Generation inputs are Qwen2.5-1.5B-Instruct / 7B-Instruct FP16 greedy caches and
1.5B Instruct random samples. Qwen weights are not trained. No new LLM
generation or Base/GGUF output is needed for this rerun.

## Run

From `${WORKSPACE}`, use a fresh output directory:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .cascade-env/bin/python -u \
  STAIR/nonLLM/experiments/router_improvements/lenient_run.py \
  --out runs/nonllm_reference_lenient_v1
```

Four GPUs train the extended linear policies while CPU classifiers train.
Status, worker logs, source hashes, frozen scorer, provenance and validation
locks are retained. Fresh router weights are fitted; no previous trained
router checkpoint or previous reward array is loaded.

After completion:

```bash
.cascade-env/bin/python \
  STAIR/nonLLM/experiments/router_improvements/build_lenient_html_report.py \
  --run runs/nonllm_reference_lenient_v1
```

The report's primary entry becomes the completed lenient report. Exact-match
and earlier bidirectional experiments remain separate linked diagnostics.

## Cost and interpretation

Costs include every executed weak / strong call. Direct methods skip unused
weak calls; cascades pay sunk weak calls; consensus pays all five weak answers;
post-answer choice pays both answers even if it retains the weak one. CPU
feature/router overhead and training cost are excluded. Confidence statistics
are assumed extractable from ordinary generation logits.

All selection uses training/validation; architecture regressor families use
grouped training CV. Test has already been examined and results are exploratory.
Small gains and unadjusted bootstrap intervals are not confirmatory evidence.
Completed training under this reconstruction must not be described as recovery
of the original grader or exact reproduction of the original output grid.

Set `WORKSPACE` to the directory containing the STAIR checkout and local experiment data before using these example commands.
