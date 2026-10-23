# Non-LLM architecture experiments

Run from the project root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .cascade-env/bin/python -u \
  STAIR/nonLLM/experiments/router_improvements/architecture.py \
  --out runs/nonllm_architecture_v1
```

Uses the existing complete Instruct grid and greedy weak-answer confidence caches.
No new Qwen inference or fine-tuning is required. These scikit-learn regressors
use CPU; GPUs are not occupied unnecessarily.

Four workflows are evaluated:

* `direct4`: predict correctness of WW/WS/SW/SS from question and candidate pool,
  then choose one route directly. Only the selected retrieval and answering calls
  are charged. The first letter describes retrieval; the second describes QA.
* `pre_ret_qa`: choose retrieval before executing either retrieval model; review
  its weak QA answer, then upgrade QA if predicted accuracy gain justifies cost.
  Strong QA branches include the already paid weak QA call. Retrieval supervision
  is SS minus WS correctness, rather than answer-coverage labels.
* `fixed_S_qa`: always strong retrieval, then weak QA review and optional strong
  QA upgrade. No weak retrieval is executed or used as a feature.
* `post_answer_chooser`: strong retrieval and both QA calls, then choose which
  answer to retain. Both calls are charged even if the weak answer is retained.

The pre-router reads only `question` and the supplied first 100 candidate
passages capped at 600 characters. Lexical overlap selects eight candidate texts
and all candidate titles; there is no model retrieval in this feature step.
Answers, supporting titles, outcome labels, difficulty, annotated recall and
other gold metadata are excluded. QA features use only the current paid evidence,
weak answer and seven confidence statistics available from its generation.
The post-answer chooser additionally sees the paid strong answer, without
requiring another confidence-generation pass.

Regressors predict four-route correctness or comparative correctness gain.
Ridge (alpha 10/100), ExtraTrees and histogram gradient boosting are compared by
three-fold MSE on the training split. Both QA branches of a question remain in
the same fold. TF-IDF, numeric scaling and SVD fit within each training fold.
The selected model is then fitted on all training questions. TF-IDF uses up to
5,000 features; tree models use numeric features plus 48 SVD components.

Validation evaluates a predeclared small grid of cost penalties and answer
selection thresholds. Each workflow includes an explicitly executable direct-SS
fallback, which skips all weak calls. The three objectives are maximum accuracy,
maximum accuracy under mean SS cost, and minimum cost at 95% of SS accuracy.
All 12 workflow/objective selections are saved to `validation_lock.json` before
building test features. The global choice for each objective depends solely on
these validation records. Test is already examined in prior experiments and is
descriptive, with unadjusted paired bootstrap intervals.

Costs retain the earlier mean parameter-weighted token ruler for comparability,
and exclude CPU feature/router overhead. A validation mean-cost constraint is
not a hard per-question or future-batch budget guarantee. No outcome oracle is
used to decide routes at inference. Training labels use cached outcomes.

Outputs include three loadable `.joblib` bundles (feature transform and model),
all locked test route predictions, a manifest, status, JSON results and report.
When loading bundles, add this script directory to Python's import path so
`architecture.Features` can be imported by joblib.

## Follow-up combination

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .cascade-env/bin/python -u \
  STAIR/nonLLM/experiments/router_improvements/architecture_combine.py \
  --parent runs/nonllm_architecture_v1 \
  --out runs/nonllm_architecture_combined_v1
```

This follow-up reuses the inherited training-CV-selected pre-router and answer
chooser. Pre-routing first chooses a direct WW/WS/SW/SS workflow. Only initial
SS decisions additionally generate weak QA and invoke the post-answer chooser;
other decisions execute their directly selected QA model. Initial SS decisions
always pay both answering calls, even when the retained answer is SW. Seven
cost penalties and five chooser thresholds are predeclared; uncombined routes
and direct SS are also candidates. Validation locks precede test replay. This
follow-up was motivated during the architecture experiments and is exploratory.

The combined SS-budget validation selection uses lambda 0 and chooser threshold
0. Test accuracy is 59.42% versus SS 59.08%, at 98.69% of the prior mean-cost
ruler. Its unadjusted paired interval is [-0.14, +0.88] percentage points, so
a stable accuracy gain is not established. Recalculation from recorded
per-question input/output tokens gives 98.91% of test direct-SS cost. Both cost
calculations omit CPU feature extraction and router overhead.
