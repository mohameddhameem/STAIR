# Non-LLM router improvement suite

This experiment uses the completed local FP16 Qwen2.5 Instruct grid at
`runs/nonllm_cascade_four_gpu_full_v1`, with the same 4,814/1,110/1,481
question-level split. It does not use the other Base-model sampling experiment.
All fitting uses train; hyperparameters, lambdas, policy seeds (averaged,
not selected), and operating thresholds use validation. Test results are
descriptive because this benchmark has already been examined repeatedly.
No configuration is selected by test accuracy.

## Predefined comparisons

- Frozen original selected TF-IDF naive-Bayes cascade, including its locked
  thresholds, as the common control.
- Rich TF-IDF plus numeric logistic regression, numeric ExtraTrees, and
  histogram gradient boosting. Features include branch identity, question /
  evidence / answer overlap, answer formatting and abstention indicators.
- Retrieval labels from final cell outcomes instead of answer-coverage labels.
- Gain regression: predict strong-minus-weak correctness; retrieval compares
  SS against WS under a strong-answer continuation, QA compares strong and
  weak answering on the same branch.
- Cost-sensitive logistic regression: positive utility difference as target,
  absolute utility difference as sample weight, lambda=0.4. Costs include
  already-incurred weak calls, and use the existing fixed token-aware ruler.
- Direct expected-reward optimization over the four cached cells using linear
  TF-IDF/numeric two-stage policies; reward=correctness-lambda*normalized cost.
- Finite-action GRPO-style comparison with the same policy features and
  supervised warm start. Groups contain eight routing trajectories. Group
  reward normalization, two PPO updates, clipping=0.2 and rollout-policy KL
  coefficient=0.01 are used. This is an explicit adaptation, not the original
  token-level LLM GRPO implementation. Lambda is {0,0.15,0.4,0.8}; seeds
  {17,29,43} are averaged. Direct and GRPO methods use identical initialization
  and minibatch seed schedules. No Qwen parameters are trained.
- Raw-token confidence features for greedy weak answers (mean/min/std/first
  log probability, mean highest-token probability and mean top-two logit
  gap). Generated answer text is teacher-forced through the unchanged weak
  model. Such features require weak inference diagnostics in deployment;
  the experimental extra teacher-forcing pass is collection overhead, not a
  claimed zero-cost operation in a deployment that cannot expose logits.
- Four sampled weak answers per training question and retrieval branch,
  plus the greedy answer. Their total training weight is one per branch.
  Models trained on these variants are evaluated on the original greedy
  deployment inputs: no runtime multi-sample cost is hidden here.
- Deployment consensus: one greedy plus four sampled weak answers, normalized
  majority voting with earliest-answer tie-break, agreement/diversity/entropy
  features, followed by possible strong escalation. All five weak calls are
  charged. Thresholds are locked on validation before test samples are collected.

Weak sampling is temperature=0.7, top_p=0.95, top_k=0, max_new_tokens=128,
with SHA256-derived distinct seeds. Retrieval evidence and strong greedy
answers are held fixed. This estimates weak-output variability and usefulness
of augmentation; it does not estimate a stochastic strong-model success rate.
Greedy baseline answers are not independent stochastic samples.

The original explicit local normalized EM / guarded whole-word containment
scorer is reused, with no scoring changes based on test results. Sampled
correctness is supervision only; it is never a routing feature. Answer
agreement does not use gold. Confidence is computed from raw model logits,
before temperature and top-p filtering; special output tokens are excluded.

## Execution and artifacts

`collect.py` runs four disjoint question shards on GPUs 0-3 using the weak
Instruct model. The trainval phase scores greedy outputs in all splits and
generates four stochastic outputs only for train/val; the test phase generates
the four test outputs after consensus thresholds have been locked.

`run.py` starts cached CPU comparisons concurrently, waits for train/val
sampling, fits confidence/augmentation/consensus models, collects test samples,
and finishes consensus evaluation. Worker errors stop that collection phase
and mark the coordinator failed. JSONL progress is resumable per sample key.
Generation settings, source-grid manifest and executable source hashes are
recorded. The launched run uses an immutable copy of these scripts under its
output directory's `code/`.

```bash
cd ${WORKSPACE}
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .cascade-env/bin/python -u STAIR/nonLLM/experiments/router_improvements/run.py \
  --base ${WORKSPACE}/runs/nonllm_cascade_four_gpu_full_v1 \
  --out ${WORKSPACE}/runs/nonllm_router_improvements_v2
```

Use a fresh output directory for an independent experiment. Run-level progress
is `status.json`; each GPU writes a status and log. Results include per-method
validation locks, selected classifiers, linear-policy seed weights, test
decisions, a comparison table and paired question-bootstrap accuracy intervals.
Intervals are descriptive, without a multiple-comparison adjustment.
The final report is `results/REPORT.md`.

Model-weighted input+output token cost is a compute proxy, not wall time or
exact FLOPs. CPU routing overhead is excluded. Consensus extra weak costs use
validation mean measured tokens, on the original fixed cost unit, rather than
redefining the ruler using test. Deployment claims must additionally account
for any separate confidence-scoring pass if logits cannot be obtained from
the normal weak generation call.

Tests: `.cascade-env/bin/python -m unittest discover -s
STAIR/nonLLM/experiments/router_improvements -p test_methods.py`.
They check exact threshold search against brute force, exclusion of gold and
outcome fields from features, token confidence, stable independent seeds and
policy-gradient/group-advantage behavior. A real GPU smoke test covers
greedy confidence and sampled generation before the full run.

Set `WORKSPACE` to the directory containing the STAIR checkout and local experiment data before using these example commands.
