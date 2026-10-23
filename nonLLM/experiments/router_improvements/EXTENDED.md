# Longer non-LLM policy training

The new run adds 38,512 weak-model draws: four new independently seeded
answers on each training question and retrieval branch. There is no new test
generation. The existing greedy strong answers and both retrieval branches
remain fixed. As before, the strong prompt does not depend on the weak answer,
so the cached strong answer is a valid counterpart for every new weak sample.

Four feature/data modes are compared: base features without token confidence;
greedy inputs with confidence; confidence with one greedy plus four sampled
training answers; confidence with one greedy plus eight sampled training
answers. Every deployed policy sees only one greedy answer. Training performs
question-level minibatching and includes all variants of each selected question,
averaging their losses. Thus question weights and optimizer-update counts per
epoch are identical across modes. TF-IDF/scalers are fit on train only.

All modes share corrected document-only numerical overlap features: copied
query text is excluded when computing question/document and answer/document
overlap. The linear policy capacity remains 2,500 TF-IDF terms plus numerical
features, separately for retrieval and answering. Retrieval features use only
weak retrieval evidence; no answer or strong evidence is inspected before
the relevant stage has run.

The comparison includes exact expected-reward optimization and finite-action
GRPO (groups of 8, 16 or 32 routing trajectories). Both methods use the same
10-epoch supervised warm start, initialization, minibatch schedule, entropy
regularizer 0.002, gradient-norm clip 1 and Adam settings. GRPO uses two PPO
updates, ratio clip 0.2 and rollout-policy KL coefficient 0.01. Stable joint
log probabilities avoid probability-subtraction saturation in training and
weak-score inference. GRPO remains an adaptation for finite routing actions,
not a language-model GRPO implementation. Qwen weights are unchanged.

Seeds are 17/29/43; lambdas are 0/0.15/0.4. Each stream trains to 120 epochs
and saves epochs 30, 60 and 120. Validation curves are recorded every 10 epochs,
including expected accuracy/cost and the fraction of zero-advantage GRPO groups.
There are 144 seed-specific training streams and 144 three-seed checkpoint /
hyperparameter candidates. This design compares longer optimization and larger
action groups, rather than treating repeated choices among four known actions
as new language-model answers.

Three operating objectives are searched exactly over distinct validation
threshold pairs: cheapest at 95% of SS validation accuracy; maximum validation
accuracy; maximum validation accuracy within direct-SS validation mean cost.
For each of four modes and two training methods, three winners are locked:
24 predeclared policies. Global winners also use validation alone. Only these
locked winners are scored on test. The test set has been examined previously,
so results and paired bootstrap intervals are descriptive, not confirmation
on an untouched holdout. The budget condition is on validation mean cost;
test mean cost is reported and can differ.

Costs use the existing six mean token-aware prices. Extra training draws are
collection costs, not deployment costs for single-answer inference. CPU router
overhead is excluded; confidence statistics assume access to normal weak
generation logits. A separate confidence-scoring pass would need to be charged
if required by the deployment backend.

`extended_run.py` first expands samples on four GPUs, prepares data arrays,
trains disjoint policy streams on all four GPUs, then locks and evaluates.
Unlike the sklearn classifiers, these small PyTorch linear heads can train
on CUDA. The launched scripts are immutable copies under the run's `code/`.

```bash
cd ${WORKSPACE}
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .cascade-env/bin/python -u STAIR/nonLLM/experiments/router_improvements/extended_run.py \
  --out ${WORKSPACE}/runs/nonllm_grpo_extended_v1
```

Progress is in `status.json`, `more_samples/*.status.json`, worker status files
and per-GPU logs. `validation_lock.json` contains every validation candidate
and all selected settings; `REPORT.md` and `results.json` contain final tests.
Training curves and models are under `training/`. Completed individual training
streams can be reused on a resume; an interrupted stream is retrained.

Tests: `.cascade-env/bin/python -m unittest discover -s
STAIR/nonLLM/experiments/router_improvements -p test_extended.py`.
They verify all operating objectives against direct enumeration, stable
probabilities and gradients, exclusion of unpaid strong evidence and gold
metadata, and the size of the predefined training grid. Real GPU smoke checks
cover 120 GRPO epochs and the new sample indices before launch.

Set `WORKSPACE` to the directory containing the STAIR checkout and local experiment data before using these example commands.
