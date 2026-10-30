# Jonathan-aligned non-LLM routing

Run: `runs/nonllm_jonathan_aligned_v1` in the sibling workspace. Reference:
Jonathan branch commit `5b69ccef5bbb30d4cc5606f45a16695466770e21`.

This run reuses **Jonathan's committed Config B executor outputs**, not our
GGUF-generated outputs, and the common `slr-train/val/test` IDs (4,814/1,110/1,481).
All 29,620 cached four-path verdicts match the shared standard lenient scorer.
The common SS baseline is therefore 76.77%, not the older GGUF test500's 71.8%.
Executors: retrieval Qwen2.5-7B/72B; answering Llama-3.2-3B/Llama-3.1-70B.
No executor is trained, downloaded or called for this experiment.

The decision models are linear classifiers on train-fitted TF-IDF and numeric
text statistics. They do not contain an LLM encoder. A random forest cannot be
updated with this gradient-based GRPO rule, so the differentiable classifier is
used. Three independent seeds 17/18/19 are averaged before validation-only
threshold selection. Three thresholds (ret, QA after W ret, QA after S ret) are
searched over 0:.05:1; learned-only policies and direct fixed controls are separate.

Inputs match the stage-level observation contract: retrieval sees the question
and already-paid weak selections; QA sees question, selected evidence and the
already-paid weak answer. SFT labels use the author's retrieval coverage proxy
and QA outcome rule; unrecoverable X maps to W. Class-weighted SFT: three epochs,
lr1e-4, batch8; best validation AUC. Unlike an LLM's LoRA training, gradients
update only the small classifier weights.

Production reward is `tokenF1 - .1*C/929`, entropy .03, three epochs, lr1e-5,
batch8, selected by validation expected reward. Retrieval's reward is max over
the downstream QA actions (author's A2 assumption). Both QA-only reward and
both-stage reward are evaluated.

Stage GRPO **is an explicit adaptation**, combining these stage rewards with
the update in `router_ablation.py`: eight sampled actions, within-group mean/std
(sample std) advantage, log-probability policy gradient, fixed-reference
KL(policy || reference) .02, no PPO clipping. Cold and SFT-warm two-epoch runs
use lr1e-4/batch16. A separate extension uses at most30 epochs with patience5
on validation expected reward. It is not a claimed reproduction of the slide's
76.23% at80.5%: the exact code/config for that slide row was not identified in
the pinned branch. Frozen-reference KL is initialized separately for each run.

Additional reference audit: the committed `configb_pair_sweep.py` selects the
both-reward peak thresholds (0.95,0.60,1.00). Its frozen test scores reproduce
76.23% at76.98% SS cost excluding router calls. This matches the slide's accuracy,
but leaves80.5% unreconciled. These adapters were trained using exact expected
reward with SFT warm starts; the audit does not identify a sampled GRPO run for
that slide row. The directly corresponding non-LLM recipe is `sft_reward_both`.

Three query-only four-action arms preserve the *historical exploratory*
script's reward `official_EM - max(0,C/(alpha*9.4)-1)`, with its old surrogate
prices [2,5.7,5.7,9.4] and alpha .6 from the pinned split metadata. These prices
do **not** describe Config B's current inference cost. Actual final cost is
always assessed with the Config B table below. These arms are auxiliary,
distinct from the stage-level adaptation and formal SFT+reward recipe.
Query SFT is unweighted CE on argmax reward. Reward-weighted SFT uses
softmax(reward/.5) soft targets.

Training retains the author's simplified SS path price929. Final weak-first
evaluation charges [80,195,894,1009], including every weak call that ran.
Direct fixed paths cost [80,192,817,929]. All costs are Table5 normalized budget
prices, excluding decision-model compute, matching the reference readout; they
are not measured-token FLOP units. Classifier CPU cost is excluded from that
proxy. The batch timing excludes TF-IDF extraction and is not online latency.

Test features are transformed with train-only vocabulary/scaler and saved
independently; train/validation workers do not load test arrays or outcomes.
Validation locks all checkpoints and thresholds. These questions were inspected
in previous research; results are descriptive, not a fresh sealed-test claim.

Verification:

```bash
python nonLLM/experiments/menu_ws_gguf/test_jonathan_aligned.py
python nonLLM/experiments/menu_ws_gguf/verify_jonathan_aligned.py --out ../runs/nonllm_jonathan_aligned_v1
```

All source caches, feature matrices and trained weights stay outside Git.
