# Corrected four-model finite-action GRPO

Source: corrected GGUF K1 cache; common0.2 train_extra6405/val500/test500. Frozen executors, author lenient score, actual executed-call token units. Not common0.3 SLR golden set. Test previously inspected; results descriptive.

Reward = lenient - lambda * route units / training mean direct SS. Direct routing pays only selected retrieval+QA, no weak-first sunk calls. Three-seed stage-probability ensembles. Learned-only and fixed-fallback selections reported separately.

SS: 71.80% lenient; 1574.983 units.

| Policy | Lenient % | Cost % SS | ms/router |
|---|---:|---:|---:|
| grpo-control-qwen7-llama3 | 54.00 | 9.62 | 0.00 |
| grpo-control-qwen7-llama70 | 62.20 | 12.60 | 0.00 |
| grpo-control-qwen72-llama3 | 65.80 | 97.26 | 0.00 |
| grpo-control-qwen72-llama70 | 71.80 | 100.00 | 0.00 |
| grpo-v2-sft-max_accuracy-learned_only | 54.00 | 9.88 | 9.96 |
| grpo-v2-sft-max_accuracy-with_fixed_fallback | 71.80 | 100.00 | 0.00 |
| grpo-v2-sft-ss_budget-learned_only | 54.00 | 9.88 | 9.90 |
| grpo-v2-sft-ss_budget-with_fixed_fallback | 71.80 | 100.00 | 0.00 |
| grpo-v2-sft-cheap95-with_fixed_fallback | 71.80 | 100.00 | 0.00 |
| grpo-v2-grpo-max_accuracy-learned_only | 71.80 | 100.00 | 9.95 |
| grpo-v2-grpo-max_accuracy-with_fixed_fallback | 71.80 | 100.00 | 9.95 |
| grpo-v2-grpo-ss_budget-learned_only | 71.80 | 100.00 | 9.98 |
| grpo-v2-grpo-ss_budget-with_fixed_fallback | 71.80 | 100.00 | 9.92 |
| grpo-v2-grpo-cheap95-learned_only | 71.80 | 100.00 | 9.97 |
| grpo-v2-grpo-cheap95-with_fixed_fallback | 71.80 | 100.00 | 9.90 |
| grpo-v2-exact-max_accuracy-learned_only | 71.40 | 83.79 | 10.00 |
| grpo-v2-exact-max_accuracy-with_fixed_fallback | 71.40 | 83.79 | 10.05 |
| grpo-v2-exact-ss_budget-learned_only | 71.40 | 83.79 | 10.25 |
| grpo-v2-exact-ss_budget-with_fixed_fallback | 71.40 | 83.79 | 9.93 |
| grpo-v2-exact-cheap95-learned_only | 71.40 | 83.79 | 9.95 |
| grpo-v2-exact-cheap95-with_fixed_fallback | 71.40 | 83.79 | 9.91 |

No validation-feasible learned policy: sft / cheap95.

GRPO samples discrete routing actions from the cached four-action grid. It does not generate extra LLM answers. Exact control enumerates expected reward over these same actions. KL is to the rollout policy, not a frozen reference LLM. CPU feature/decision time is reported separately; CPU/training work is excluded from the FLOP proxy. All operating points were validation-selected before test scoring.

## Training and interpretation

24 GRPO streams (G8/G32 x four cost weights x three seeds) and12 exact controls completed;21/36 streams stopped early. Maximum300 RL epochs. The selected GRPO configuration is G8, lambda0, with best seed checkpoints260/260/290; it learned to select SS for all500 test questions. This is an actual learned-policy outcome, not the fixed fallback being silently substituted. It matches SS but provides no executor cost saving; its CPU decision overhead is additional.

The exact expected-reward control selects training lambda0.05 and validation retrieval penalty0.4. It chooses weak retrieval plus strong QA for96 questions and SS for404. Compared with SS, four questions improve and six worsen, for two fewer correct overall (357 vs359). Executor cost is83.79% of SS, saving16.21%. The descriptive paired bootstrap95% interval for its accuracy difference is[-1.6,+0.8] percentage points. This is not an equivalence/non-inferiority result; the test was previously inspected and model selection considers many validation configurations.

All scoring remains author lenient. "Exact" refers to enumerating expected reward during policy training, not exact-match answer grading. Validation constraints do not guarantee the same test constraint.

[Training curves](learning_curves.png) show lambda0 validation expected accuracy and cost for all three seeds. The curve cost normalizer is training-mean SS; table costs use measured test SS. Raw surrogate loss and expected task loss have different meanings and are logged separately. Warm-start copies were verified bitwise identical across worker copies for each seed.
