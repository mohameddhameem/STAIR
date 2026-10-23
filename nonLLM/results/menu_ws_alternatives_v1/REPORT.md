# Additional corrected-cache non-LLM routers

Same frozen executors, common0.2 train_extra6405/val500/test500, author lenient and actual direct-call token units. Not the common0.3 golden set. Repeatedly inspected test: exploratory comparisons.

All models are non-LLM TFIDF/numeric predictors. New lexical statistics use only candidate pool before retrieval or selected paid evidence after retrieval. No fresh executor generation. Learned-only and fixed-fallback selections are separate. Offline vectorized evaluation does not measure full routing latency; ms_router is left empty.

| Policy | Lenient % | Cost % SS | SS choices |
|---|---:|---:|---:|
| alternative-control-qwen7-llama3 | 54.00 | 9.62 | 0 |
| alternative-control-qwen7-llama70 | 62.20 | 12.60 | 0 |
| alternative-control-qwen72-llama3 | 65.80 | 97.26 | 0 |
| alternative-control-qwen72-llama70 | 71.80 | 100.00 | 500 |
| alternative-exact_augmented-max_accuracy-learned_only | 71.80 | 89.25 | 429 |
| alternative-exact_augmented-max_accuracy-with_fixed_fallback | 71.80 | 89.25 | 429 |
| alternative-exact_augmented-ss_budget-learned_only | 71.80 | 89.25 | 429 |
| alternative-exact_augmented-ss_budget-with_fixed_fallback | 71.80 | 89.25 | 429 |
| alternative-exact_augmented-cheap95-learned_only | 71.00 | 71.99 | 332 |
| alternative-exact_augmented-cheap95-with_fixed_fallback | 71.00 | 71.99 | 332 |
| alternative-exact_augmented-match_ss_accuracy-learned_only | 71.20 | 79.04 | 372 |
| alternative-exact_augmented-match_ss_accuracy-with_fixed_fallback | 71.20 | 79.04 | 372 |
| alternative-expected_baseline-max_accuracy-learned_only | 72.00 | 96.54 | 460 |
| alternative-expected_baseline-max_accuracy-with_fixed_fallback | 72.00 | 96.54 | 460 |
| alternative-expected_baseline-ss_budget-learned_only | 72.00 | 96.54 | 460 |
| alternative-expected_baseline-ss_budget-with_fixed_fallback | 72.00 | 96.54 | 460 |
| alternative-expected_baseline-cheap95-learned_only | 71.00 | 71.89 | 331 |
| alternative-expected_baseline-cheap95-with_fixed_fallback | 71.00 | 71.89 | 331 |
| alternative-expected_baseline-match_ss_accuracy-learned_only | 70.80 | 79.60 | 375 |
| alternative-expected_baseline-match_ss_accuracy-with_fixed_fallback | 70.80 | 79.60 | 375 |
| alternative-gain_linear-max_accuracy-learned_only | 71.00 | 85.94 | 417 |
| alternative-gain_linear-max_accuracy-with_fixed_fallback | 71.00 | 85.94 | 417 |
| alternative-gain_linear-ss_budget-learned_only | 71.00 | 85.94 | 417 |
| alternative-gain_linear-ss_budget-with_fixed_fallback | 71.00 | 85.94 | 417 |
| alternative-gain_linear-cheap95-learned_only | 68.60 | 52.31 | 226 |
| alternative-gain_linear-cheap95-with_fixed_fallback | 68.60 | 52.31 | 226 |
| alternative-gain_linear-match_ss_accuracy-learned_only | 70.80 | 74.41 | 350 |
| alternative-gain_linear-match_ss_accuracy-with_fixed_fallback | 70.80 | 74.41 | 350 |
| alternative-gain_trees-max_accuracy-learned_only | 71.40 | 82.59 | 397 |
| alternative-gain_trees-max_accuracy-with_fixed_fallback | 71.40 | 82.59 | 397 |
| alternative-gain_trees-ss_budget-learned_only | 71.40 | 82.59 | 397 |
| alternative-gain_trees-ss_budget-with_fixed_fallback | 71.40 | 82.59 | 397 |
| alternative-gain_trees-cheap95-learned_only | 69.20 | 50.95 | 219 |
| alternative-gain_trees-cheap95-with_fixed_fallback | 69.20 | 50.95 | 219 |
| alternative-gain_trees-match_ss_accuracy-learned_only | 71.00 | 81.35 | 390 |
| alternative-gain_trees-match_ss_accuracy-with_fixed_fallback | 71.00 | 81.35 | 390 |
| alternative-group_centered-max_accuracy-learned_only | 70.80 | 78.61 | 369 |
| alternative-group_centered-max_accuracy-with_fixed_fallback | 70.80 | 78.61 | 369 |
| alternative-group_centered-ss_budget-learned_only | 70.80 | 78.61 | 369 |
| alternative-group_centered-ss_budget-with_fixed_fallback | 70.80 | 78.61 | 369 |
| alternative-group_centered-cheap95-learned_only | 70.80 | 70.69 | 324 |
| alternative-group_centered-cheap95-with_fixed_fallback | 70.80 | 70.69 | 324 |
| alternative-group_centered-match_ss_accuracy-learned_only | 70.80 | 70.70 | 324 |
| alternative-group_centered-match_ss_accuracy-with_fixed_fallback | 70.80 | 70.70 | 324 |
| alternative-outcome_linear-max_accuracy-learned_only | 71.20 | 85.61 | 396 |
| alternative-outcome_linear-max_accuracy-with_fixed_fallback | 71.20 | 85.61 | 396 |
| alternative-outcome_linear-ss_budget-learned_only | 71.20 | 85.61 | 396 |
| alternative-outcome_linear-ss_budget-with_fixed_fallback | 71.20 | 85.61 | 396 |
| alternative-outcome_linear-cheap95-learned_only | 69.60 | 55.63 | 239 |
| alternative-outcome_linear-cheap95-with_fixed_fallback | 69.60 | 55.63 | 239 |
| alternative-outcome_linear-match_ss_accuracy-learned_only | 71.00 | 76.15 | 344 |
| alternative-outcome_linear-match_ss_accuracy-with_fixed_fallback | 71.00 | 76.15 | 344 |
| alternative-outcome_residual-max_accuracy-learned_only | 71.60 | 89.80 | 419 |
| alternative-outcome_residual-max_accuracy-with_fixed_fallback | 71.60 | 89.80 | 419 |
| alternative-outcome_residual-ss_budget-learned_only | 71.60 | 89.80 | 419 |
| alternative-outcome_residual-ss_budget-with_fixed_fallback | 71.60 | 89.80 | 419 |
| alternative-outcome_residual-cheap95-learned_only | 70.40 | 62.63 | 280 |
| alternative-outcome_residual-cheap95-with_fixed_fallback | 70.40 | 62.63 | 280 |
| alternative-outcome_residual-match_ss_accuracy-learned_only | 71.00 | 78.59 | 355 |
| alternative-outcome_residual-match_ss_accuracy-with_fixed_fallback | 71.00 | 78.59 | 355 |
| alternative-outcome_trees-max_accuracy-learned_only | 72.40 | 96.77 | 453 |
| alternative-outcome_trees-max_accuracy-with_fixed_fallback | 72.40 | 96.77 | 453 |
| alternative-outcome_trees-ss_budget-learned_only | 72.40 | 96.77 | 453 |
| alternative-outcome_trees-ss_budget-with_fixed_fallback | 72.40 | 96.77 | 453 |
| alternative-outcome_trees-cheap95-learned_only | 69.80 | 66.05 | 291 |
| alternative-outcome_trees-cheap95-with_fixed_fallback | 69.80 | 66.05 | 291 |
| alternative-outcome_trees-match_ss_accuracy-learned_only | 71.20 | 82.77 | 380 |
| alternative-outcome_trees-match_ss_accuracy-with_fixed_fallback | 71.20 | 82.77 | 380 |
| alternative-global-max_accuracy-learned_only | 71.20 | 85.61 | 396 |
| alternative-global-max_accuracy-with_fixed_fallback | 71.20 | 85.61 | 396 |
| alternative-global-ss_budget-learned_only | 71.20 | 85.61 | 396 |
| alternative-global-ss_budget-with_fixed_fallback | 71.20 | 85.61 | 396 |
| alternative-global-cheap95-learned_only | 69.20 | 50.95 | 219 |
| alternative-global-cheap95-with_fixed_fallback | 69.20 | 50.95 | 219 |
| alternative-global-match_ss_accuracy-learned_only | 70.80 | 70.70 | 324 |
| alternative-global-match_ss_accuracy-with_fixed_fallback | 70.80 | 70.70 | 324 |

Validation constraints do not guarantee the same test constraint. Thresholds, regularization, architectures and checkpoints were locked before test construction. Exact denotes expected-reward enumeration, not exact-match grading. Statistical comparisons after repeated exploration are descriptive.

## Per-family maximum-accuracy selections

These configurations were chosen within each family on validation, not chosen by test accuracy. Selecting the family itself globally on validation gives a different result below.

| Method | Test lenient | Cost % SS | Correct /500 |
|---|---:|---:|---:|
| exact_augmented | 71.80% | 89.25 | 359 |
| expected_baseline | 72.00% | 96.54 | 360 |
| gain_linear | 71.00% | 85.94 | 355 |
| gain_trees | 71.40% | 82.59 | 357 |
| group_centered | 70.80% | 78.61 | 354 |
| outcome_linear | 71.20% | 85.61 | 356 |
| outcome_residual | 71.60% | 89.80 | 358 |
| outcome_trees | 72.40% | 96.77 | 362 |

## Global family selection on validation

| Validation objective | Chosen family | Test lenient | Cost % SS |
|---|---|---:|---:|
| max_accuracy | outcome_linear | 71.20% | 85.61 |
| ss_budget | outcome_linear | 71.20% | 85.61 |
| cheap95 | gain_trees | 69.20% | 50.95 |
| match_ss_accuracy | group_centered | 70.80% | 70.70 |

The per-family outcome-tree result is362/500 versus SS359/500, with cost96.77% SS. The expected-baseline policy is360/500 at96.54% SS. These are small observed gains, not evidence of a stable improvement. The global validation maximum-accuracy winner is outcome_linear (356/500,85.61% SS cost); do not relabel the best test family as the preselected global winner.

The augmented exact-reward family maximum-accuracy policy preserves359/500 at89.25% SS cost; its per-family match-validation-SS policy is356/500 at79.04% SS cost. Matching validation SS does not guarantee matching test SS.

[Accuracy/cost plot](accuracy_cost.png) · [Compact method table](method_summary.csv) · [Paired descriptive differences](verification.json). Test reuse and the many validation comparisons prevent confirmatory significance/equivalence claims.
