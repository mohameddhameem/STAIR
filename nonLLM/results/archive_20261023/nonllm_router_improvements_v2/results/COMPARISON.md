# Non-LLM router improvements: complete comparison

Primary method selected on validation: **exact_reward_lambda0.0**.

Every configuration and its operating thresholds were fixed using train/validation. The previously used test set is descriptive; no new confirmatory holdout was collected.
All cost percentages use direct SS as denominator. Costs count sunk weak calls and any extra sampling, using parameter-weighted token proxies; CPU routing overhead is excluded.
Two fixed-policy frontiers are reported. Direct policies skip unused weak calls when their model choices are fixed; cascade policies still run weak first. The direct frontier is the stronger cost-matched comparator.
The previously quoted +1.44 pp for the frozen baseline applies only to the cascade fixed-policy frontier. Against direct fixed policies its descriptive gain is about -0.14 pp.

| Method | Val cost % SS | Test lenient % | Test strict % | Test cost % SS | Gain vs direct mixture (pp) | Gain vs cascade mixture (pp) |
|---|---:|---:|---:|---:|---:|---:|
| confidence_cost_sensitive_logreg | 81.78 | 56.72 | 45.17 | 80.33 | +0.23 | +1.89 |
| confidence_gain_trees | 78.03 | 56.38 | 44.56 | 76.19 | +0.44 | +1.98 |
| confidence_outcome_hist | 87.25 | 57.12 | 45.85 | 85.94 | -0.11 | +1.71 |
| confidence_outcome_logreg | 82.97 | 57.26 | 45.85 | 82.45 | +0.49 | +2.21 |
| confidence_rich_hist | 79.94 | 55.98 | 44.36 | 79.14 | -0.36 | +1.27 |
| confidence_rich_logreg | 81.10 | 56.65 | 44.77 | 79.57 | +0.26 | +1.90 |
| confidence_rich_trees | 77.25 | 56.04 | 44.50 | 76.03 | +0.12 | +1.66 |
| cost_sensitive_logreg | 81.71 | 56.72 | 45.10 | 80.29 | +0.23 | +1.89 |
| exact_reward_lambda0.0 | 77.11 | 55.57 | 44.43 | 73.69 | -0.04 | +1.43 |
| exact_reward_lambda0.15 | 80.29 | 55.91 | 44.50 | 79.41 | -0.46 | +1.17 |
| exact_reward_lambda0.4 | 83.86 | 55.77 | 44.29 | 82.28 | -0.97 | +0.74 |
| exact_reward_lambda0.8 | 84.95 | 55.23 | 43.82 | 83.02 | -1.61 | +0.13 |
| frozen_baseline | 80.76 | 55.98 | 44.56 | 77.53 | -0.14 | +1.44 |
| gain_trees | 78.03 | 56.45 | 44.63 | 76.17 | +0.51 | +2.05 |
| grpo_lambda0.0 | 83.81 | 56.18 | 44.56 | 82.94 | -0.65 | +1.08 |
| grpo_lambda0.15 | 83.65 | 55.64 | 44.23 | 82.24 | -1.10 | +0.61 |
| grpo_lambda0.4 | 84.14 | 55.10 | 43.89 | 83.26 | -1.78 | -0.03 |
| grpo_lambda0.8 | 84.31 | 55.03 | 43.82 | 83.21 | -1.84 | -0.10 |
| outcome_hist | 87.01 | 57.19 | 45.51 | 85.74 | -0.01 | +1.80 |
| outcome_logreg | 82.86 | 56.99 | 45.78 | 82.36 | +0.23 | +1.95 |
| rich_hist | 78.49 | 55.71 | 44.23 | 78.39 | -0.53 | +1.08 |
| rich_logreg | 81.09 | 56.65 | 44.77 | 79.57 | +0.26 | +1.90 |
| rich_trees | 80.14 | 55.98 | 44.50 | 78.74 | -0.30 | +1.31 |
| sampled_augmentation_hist | 87.21 | 57.19 | 45.78 | 85.94 | -0.04 | +1.78 |
| sampled_augmentation_logreg | 82.96 | 57.19 | 45.91 | 82.44 | +0.42 | +2.14 |
| sampled_consensus_hist | 90.25 | 57.12 | 45.78 | 88.99 | -0.51 | +1.40 |
| sampled_consensus_logreg | 86.03 | 57.26 | 45.91 | 85.52 | +0.09 | +1.89 |

![Accuracy and token-cost comparison](comparison.png)

GRPO is a finite-action group-relative PPO adaptation of linear TF-IDF/numeric policies, with three fixed seeds averaged. No LLM router or Qwen fine-tuning was performed.
Augmentation policies consume one greedy answer at deployment. Consensus policies consume one greedy plus four sampled weak answers and pay for all five.
Confidence statistics were collected with teacher forcing for cached greedy answers. The reported deployment cost assumes those statistics can be extracted from the ordinary weak generation logits. If a separate model forward pass is required in deployment, its cost must be added.
Paired accuracy intervals and raw model artifacts are available in REPORT.md and per-method folders. Differences across the full exploratory suite are not corrected for multiple comparisons.
