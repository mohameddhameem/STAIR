# Non-LLM router improvement experiments

All hyperparameters and thresholds use validation only. Existing test is descriptive and has been examined previously.
Costs are parameter-weighted measured token proxies; CPU router overhead is excluded.
GRPO here is a two-stage finite-action adaptation on linear non-LLM policies, not an LLM training run.

Cheapest validation-feasible method: exact_reward_lambda0.0.

| Method | Val accuracy % | Val cost % direct SS | Test lenient % | Test strict % | Test cost % direct SS | Gain over cost-matched fixed mixture (pp) |
|---|---:|---:|---:|---:|---:|---:|
| confidence_cost_sensitive_logreg | 56.76 | 81.78 | 56.72 | 45.17 | 80.33 | +1.89 |
| confidence_gain_trees | 56.76 | 78.03 | 56.38 | 44.56 | 76.19 | +1.98 |
| confidence_outcome_hist | 56.76 | 87.25 | 57.12 | 45.85 | 85.94 | +1.71 |
| confidence_outcome_logreg | 56.76 | 82.97 | 57.26 | 45.85 | 82.45 | +2.21 |
| confidence_rich_hist | 56.76 | 79.94 | 55.98 | 44.36 | 79.14 | +1.27 |
| confidence_rich_logreg | 56.76 | 81.10 | 56.65 | 44.77 | 79.57 | +1.90 |
| confidence_rich_trees | 56.76 | 77.25 | 56.04 | 44.50 | 76.03 | +1.66 |
| cost_sensitive_logreg | 56.76 | 81.71 | 56.72 | 45.10 | 80.29 | +1.89 |
| exact_reward_lambda0.0 | 56.76 | 77.11 | 55.57 | 44.43 | 73.69 | +1.43 |
| exact_reward_lambda0.15 | 56.76 | 80.29 | 55.91 | 44.50 | 79.41 | +1.17 |
| exact_reward_lambda0.4 | 56.76 | 83.86 | 55.77 | 44.29 | 82.28 | +0.74 |
| exact_reward_lambda0.8 | 56.76 | 84.95 | 55.23 | 43.82 | 83.02 | +0.13 |
| frozen_baseline | 56.76 | 80.76 | 55.98 | 44.56 | 77.53 | +1.44 |
| gain_trees | 56.76 | 78.03 | 56.45 | 44.63 | 76.17 | +2.05 |
| grpo_lambda0.0 | 56.76 | 83.81 | 56.18 | 44.56 | 82.94 | +1.08 |
| grpo_lambda0.15 | 56.76 | 83.65 | 55.64 | 44.23 | 82.24 | +0.61 |
| grpo_lambda0.4 | 56.76 | 84.14 | 55.10 | 43.89 | 83.26 | -0.03 |
| grpo_lambda0.8 | 56.76 | 84.31 | 55.03 | 43.82 | 83.21 | -0.10 |
| outcome_hist | 56.76 | 87.01 | 57.19 | 45.51 | 85.74 | +1.80 |
| outcome_logreg | 56.76 | 82.86 | 56.99 | 45.78 | 82.36 | +1.95 |
| rich_hist | 56.76 | 78.49 | 55.71 | 44.23 | 78.39 | +1.08 |
| rich_logreg | 56.76 | 81.09 | 56.65 | 44.77 | 79.57 | +1.90 |
| rich_trees | 56.76 | 80.14 | 55.98 | 44.50 | 78.74 | +1.31 |
| sampled_augmentation_hist | 56.76 | 87.21 | 57.19 | 45.78 | 85.94 | +1.78 |
| sampled_augmentation_logreg | 56.76 | 82.96 | 57.19 | 45.91 | 82.44 | +2.14 |
| sampled_consensus_hist | 56.76 | 90.25 | 57.12 | 45.78 | 88.99 | +1.40 |
| sampled_consensus_logreg | 56.76 | 86.03 | 57.26 | 45.91 | 85.52 | +1.89 |

Paired question-bootstrap intervals against the frozen baseline (descriptive, no multiple-comparison correction):

- confidence_rich_trees: accuracy difference +0.07 pp, 95% interval [-0.95, +1.15].
- outcome_hist: accuracy difference +1.22 pp, 95% interval [-0.07, +2.63].
- outcome_logreg: accuracy difference +1.01 pp, 95% interval [-0.20, +2.30].
- gain_trees: accuracy difference +0.47 pp, 95% interval [-0.68, +1.62].
- sampled_augmentation_logreg: accuracy difference +1.22 pp, 95% interval [+0.07, +2.50].
- rich_trees: accuracy difference +0.00 pp, 95% interval [-1.08, +1.15].
- exact_reward_lambda0.0: accuracy difference -0.41 pp, 95% interval [-1.55, +0.74].
- rich_hist: accuracy difference -0.27 pp, 95% interval [-1.55, +1.01].
- confidence_cost_sensitive_logreg: accuracy difference +0.74 pp, 95% interval [-0.47, +1.96].
- sampled_consensus_logreg: accuracy difference +1.28 pp, 95% interval [+0.07, +2.50].
- confidence_rich_hist: accuracy difference +0.00 pp, 95% interval [-1.28, +1.22].
- exact_reward_lambda0.8: accuracy difference -0.74 pp, 95% interval [-2.23, +0.74].
- grpo_lambda0.8: accuracy difference -0.95 pp, 95% interval [-2.43, +0.54].
- rich_logreg: accuracy difference +0.68 pp, 95% interval [-0.41, +1.76].
- confidence_outcome_hist: accuracy difference +1.15 pp, 95% interval [-0.14, +2.57].
- grpo_lambda0.0: accuracy difference +0.20 pp, 95% interval [-0.95, +1.42].
- exact_reward_lambda0.4: accuracy difference -0.20 pp, 95% interval [-1.55, +1.08].
- cost_sensitive_logreg: accuracy difference +0.74 pp, 95% interval [-0.47, +1.96].
- grpo_lambda0.4: accuracy difference -0.88 pp, 95% interval [-2.36, +0.61].
- confidence_outcome_logreg: accuracy difference +1.28 pp, 95% interval [+0.14, +2.57].
- exact_reward_lambda0.15: accuracy difference -0.07 pp, 95% interval [-1.28, +1.15].
- grpo_lambda0.15: accuracy difference -0.34 pp, 95% interval [-1.82, +1.08].
- confidence_gain_trees: accuracy difference +0.41 pp, 95% interval [-0.74, +1.55].
- sampled_augmentation_hist: accuracy difference +1.22 pp, 95% interval [-0.07, +2.63].
- sampled_consensus_hist: accuracy difference +1.15 pp, 95% interval [-0.14, +2.57].
- confidence_rich_logreg: accuracy difference +0.68 pp, 95% interval [-0.41, +1.76].
