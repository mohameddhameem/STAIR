# Non-LLM router improvement experiments

All hyperparameters and thresholds use validation only. Existing test is descriptive and has been examined previously.
Costs are parameter-weighted measured token proxies; CPU router overhead is excluded.
GRPO here is a two-stage finite-action adaptation on linear non-LLM policies, not an LLM training run.

Cheapest validation-feasible method: exact_reward_lambda0.0.

| Method | Val accuracy % | Val cost % direct SS | Test lenient % | Test strict % | Test cost % direct SS |
|---|---:|---:|---:|---:|---:|
| exact_reward_lambda0.0 | 56.76 | 77.27 | 55.57 | 44.50 | 73.81 |
| exact_reward_lambda0.15 | 56.76 | 80.79 | 55.71 | 44.29 | 79.91 |
| exact_reward_lambda0.4 | 56.76 | 82.92 | 55.71 | 44.29 | 81.13 |
| exact_reward_lambda0.8 | 56.76 | 84.94 | 55.37 | 43.89 | 82.76 |
| frozen_baseline | 56.76 | 80.76 | 55.98 | 44.56 | 77.53 |
| gain_trees | 56.76 | 78.03 | 56.45 | 44.63 | 76.17 |
| grpo_lambda0.0 | 56.76 | 80.76 | 56.04 | 44.63 | 79.89 |
| grpo_lambda0.15 | 56.76 | 84.29 | 55.23 | 43.75 | 83.56 |
| grpo_lambda0.4 | 56.76 | 83.95 | 55.30 | 43.82 | 82.72 |
| grpo_lambda0.8 | 56.76 | 84.46 | 55.50 | 43.96 | 83.42 |
| outcome_hist | 56.76 | 87.01 | 57.19 | 45.51 | 85.74 |
| outcome_logreg | 56.76 | 82.86 | 56.99 | 45.78 | 82.36 |
| rich_hist | 56.76 | 78.49 | 55.71 | 44.23 | 78.39 |
| rich_logreg | 56.76 | 81.09 | 56.65 | 44.77 | 79.57 |
| rich_trees | 56.76 | 80.14 | 55.98 | 44.50 | 78.74 |

Paired question-bootstrap intervals against the frozen baseline (descriptive, no multiple-comparison correction):

- outcome_hist: accuracy difference +1.22 pp, 95% interval [-0.07, +2.63].
- outcome_logreg: accuracy difference +1.01 pp, 95% interval [-0.20, +2.30].
- gain_trees: accuracy difference +0.47 pp, 95% interval [-0.68, +1.62].
- rich_trees: accuracy difference +0.00 pp, 95% interval [-1.08, +1.15].
- exact_reward_lambda0.0: accuracy difference -0.41 pp, 95% interval [-1.55, +0.74].
- rich_hist: accuracy difference -0.27 pp, 95% interval [-1.55, +1.01].
- exact_reward_lambda0.8: accuracy difference -0.61 pp, 95% interval [-2.09, +0.88].
- grpo_lambda0.8: accuracy difference -0.47 pp, 95% interval [-1.96, +1.01].
- rich_logreg: accuracy difference +0.68 pp, 95% interval [-0.41, +1.76].
- grpo_lambda0.0: accuracy difference +0.07 pp, 95% interval [-1.08, +1.28].
- exact_reward_lambda0.4: accuracy difference -0.27 pp, 95% interval [-1.62, +1.01].
- grpo_lambda0.4: accuracy difference -0.68 pp, 95% interval [-2.16, +0.81].
- exact_reward_lambda0.15: accuracy difference -0.27 pp, 95% interval [-1.49, +0.95].
- grpo_lambda0.15: accuracy difference -0.74 pp, 95% interval [-2.23, +0.68].
