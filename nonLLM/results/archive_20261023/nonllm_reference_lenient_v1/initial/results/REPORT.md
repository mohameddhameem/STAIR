# Non-LLM router improvement experiments

All hyperparameters and thresholds use validation only. Existing test is descriptive and has been examined previously.
Costs are parameter-weighted measured token proxies; CPU router overhead is excluded.
GRPO here is a two-stage finite-action adaptation on linear non-LLM policies, not an LLM training run.

Cheapest validation-feasible method: gain_trees.

| Method | Val accuracy % | Val cost % direct SS | Test lenient % | Test strict % | Test cost % direct SS | Gain over cost-matched cascade-policy mixture (pp) |
|---|---:|---:|---:|---:|---:|---:|
| confidence_cost_sensitive_logreg | 49.91 | 78.19 | 49.83 | 44.29 | 76.11 | +1.39 |
| confidence_gain_trees | 49.91 | 75.63 | 50.10 | 44.50 | 75.71 | +1.70 |
| confidence_outcome_hist | 49.91 | 81.89 | 51.11 | 45.44 | 81.77 | +2.11 |
| confidence_outcome_logreg | 49.91 | 80.67 | 50.91 | 45.10 | 80.53 | +2.03 |
| confidence_rich_hist | 49.91 | 78.32 | 49.90 | 44.29 | 78.31 | +1.24 |
| confidence_rich_logreg | 49.91 | 83.83 | 50.78 | 44.97 | 82.00 | +1.75 |
| confidence_rich_trees | 49.91 | 76.89 | 49.97 | 44.29 | 75.34 | +1.60 |
| cost_sensitive_logreg | 49.91 | 78.17 | 49.83 | 44.29 | 76.11 | +1.39 |
| exact_reward_lambda0.0 | 49.91 | 76.06 | 49.76 | 44.56 | 74.96 | +1.44 |
| exact_reward_lambda0.15 | 49.91 | 79.98 | 49.83 | 44.43 | 78.74 | +1.13 |
| exact_reward_lambda0.4 | 49.91 | 80.24 | 49.63 | 44.02 | 79.50 | +0.85 |
| exact_reward_lambda0.8 | 49.91 | 86.21 | 49.63 | 44.09 | 85.00 | +0.30 |
| frozen_baseline | 49.91 | 80.38 | 50.03 | 44.16 | 79.36 | +1.27 |
| gain_trees | 49.91 | 75.36 | 49.70 | 43.96 | 75.51 | +1.31 |
| grpo_lambda0.0 | 49.91 | 84.27 | 50.30 | 44.90 | 82.93 | +1.19 |
| grpo_lambda0.15 | 49.91 | 82.11 | 49.63 | 43.89 | 81.41 | +0.66 |
| grpo_lambda0.4 | 49.91 | 82.38 | 49.56 | 43.82 | 81.04 | +0.63 |
| grpo_lambda0.8 | 49.91 | 81.69 | 49.49 | 43.75 | 79.89 | +0.68 |
| outcome_hist | 49.91 | 84.06 | 51.25 | 45.37 | 84.14 | +2.01 |
| outcome_logreg | 49.91 | 80.49 | 50.64 | 45.04 | 80.39 | +1.77 |
| rich_hist | 49.91 | 78.28 | 49.97 | 44.16 | 78.25 | +1.31 |
| rich_logreg | 49.91 | 83.82 | 50.78 | 44.97 | 82.00 | +1.75 |
| rich_trees | 49.91 | 77.33 | 50.03 | 44.36 | 76.11 | +1.59 |
| sampled_augmentation_hist | 49.91 | 84.29 | 51.32 | 45.71 | 84.33 | +2.06 |
| sampled_augmentation_logreg | 49.91 | 80.67 | 50.98 | 45.10 | 80.53 | +2.10 |
| sampled_consensus_hist | 49.91 | 87.30 | 51.32 | 45.71 | 87.34 | +1.76 |
| sampled_consensus_logreg | 49.91 | 83.78 | 51.05 | 45.10 | 83.65 | +1.86 |

Paired question-bootstrap intervals against the frozen baseline (descriptive, no multiple-comparison correction):

- confidence_rich_trees: accuracy difference -0.07 pp, 95% interval [-1.01, +0.95].
- outcome_hist: accuracy difference +1.22 pp, 95% interval [+0.07, +2.36].
- outcome_logreg: accuracy difference +0.61 pp, 95% interval [-0.68, +1.89].
- gain_trees: accuracy difference -0.34 pp, 95% interval [-1.35, +0.74].
- sampled_augmentation_logreg: accuracy difference +0.95 pp, 95% interval [-0.27, +2.23].
- rich_trees: accuracy difference +0.00 pp, 95% interval [-0.95, +1.01].
- exact_reward_lambda0.0: accuracy difference -0.27 pp, 95% interval [-1.35, +0.81].
- rich_hist: accuracy difference -0.07 pp, 95% interval [-1.01, +0.88].
- confidence_cost_sensitive_logreg: accuracy difference -0.20 pp, 95% interval [-1.28, +0.95].
- sampled_consensus_logreg: accuracy difference +1.01 pp, 95% interval [-0.20, +2.23].
- confidence_rich_hist: accuracy difference -0.14 pp, 95% interval [-0.95, +0.74].
- exact_reward_lambda0.8: accuracy difference -0.41 pp, 95% interval [-1.82, +0.95].
- grpo_lambda0.8: accuracy difference -0.54 pp, 95% interval [-2.09, +1.01].
- rich_logreg: accuracy difference +0.74 pp, 95% interval [-0.34, +1.89].
- confidence_outcome_hist: accuracy difference +1.08 pp, 95% interval [-0.07, +2.16].
- grpo_lambda0.0: accuracy difference +0.27 pp, 95% interval [-0.88, +1.49].
- exact_reward_lambda0.4: accuracy difference -0.41 pp, 95% interval [-1.69, +0.88].
- cost_sensitive_logreg: accuracy difference -0.20 pp, 95% interval [-1.28, +0.95].
- grpo_lambda0.4: accuracy difference -0.47 pp, 95% interval [-2.03, +1.01].
- confidence_outcome_logreg: accuracy difference +0.88 pp, 95% interval [-0.34, +2.16].
- exact_reward_lambda0.15: accuracy difference -0.20 pp, 95% interval [-1.35, +0.95].
- grpo_lambda0.15: accuracy difference -0.41 pp, 95% interval [-1.96, +1.08].
- confidence_gain_trees: accuracy difference +0.07 pp, 95% interval [-0.81, +1.08].
- sampled_augmentation_hist: accuracy difference +1.28 pp, 95% interval [+0.20, +2.36].
- sampled_consensus_hist: accuracy difference +1.28 pp, 95% interval [+0.20, +2.36].
- confidence_rich_logreg: accuracy difference +0.74 pp, 95% interval [-0.34, +1.89].
