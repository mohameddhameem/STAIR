> Corrected run: both legacy accuracy columns represent official normalized EM. See root protocol.json.

# Non-LLM router improvement experiments

All hyperparameters and thresholds use validation only. Existing test is descriptive and has been examined previously.
Costs are parameter-weighted measured token proxies; CPU router overhead is excluded.
GRPO here is a two-stage finite-action adaptation on linear non-LLM policies, not an LLM training run.

Cheapest validation-feasible method: gain_trees.

| Method | Val accuracy % | Val cost % direct SS | Test exact-match % | Test strict % | Test cost % direct SS | Gain over cost-matched cascade-policy mixture (pp) |
|---|---:|---:|---:|---:|---:|---:|
| confidence_cost_sensitive_logreg | 44.59 | 78.79 | 44.63 | 44.63 | 79.06 | +1.30 |
| confidence_gain_trees | 44.59 | 73.84 | 44.29 | 44.29 | 74.49 | +1.38 |
| confidence_outcome_hist | 44.59 | 87.56 | 45.85 | 45.85 | 87.99 | +1.70 |
| confidence_outcome_logreg | 44.59 | 78.20 | 45.04 | 45.04 | 77.97 | +1.81 |
| confidence_rich_hist | 44.59 | 79.93 | 44.36 | 44.36 | 79.12 | +1.03 |
| confidence_rich_logreg | 44.59 | 83.74 | 44.90 | 44.90 | 81.94 | +1.31 |
| confidence_rich_trees | 44.59 | 77.00 | 44.50 | 44.50 | 75.41 | +1.50 |
| cost_sensitive_logreg | 44.59 | 78.79 | 44.63 | 44.63 | 79.06 | +1.30 |
| exact_reward_lambda0.0 | 44.59 | 75.36 | 44.09 | 44.09 | 74.26 | +1.20 |
| exact_reward_lambda0.15 | 44.59 | 78.71 | 44.29 | 44.29 | 77.33 | +1.12 |
| exact_reward_lambda0.4 | 44.59 | 81.17 | 43.96 | 43.96 | 80.77 | +0.47 |
| exact_reward_lambda0.8 | 44.59 | 88.93 | 44.09 | 44.09 | 87.43 | -0.00 |
| frozen_baseline | 44.59 | 76.76 | 43.62 | 43.62 | 77.16 | +0.46 |
| gain_trees | 44.59 | 73.69 | 44.36 | 44.36 | 74.38 | +1.46 |
| grpo_lambda0.0 | 44.59 | 86.77 | 44.56 | 44.56 | 85.37 | +0.66 |
| grpo_lambda0.15 | 44.59 | 82.11 | 43.69 | 43.69 | 81.16 | +0.17 |
| grpo_lambda0.4 | 44.59 | 82.54 | 43.75 | 43.75 | 81.48 | +0.20 |
| grpo_lambda0.8 | 44.59 | 83.30 | 43.82 | 43.82 | 81.80 | +0.24 |
| outcome_hist | 44.59 | 86.60 | 44.90 | 44.90 | 87.24 | +0.83 |
| outcome_logreg | 44.59 | 78.20 | 45.04 | 45.04 | 77.97 | +1.81 |
| rich_hist | 44.59 | 79.74 | 44.02 | 44.02 | 79.03 | +0.70 |
| rich_logreg | 44.59 | 83.74 | 44.90 | 44.90 | 81.94 | +1.31 |
| rich_trees | 44.59 | 77.00 | 44.50 | 44.50 | 75.42 | +1.50 |
| sampled_augmentation_hist | 44.59 | 86.48 | 45.17 | 45.17 | 87.01 | +1.12 |
| sampled_augmentation_logreg | 44.59 | 78.20 | 45.04 | 45.04 | 77.98 | +1.81 |
| sampled_consensus_hist | 44.59 | 90.67 | 45.78 | 45.78 | 91.18 | +1.34 |
| sampled_consensus_logreg | 44.59 | 81.36 | 45.04 | 45.04 | 81.15 | +1.52 |

Paired question-bootstrap intervals against the frozen baseline (descriptive, no multiple-comparison correction):

- confidence_rich_trees: accuracy difference +0.88 pp, 95% interval [-0.07, +1.89].
- outcome_hist: accuracy difference +1.28 pp, 95% interval [+0.07, +2.43].
- outcome_logreg: accuracy difference +1.42 pp, 95% interval [+0.14, +2.77].
- gain_trees: accuracy difference +0.74 pp, 95% interval [-0.20, +1.69].
- sampled_augmentation_logreg: accuracy difference +1.42 pp, 95% interval [+0.14, +2.77].
- rich_trees: accuracy difference +0.88 pp, 95% interval [-0.07, +1.89].
- exact_reward_lambda0.0: accuracy difference +0.47 pp, 95% interval [-0.61, +1.62].
- rich_hist: accuracy difference +0.41 pp, 95% interval [-0.47, +1.28].
- confidence_cost_sensitive_logreg: accuracy difference +1.01 pp, 95% interval [-0.20, +2.23].
- sampled_consensus_logreg: accuracy difference +1.42 pp, 95% interval [+0.14, +2.77].
- confidence_rich_hist: accuracy difference +0.74 pp, 95% interval [-0.14, +1.69].
- exact_reward_lambda0.8: accuracy difference +0.47 pp, 95% interval [-0.95, +1.82].
- grpo_lambda0.8: accuracy difference +0.20 pp, 95% interval [-1.22, +1.62].
- rich_logreg: accuracy difference +1.28 pp, 95% interval [+0.14, +2.43].
- confidence_outcome_hist: accuracy difference +2.23 pp, 95% interval [+1.01, +3.38].
- grpo_lambda0.0: accuracy difference +0.95 pp, 95% interval [-0.27, +2.16].
- exact_reward_lambda0.4: accuracy difference +0.34 pp, 95% interval [-1.01, +1.69].
- cost_sensitive_logreg: accuracy difference +1.01 pp, 95% interval [-0.20, +2.23].
- grpo_lambda0.4: accuracy difference +0.14 pp, 95% interval [-1.28, +1.55].
- confidence_outcome_logreg: accuracy difference +1.42 pp, 95% interval [+0.14, +2.77].
- exact_reward_lambda0.15: accuracy difference +0.68 pp, 95% interval [-0.61, +1.89].
- grpo_lambda0.15: accuracy difference +0.07 pp, 95% interval [-1.35, +1.49].
- confidence_gain_trees: accuracy difference +0.68 pp, 95% interval [-0.27, +1.69].
- sampled_augmentation_hist: accuracy difference +1.55 pp, 95% interval [+0.27, +2.77].
- sampled_consensus_hist: accuracy difference +2.16 pp, 95% interval [+0.95, +3.31].
- confidence_rich_logreg: accuracy difference +1.28 pp, 95% interval [+0.14, +2.43].
