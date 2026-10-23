# Extended non-LLM policy experiments

144 seed-specific training runs; 144 three-seed checkpoint/configuration candidates evaluated on validation. 24 predeclared family/objective winners evaluated on the previously used descriptive test set.
Heads are linear over TF-IDF and numeric features. Qwen weights are unchanged. Every deployed policy sees one greedy weak answer, including augmented-training policies.
Training uses one, five or nine observation bundles per question; all questions have equal total weight. Maximum accuracy and SS-budget objectives are selected on validation, not test.
Costs are parameter-weighted token proxies including sunk weak calls, excluding CPU router overhead. Confidence costs assume extraction from normal generation logits.

SS test lenient accuracy: **52.94%**.

| Family / objective | G | λ | Epoch | Val acc % | Val cost % SS | Test acc % | Test strict % | Test cost % SS | Δ vs SS (pp) | Paired 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_exact_cheap95 | 0 | 0.0 | 30 | 49.91 | 68.21 | 49.16 | 43.82 | 67.36 | -3.78 | [-5.20, -2.36] |
| base_exact_max_accuracy | 0 | 0.0 | 120 | 52.70 | 115.16 | 51.92 | 46.19 | 114.47 | -1.01 | [-1.69, -0.41] |
| base_exact_ss_budget | 0 | 0.4 | 120 | 51.98 | 99.33 | 51.59 | 45.85 | 96.05 | -1.35 | [-2.30, -0.41] |
| base_grpo_cheap95 | 8 | 0.0 | 30 | 49.91 | 69.24 | 49.02 | 43.69 | 67.45 | -3.92 | [-5.47, -2.43] |
| base_grpo_max_accuracy | 16 | 0.15 | 30 | 52.79 | 120.25 | 51.86 | 46.59 | 119.92 | -1.08 | [-1.76, -0.41] |
| base_grpo_ss_budget | 16 | 0.4 | 60 | 51.62 | 99.35 | 51.86 | 45.98 | 101.75 | -1.08 | [-1.89, -0.27] |
| confidence_exact_cheap95 | 0 | 0.0 | 30 | 49.91 | 68.04 | 49.02 | 43.69 | 66.72 | -3.92 | [-5.40, -2.50] |
| confidence_exact_max_accuracy | 0 | 0.0 | 120 | 52.61 | 114.59 | 52.13 | 46.39 | 113.55 | -0.81 | [-1.42, -0.20] |
| confidence_exact_ss_budget | 0 | 0.15 | 120 | 51.89 | 94.91 | 51.18 | 45.58 | 91.20 | -1.76 | [-2.77, -0.68] |
| confidence_grpo_cheap95 | 8 | 0.0 | 30 | 49.91 | 69.19 | 49.02 | 43.69 | 67.33 | -3.92 | [-5.47, -2.43] |
| confidence_grpo_max_accuracy | 32 | 0.4 | 120 | 52.70 | 119.53 | 52.67 | 46.93 | 118.92 | -0.27 | [-0.54, -0.07] |
| confidence_grpo_ss_budget | 16 | 0.15 | 60 | 51.53 | 97.04 | 51.38 | 45.71 | 98.87 | -1.55 | [-2.50, -0.68] |
| aug5_exact_cheap95 | 0 | 0.0 | 30 | 49.91 | 69.23 | 49.09 | 43.82 | 67.99 | -3.85 | [-5.40, -2.43] |
| aug5_exact_max_accuracy | 0 | 0.0 | 120 | 52.61 | 114.51 | 52.33 | 46.59 | 113.76 | -0.61 | [-1.22, -0.07] |
| aug5_exact_ss_budget | 0 | 0.15 | 120 | 51.89 | 93.29 | 51.18 | 45.51 | 89.86 | -1.76 | [-2.84, -0.74] |
| aug5_grpo_cheap95 | 32 | 0.0 | 30 | 49.91 | 69.56 | 49.36 | 43.89 | 68.18 | -3.58 | [-5.13, -2.09] |
| aug5_grpo_max_accuracy | 8 | 0.4 | 120 | 52.70 | 120.32 | 52.73 | 47.06 | 119.78 | -0.20 | [-0.47, +0.00] |
| aug5_grpo_ss_budget | 8 | 0.15 | 120 | 51.53 | 98.55 | 51.32 | 45.58 | 98.53 | -1.62 | [-2.50, -0.74] |
| aug9_exact_cheap95 | 0 | 0.0 | 30 | 49.91 | 69.31 | 49.16 | 43.89 | 68.25 | -3.78 | [-5.33, -2.36] |
| aug9_exact_max_accuracy | 0 | 0.4 | 30 | 52.61 | 120.83 | 52.67 | 47.06 | 120.47 | -0.27 | [-0.54, -0.07] |
| aug9_exact_ss_budget | 0 | 0.15 | 120 | 51.89 | 94.31 | 51.25 | 45.51 | 90.88 | -1.69 | [-2.70, -0.68] |
| aug9_grpo_cheap95 | 8 | 0.0 | 30 | 49.91 | 70.92 | 49.56 | 44.16 | 69.97 | -3.38 | [-4.86, -1.96] |
| aug9_grpo_max_accuracy | 16 | 0.0 | 60 | 52.70 | 121.40 | 52.73 | 47.00 | 121.41 | -0.20 | [-0.54, +0.07] |
| aug9_grpo_ss_budget | 8 | 0.4 | 60 | 51.62 | 96.10 | 50.78 | 44.90 | 97.58 | -2.16 | [-3.17, -1.22] |

Global validation selections:

- cheap95: confidence_exact_cheap95
- max_accuracy: base_grpo_max_accuracy
- ss_budget: base_exact_ss_budget

Training curves record validation expected accuracy/cost and GRPO zero-advantage group frequency every 10 epochs. Only epochs 30/60/120 participate in configuration selection.
This is a finite-action GRPO adaptation, with the same entropy regularizer as exact optimization. Increased validation search and previously examined test data limit confirmatory claims. Bootstrap intervals are descriptive and not adjusted for multiple comparisons.
