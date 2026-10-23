# Extended non-LLM policy experiments

54 seed-specific training runs; 108 three-seed checkpoint/configuration candidates evaluated on validation. 12 predeclared family/objective winners evaluated on the previously used descriptive test set.
Heads are linear over TF-IDF and numeric features. Qwen weights are unchanged. Every deployed policy sees one greedy weak answer, including augmented-training policies.
Training uses one, five or nine observation bundles per question; all questions have equal total weight. Maximum accuracy and SS-budget objectives are selected on validation, not test.
Costs are parameter-weighted token proxies including sunk weak calls, excluding CPU router overhead. Confidence costs assume extraction from normal generation logits.

SS test lenient accuracy: **52.94%**.

| Family / objective | G | λ | Epoch | Val acc % | Val cost % SS | Test acc % | Test strict % | Test cost % SS | Δ vs SS (pp) | Paired 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_exact_cheap95 | 0 | 0.0 | 30 | 49.91 | 68.21 | 49.16 | 43.82 | 67.36 | -3.78 | [-5.20, -2.36] |
| base_exact_max_accuracy | 0 | 0.0 | 120 | 52.70 | 115.16 | 51.92 | 46.19 | 114.47 | -1.01 | [-1.69, -0.41] |
| base_exact_ss_budget | 0 | 0.15 | 240 | 52.07 | 93.99 | 50.84 | 45.24 | 90.35 | -2.09 | [-3.17, -1.01] |
| base_grpo_cheap95 | 16 | 0.0 | 240 | 49.91 | 70.05 | 49.49 | 44.02 | 69.06 | -3.44 | [-4.93, -1.96] |
| base_grpo_max_accuracy | 16 | 0.15 | 30 | 52.79 | 120.25 | 51.86 | 46.59 | 119.92 | -1.08 | [-1.76, -0.41] |
| base_grpo_ss_budget | 32 | 0.15 | 300 | 52.25 | 99.95 | 51.86 | 45.98 | 100.16 | -1.08 | [-1.96, -0.27] |
| aug9_exact_cheap95 | 0 | 0.0 | 30 | 49.91 | 69.31 | 49.16 | 43.89 | 68.25 | -3.78 | [-5.33, -2.36] |
| aug9_exact_max_accuracy | 0 | 0.0 | 180 | 52.70 | 117.70 | 52.46 | 46.79 | 117.92 | -0.47 | [-1.01, +0.07] |
| aug9_exact_ss_budget | 0 | 0.4 | 300 | 52.07 | 99.93 | 51.79 | 45.91 | 96.51 | -1.15 | [-2.09, -0.20] |
| aug9_grpo_cheap95 | 32 | 0.0 | 30 | 49.91 | 70.99 | 49.43 | 43.96 | 70.22 | -3.51 | [-5.06, -2.03] |
| aug9_grpo_max_accuracy | 16 | 0.15 | 300 | 52.70 | 118.15 | 52.40 | 46.86 | 118.32 | -0.54 | [-0.95, -0.20] |
| aug9_grpo_ss_budget | 32 | 0.4 | 240 | 51.89 | 97.97 | 51.25 | 45.51 | 99.06 | -1.69 | [-2.63, -0.74] |

Global validation selections:

- cheap95: base_exact_cheap95
- max_accuracy: base_grpo_max_accuracy
- ss_budget: base_grpo_ss_budget

Training curves record validation expected accuracy/cost and GRPO zero-advantage group frequency every 10 epochs. Only epochs 30/60/120/180/240/300 participate in configuration selection.
This is a finite-action GRPO adaptation, with the same entropy regularizer as exact optimization. Increased validation search and previously examined test data limit confirmatory claims. Bootstrap intervals are descriptive and not adjusted for multiple comparisons.
