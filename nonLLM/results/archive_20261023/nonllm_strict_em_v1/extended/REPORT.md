> Corrected run: both legacy accuracy columns represent official normalized EM. See root protocol.json.

# Extended non-LLM policy experiments

144 seed-specific training runs; 144 three-seed checkpoint/configuration candidates evaluated on validation. 24 predeclared family/objective winners evaluated on the previously used descriptive test set.
Heads are linear over TF-IDF and numeric features. Qwen weights are unchanged. Every deployed policy sees one greedy weak answer, including augmented-training policies.
Training uses one, five or nine observation bundles per question; all questions have equal total weight. Maximum accuracy and SS-budget objectives are selected on validation, not test.
Costs are parameter-weighted token proxies including sunk weak calls, excluding CPU router overhead. Confidence costs assume extraction from normal generation logits.

SS test exact-match accuracy: **47.20%**.

| Family / objective | G | λ | Epoch | Val acc % | Val cost % SS | Test acc % | Test strict % | Test cost % SS | Δ vs SS (pp) | Paired 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base_exact_cheap95 | 0 | 0.0 | 30 | 44.59 | 67.87 | 43.89 | 43.89 | 66.53 | -3.31 | [-4.79, -1.89] |
| base_exact_max_accuracy | 0 | 0.15 | 120 | 47.03 | 115.04 | 46.66 | 46.66 | 114.28 | -0.54 | [-1.01, -0.14] |
| base_exact_ss_budget | 0 | 0.4 | 120 | 46.31 | 99.18 | 45.58 | 45.58 | 96.69 | -1.62 | [-2.70, -0.68] |
| base_grpo_cheap95 | 8 | 0.0 | 30 | 44.59 | 71.16 | 43.69 | 43.69 | 70.16 | -3.51 | [-5.00, -2.09] |
| base_grpo_max_accuracy | 32 | 0.0 | 120 | 47.21 | 115.63 | 46.46 | 46.46 | 115.46 | -0.74 | [-1.35, -0.14] |
| base_grpo_ss_budget | 8 | 0.0 | 30 | 46.13 | 90.04 | 44.90 | 44.90 | 89.35 | -2.30 | [-3.51, -1.15] |
| confidence_exact_cheap95 | 0 | 0.0 | 30 | 44.59 | 68.80 | 43.89 | 43.89 | 67.81 | -3.31 | [-4.79, -1.89] |
| confidence_exact_max_accuracy | 0 | 0.0 | 30 | 47.03 | 110.34 | 46.79 | 46.79 | 108.59 | -0.41 | [-1.15, +0.34] |
| confidence_exact_ss_budget | 0 | 0.15 | 30 | 46.13 | 99.68 | 45.58 | 45.58 | 95.80 | -1.62 | [-2.63, -0.68] |
| confidence_grpo_cheap95 | 32 | 0.4 | 60 | 44.59 | 72.83 | 43.35 | 43.35 | 72.93 | -3.85 | [-5.33, -2.43] |
| confidence_grpo_max_accuracy | 32 | 0.0 | 120 | 47.21 | 115.60 | 46.46 | 46.46 | 115.70 | -0.74 | [-1.42, -0.07] |
| confidence_grpo_ss_budget | 8 | 0.0 | 30 | 46.13 | 93.45 | 45.04 | 45.04 | 90.62 | -2.16 | [-3.31, -1.08] |
| aug5_exact_cheap95 | 0 | 0.0 | 30 | 44.59 | 66.76 | 43.75 | 43.75 | 65.76 | -3.44 | [-4.93, -2.03] |
| aug5_exact_max_accuracy | 0 | 0.0 | 120 | 47.12 | 116.13 | 46.39 | 46.39 | 115.54 | -0.81 | [-1.49, -0.20] |
| aug5_exact_ss_budget | 0 | 0.15 | 30 | 46.22 | 99.51 | 45.51 | 45.51 | 95.36 | -1.69 | [-2.70, -0.68] |
| aug5_grpo_cheap95 | 16 | 0.0 | 30 | 44.59 | 72.46 | 43.89 | 43.89 | 70.49 | -3.31 | [-4.79, -1.82] |
| aug5_grpo_max_accuracy | 8 | 0.0 | 60 | 47.12 | 116.56 | 46.79 | 46.79 | 116.62 | -0.41 | [-0.88, +0.07] |
| aug5_grpo_ss_budget | 8 | 0.0 | 30 | 46.04 | 88.56 | 44.97 | 44.97 | 88.06 | -2.23 | [-3.51, -1.01] |
| aug9_exact_cheap95 | 0 | 0.0 | 30 | 44.59 | 67.27 | 43.89 | 43.89 | 66.09 | -3.31 | [-4.79, -1.89] |
| aug9_exact_max_accuracy | 0 | 0.0 | 120 | 47.12 | 114.50 | 46.39 | 46.39 | 113.47 | -0.81 | [-1.55, -0.07] |
| aug9_exact_ss_budget | 0 | 0.4 | 60 | 46.13 | 98.83 | 45.64 | 45.64 | 96.96 | -1.55 | [-2.63, -0.61] |
| aug9_grpo_cheap95 | 16 | 0.0 | 30 | 44.59 | 72.38 | 43.89 | 43.89 | 70.43 | -3.31 | [-4.79, -1.82] |
| aug9_grpo_max_accuracy | 16 | 0.0 | 60 | 47.12 | 116.22 | 46.86 | 46.86 | 116.36 | -0.34 | [-0.81, +0.14] |
| aug9_grpo_ss_budget | 8 | 0.15 | 60 | 46.31 | 96.45 | 45.17 | 45.17 | 98.37 | -2.03 | [-3.04, -1.08] |

Global validation selections:

- cheap95: aug5_exact_cheap95
- max_accuracy: confidence_grpo_max_accuracy
- ss_budget: aug9_grpo_ss_budget

Training curves record validation expected accuracy/cost and GRPO zero-advantage group frequency every 10 epochs. Only epochs 30/60/120 participate in configuration selection.
This is a finite-action GRPO adaptation, with the same entropy regularizer as exact optimization. Increased validation search and previously examined test data limit confirmatory claims. Bootstrap intervals are descriptive and not adjusted for multiple comparisons.
