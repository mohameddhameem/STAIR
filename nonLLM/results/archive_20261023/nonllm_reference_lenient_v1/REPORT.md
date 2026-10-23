# Non-LLM routers: review-validated lenient rerun

**Primary metric: review-validated reference-lenient reconstruction, not exact match.**
All 73 method/operating-point records were retrained and selected using lenient targets/rewards/validation accuracy. Strict EM is diagnostic only.
The scorer agrees with all 4267 available original answer-review judgments. The original scoring source is unavailable; full equivalence on unobserved cases is not certified.
Regenerated Qwen2.5 Instruct outputs and matching labels form one cohort. Original reference outcomes remain separate: SS validation 52.79%, test 53.54%.
Token-price costs include sunk weak calls and consensus answers; CPU router overhead is excluded. Previously examined test split is exploratory.

Regenerated cohort SS: validation **52.52%**, test **52.94%**.

| Method | Test lenient % | Strict EM diagnostic % | Cost % SS |
|---|---:|---:|---:|
| Original / majority | 31.60 | 29.03 | 21.92 |
| Original / logreg | 39.23 | 35.38 | 45.48 |
| Original / linear_svm | 39.64 | 35.72 | 46.33 |
| Original / multinomial_nb | 36.12 | 33.09 | 42.96 |
| Original / random_forest | 40.78 | 37.14 | 64.69 |
| October full cascade | 50.03 | 44.16 | 79.36 |
| October no-answer ablation | 49.90 | 43.96 | 77.67 |
| Initial / confidence_rich_trees | 49.97 | 44.29 | 75.34 |
| Initial / frozen_baseline | 50.03 | 44.16 | 79.36 |
| Initial / outcome_hist | 51.25 | 45.37 | 84.14 |
| Initial / outcome_logreg | 50.64 | 45.04 | 80.39 |
| Initial / gain_trees | 49.70 | 43.96 | 75.51 |
| Initial / sampled_augmentation_logreg | 50.98 | 45.10 | 80.53 |
| Initial / rich_trees | 50.03 | 44.36 | 76.11 |
| Initial / exact_reward_lambda0.0 | 49.76 | 44.56 | 74.96 |
| Initial / rich_hist | 49.97 | 44.16 | 78.25 |
| Initial / confidence_cost_sensitive_logreg | 49.83 | 44.29 | 76.11 |
| Initial / sampled_consensus_logreg | 51.05 | 45.10 | 83.65 |
| Initial / confidence_rich_hist | 49.90 | 44.29 | 78.31 |
| Initial / exact_reward_lambda0.8 | 49.63 | 44.09 | 85.00 |
| Initial / grpo_lambda0.8 | 49.49 | 43.75 | 79.89 |
| Initial / rich_logreg | 50.78 | 44.97 | 82.00 |
| Initial / confidence_outcome_hist | 51.11 | 45.44 | 81.77 |
| Initial / grpo_lambda0.0 | 50.30 | 44.90 | 82.93 |
| Initial / exact_reward_lambda0.4 | 49.63 | 44.02 | 79.50 |
| Initial / cost_sensitive_logreg | 49.83 | 44.29 | 76.11 |
| Initial / grpo_lambda0.4 | 49.56 | 43.82 | 81.04 |
| Initial / confidence_outcome_logreg | 50.91 | 45.10 | 80.53 |
| Initial / exact_reward_lambda0.15 | 49.83 | 44.43 | 78.74 |
| Initial / grpo_lambda0.15 | 49.63 | 43.89 | 81.41 |
| Initial / confidence_gain_trees | 50.10 | 44.50 | 75.71 |
| Initial / sampled_augmentation_hist | 51.32 | 45.71 | 84.33 |
| Initial / sampled_consensus_hist | 51.32 | 45.71 | 87.34 |
| Initial / confidence_rich_logreg | 50.78 | 44.97 | 82.00 |
| Extended / base_exact_cheap95 | 49.16 | 43.82 | 67.36 |
| Extended / base_exact_max_accuracy | 51.92 | 46.19 | 114.47 |
| Extended / base_exact_ss_budget | 51.59 | 45.85 | 96.05 |
| Extended / base_grpo_cheap95 | 49.02 | 43.69 | 67.45 |
| Extended / base_grpo_max_accuracy | 51.86 | 46.59 | 119.92 |
| Extended / base_grpo_ss_budget | 51.86 | 45.98 | 101.75 |
| Extended / confidence_exact_cheap95 | 49.02 | 43.69 | 66.72 |
| Extended / confidence_exact_max_accuracy | 52.13 | 46.39 | 113.55 |
| Extended / confidence_exact_ss_budget | 51.18 | 45.58 | 91.20 |
| Extended / confidence_grpo_cheap95 | 49.02 | 43.69 | 67.33 |
| Extended / confidence_grpo_max_accuracy | 52.67 | 46.93 | 118.92 |
| Extended / confidence_grpo_ss_budget | 51.38 | 45.71 | 98.87 |
| Extended / aug5_exact_cheap95 | 49.09 | 43.82 | 67.99 |
| Extended / aug5_exact_max_accuracy | 52.33 | 46.59 | 113.76 |
| Extended / aug5_exact_ss_budget | 51.18 | 45.51 | 89.86 |
| Extended / aug5_grpo_cheap95 | 49.36 | 43.89 | 68.18 |
| Extended / aug5_grpo_max_accuracy | 52.73 | 47.06 | 119.78 |
| Extended / aug5_grpo_ss_budget | 51.32 | 45.58 | 98.53 |
| Extended / aug9_exact_cheap95 | 49.16 | 43.89 | 68.25 |
| Extended / aug9_exact_max_accuracy | 52.67 | 47.06 | 120.47 |
| Extended / aug9_exact_ss_budget | 51.25 | 45.51 | 90.88 |
| Extended / aug9_grpo_cheap95 | 49.56 | 44.16 | 69.97 |
| Extended / aug9_grpo_max_accuracy | 52.73 | 47.00 | 121.41 |
| Extended / aug9_grpo_ss_budget | 50.78 | 44.90 | 97.58 |
| Architecture / direct4_max_accuracy | 52.94 | 47.20 | 100.00 |
| Architecture / direct4_ss_budget | 52.94 | 47.20 | 100.00 |
| Architecture / direct4_cheap95 | 52.19 | 46.52 | 93.08 |
| Architecture / pre_ret_qa_max_accuracy | 52.94 | 47.20 | 100.00 |
| Architecture / pre_ret_qa_ss_budget | 52.94 | 47.20 | 100.00 |
| Architecture / pre_ret_qa_cheap95 | 51.72 | 46.32 | 93.41 |
| Architecture / fixed_S_qa_max_accuracy | 52.94 | 47.20 | 100.00 |
| Architecture / fixed_S_qa_ss_budget | 52.94 | 47.20 | 100.00 |
| Architecture / fixed_S_qa_cheap95 | 51.99 | 46.46 | 99.65 |
| Architecture / post_answer_chooser_max_accuracy | 53.00 | 47.33 | 100.56 |
| Architecture / post_answer_chooser_ss_budget | 52.94 | 47.20 | 100.00 |
| Architecture / post_answer_chooser_cheap95 | 52.94 | 47.20 | 100.00 |
| Combined / max_accuracy | 52.94 | 47.20 | 98.94 |
| Combined / ss_budget | 52.94 | 47.20 | 98.94 |
| Combined / cheap95 | 52.19 | 46.52 | 93.08 |
