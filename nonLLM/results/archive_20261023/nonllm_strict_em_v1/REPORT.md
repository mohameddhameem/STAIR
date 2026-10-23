# Corrected non-LLM router experiments

**Primary metric: official HotpotQA normalized answer exact match. All labels, sampled-answer rewards and operating points were rebuilt.**
Qwen2.5-1.5B/7B Instruct FP16 outputs are reused; no Base generations. No LLM weights were trained.
All cost percentages use the same direct-SS token ruler; sunk weak calls and consensus samples are included; CPU overhead excluded.
Existing test is exploratory. Original reference 52.79% is validation exact-match matching, not an EM target.

Direct SS validation EM: **46.94%**; test EM: **47.20%**.

| Method | Test EM % | Cost % SS |
|---|---:|---:|
| Original / majority | 29.03 | 21.92 |
| Original / logreg | 36.26 | 52.47 |
| Original / linear_svm | 35.25 | 46.93 |
| Original / multinomial_nb | 32.75 | 41.48 |
| Original / random_forest | 38.76 | 66.38 |
| October full cascade | 43.62 | 77.16 |
| October no-answer ablation | 43.82 | 77.12 |
| Initial / confidence_rich_trees | 44.50 | 75.41 |
| Initial / frozen_baseline | 43.62 | 77.16 |
| Initial / outcome_hist | 44.90 | 87.24 |
| Initial / outcome_logreg | 45.04 | 77.97 |
| Initial / gain_trees | 44.36 | 74.38 |
| Initial / sampled_augmentation_logreg | 45.04 | 77.98 |
| Initial / rich_trees | 44.50 | 75.42 |
| Initial / exact_reward_lambda0.0 | 44.09 | 74.26 |
| Initial / rich_hist | 44.02 | 79.03 |
| Initial / confidence_cost_sensitive_logreg | 44.63 | 79.06 |
| Initial / sampled_consensus_logreg | 45.04 | 81.15 |
| Initial / confidence_rich_hist | 44.36 | 79.12 |
| Initial / exact_reward_lambda0.8 | 44.09 | 87.43 |
| Initial / grpo_lambda0.8 | 43.82 | 81.80 |
| Initial / rich_logreg | 44.90 | 81.94 |
| Initial / confidence_outcome_hist | 45.85 | 87.99 |
| Initial / grpo_lambda0.0 | 44.56 | 85.37 |
| Initial / exact_reward_lambda0.4 | 43.96 | 80.77 |
| Initial / cost_sensitive_logreg | 44.63 | 79.06 |
| Initial / grpo_lambda0.4 | 43.75 | 81.48 |
| Initial / confidence_outcome_logreg | 45.04 | 77.97 |
| Initial / exact_reward_lambda0.15 | 44.29 | 77.33 |
| Initial / grpo_lambda0.15 | 43.69 | 81.16 |
| Initial / confidence_gain_trees | 44.29 | 74.49 |
| Initial / sampled_augmentation_hist | 45.17 | 87.01 |
| Initial / sampled_consensus_hist | 45.78 | 91.18 |
| Initial / confidence_rich_logreg | 44.90 | 81.94 |
| Extended / base_exact_cheap95 | 43.89 | 66.53 |
| Extended / base_exact_max_accuracy | 46.66 | 114.28 |
| Extended / base_exact_ss_budget | 45.58 | 96.69 |
| Extended / base_grpo_cheap95 | 43.69 | 70.16 |
| Extended / base_grpo_max_accuracy | 46.46 | 115.46 |
| Extended / base_grpo_ss_budget | 44.90 | 89.35 |
| Extended / confidence_exact_cheap95 | 43.89 | 67.81 |
| Extended / confidence_exact_max_accuracy | 46.79 | 108.59 |
| Extended / confidence_exact_ss_budget | 45.58 | 95.80 |
| Extended / confidence_grpo_cheap95 | 43.35 | 72.93 |
| Extended / confidence_grpo_max_accuracy | 46.46 | 115.70 |
| Extended / confidence_grpo_ss_budget | 45.04 | 90.62 |
| Extended / aug5_exact_cheap95 | 43.75 | 65.76 |
| Extended / aug5_exact_max_accuracy | 46.39 | 115.54 |
| Extended / aug5_exact_ss_budget | 45.51 | 95.36 |
| Extended / aug5_grpo_cheap95 | 43.89 | 70.49 |
| Extended / aug5_grpo_max_accuracy | 46.79 | 116.62 |
| Extended / aug5_grpo_ss_budget | 44.97 | 88.06 |
| Extended / aug9_exact_cheap95 | 43.89 | 66.09 |
| Extended / aug9_exact_max_accuracy | 46.39 | 113.47 |
| Extended / aug9_exact_ss_budget | 45.64 | 96.96 |
| Extended / aug9_grpo_cheap95 | 43.89 | 70.43 |
| Extended / aug9_grpo_max_accuracy | 46.86 | 116.36 |
| Extended / aug9_grpo_ss_budget | 45.17 | 98.37 |
| Architecture / direct4_max_accuracy | 47.20 | 100.00 |
| Architecture / direct4_ss_budget | 47.20 | 100.00 |
| Architecture / direct4_cheap95 | 46.46 | 91.91 |
| Architecture / pre_ret_qa_max_accuracy | 47.20 | 100.00 |
| Architecture / pre_ret_qa_ss_budget | 47.20 | 100.00 |
| Architecture / pre_ret_qa_cheap95 | 45.24 | 91.41 |
| Architecture / fixed_S_qa_max_accuracy | 47.20 | 100.00 |
| Architecture / fixed_S_qa_ss_budget | 47.20 | 100.00 |
| Architecture / fixed_S_qa_cheap95 | 45.91 | 99.64 |
| Architecture / post_answer_chooser_max_accuracy | 47.54 | 100.56 |
| Architecture / post_answer_chooser_ss_budget | 47.20 | 100.00 |
| Architecture / post_answer_chooser_cheap95 | 47.20 | 100.00 |
| Combined / max_accuracy | 47.20 | 100.00 |
| Combined / ss_budget | 47.20 | 100.00 |
| Combined / cheap95 | 46.46 | 91.91 |
