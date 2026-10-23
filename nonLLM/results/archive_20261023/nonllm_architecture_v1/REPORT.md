# Non-LLM routing architecture experiments

Grouped three-fold training CV selects model families. All operating points and global selections use validation only. Test is previously examined; results are descriptive.
Costs include every paid model call, using the fixed mean parameter-weighted token ruler; CPU router overhead is excluded. No LLM weights were trained.

| Policy | Val accuracy % | Val cost % SS | Test accuracy % | Strict % | Cost % SS | Delta SS pp |
|---|---:|---:|---:|---:|---:|---:|
| direct4_max_accuracy | 59.73 | 98.73 | 58.95 | 47.06 | 98.14 | -0.14 |
| direct4_ss_budget | 59.73 | 98.73 | 58.95 | 47.06 | 98.14 | -0.14 |
| direct4_cheap95 | 59.55 | 94.18 | 58.74 | 46.79 | 93.28 | -0.34 |
| pre_ret_qa_max_accuracy | 59.73 | 100.00 | 59.08 | 47.20 | 100.00 | +0.00 |
| pre_ret_qa_ss_budget | 59.73 | 100.00 | 59.08 | 47.20 | 100.00 | +0.00 |
| pre_ret_qa_cheap95 | 57.12 | 93.37 | 56.72 | 45.04 | 92.66 | -2.36 |
| fixed_S_qa_max_accuracy | 59.73 | 100.00 | 59.08 | 47.20 | 100.00 | +0.00 |
| fixed_S_qa_ss_budget | 59.73 | 100.00 | 59.08 | 47.20 | 100.00 | +0.00 |
| fixed_S_qa_cheap95 | 57.75 | 99.46 | 57.80 | 45.91 | 99.48 | -1.28 |
| post_answer_chooser_max_accuracy | 60.27 | 100.56 | 59.55 | 47.54 | 100.56 | +0.47 |
| post_answer_chooser_ss_budget | 59.73 | 100.00 | 59.08 | 47.20 | 100.00 | +0.00 |
| post_answer_chooser_cheap95 | 59.73 | 100.00 | 59.08 | 47.20 | 100.00 | +0.00 |

Global validation selections:
- max_accuracy: post_answer_chooser_max_accuracy
- ss_budget: direct4_ss_budget
- cheap95: pre_ret_qa_cheap95

Models selected by train CV: ridge100, ridge100, trees

Direct SS test accuracy: 59.08%.
Direct SS fallback executes no weak calls and remains an explicitly deployable candidate. Post-answer choice pays both QA calls even when retaining the weak answer.
