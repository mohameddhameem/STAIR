# Non-LLM routing architecture experiments

Grouped three-fold training CV selects model families. All operating points and global selections use validation only. Test is previously examined; results are descriptive.
Costs include every paid model call, using the fixed mean parameter-weighted token ruler; CPU router overhead is excluded. No LLM weights were trained.

| Policy | Val accuracy % | Val cost % SS | Test accuracy % | Strict % | Cost % SS | Delta SS pp |
|---|---:|---:|---:|---:|---:|---:|
| direct4_max_accuracy | 52.52 | 100.00 | 52.94 | 47.20 | 100.00 | +0.00 |
| direct4_ss_budget | 52.52 | 100.00 | 52.94 | 47.20 | 100.00 | +0.00 |
| direct4_cheap95 | 51.98 | 93.44 | 52.19 | 46.52 | 93.08 | -0.74 |
| pre_ret_qa_max_accuracy | 52.52 | 100.00 | 52.94 | 47.20 | 100.00 | +0.00 |
| pre_ret_qa_ss_budget | 52.52 | 100.00 | 52.94 | 47.20 | 100.00 | +0.00 |
| pre_ret_qa_cheap95 | 51.53 | 93.69 | 51.72 | 46.32 | 93.41 | -1.22 |
| fixed_S_qa_max_accuracy | 52.52 | 100.00 | 52.94 | 47.20 | 100.00 | +0.00 |
| fixed_S_qa_ss_budget | 52.52 | 100.00 | 52.94 | 47.20 | 100.00 | +0.00 |
| fixed_S_qa_cheap95 | 50.18 | 99.59 | 51.99 | 46.46 | 99.65 | -0.95 |
| post_answer_chooser_max_accuracy | 52.88 | 100.56 | 53.00 | 47.33 | 100.56 | +0.07 |
| post_answer_chooser_ss_budget | 52.52 | 100.00 | 52.94 | 47.20 | 100.00 | +0.00 |
| post_answer_chooser_cheap95 | 52.52 | 100.00 | 52.94 | 47.20 | 100.00 | +0.00 |

Global validation selections:
- max_accuracy: post_answer_chooser_max_accuracy
- ss_budget: direct4_ss_budget
- cheap95: direct4_cheap95

Models selected by train CV: ridge100, ridge100, trees

Direct SS test accuracy: 52.94%.
Direct SS fallback executes no weak calls and remains an explicitly deployable candidate. Post-answer choice pays both QA calls even when retaining the weak answer.
