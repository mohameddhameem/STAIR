> Corrected run: both legacy accuracy columns represent official normalized EM. See root protocol.json.

# Non-LLM routing architecture experiments

Grouped three-fold training CV selects model families. All operating points and global selections use validation only. Test is previously examined; results are descriptive.
Costs include every paid model call, using the fixed mean parameter-weighted token ruler; CPU router overhead is excluded. No LLM weights were trained.

| Policy | Val accuracy % | Val cost % SS | Test accuracy % | Strict % | Cost % SS | Delta SS pp |
|---|---:|---:|---:|---:|---:|---:|
| direct4_max_accuracy | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 |
| direct4_ss_budget | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 |
| direct4_cheap95 | 46.58 | 92.64 | 46.46 | 46.46 | 91.91 | -0.74 |
| pre_ret_qa_max_accuracy | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 |
| pre_ret_qa_ss_budget | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 |
| pre_ret_qa_cheap95 | 44.59 | 92.16 | 45.24 | 45.24 | 91.41 | -1.96 |
| fixed_S_qa_max_accuracy | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 |
| fixed_S_qa_ss_budget | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 |
| fixed_S_qa_cheap95 | 45.14 | 99.64 | 45.91 | 45.91 | 99.64 | -1.28 |
| post_answer_chooser_max_accuracy | 47.12 | 100.56 | 47.54 | 47.54 | 100.56 | +0.34 |
| post_answer_chooser_ss_budget | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 |
| post_answer_chooser_cheap95 | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 |

Global validation selections:
- max_accuracy: post_answer_chooser_max_accuracy
- ss_budget: direct4_ss_budget
- cheap95: pre_ret_qa_cheap95

Models selected by train CV: ridge100, trees, trees

Direct SS test accuracy: 47.20%.
Direct SS fallback executes no weak calls and remains an explicitly deployable candidate. Post-answer choice pays both QA calls even when retaining the weak answer.
