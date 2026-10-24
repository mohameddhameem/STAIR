# Pairwise routing, nested cross-fitting, calibration and weak-first cascades

New core5125/calibration1280 partition inside old6405; validation500 and previously inspected test500 unchanged. Three-seed tree ensembles; vocabulary-free features. Frozen four-model GGUF cache. Not common0.3 golden-set reproduction.

Outer5/inner3 strict grouped cross-fitting supplies base and stacked OOF predictions. OOF family selection is separate from validation family selection. Calibration labels are used only after all proposers freeze. CRC-style thresholds are exploratory because these samples were used by previous experiments; no certified or per-test accuracy guarantee.

Cascade costs include weak retrieval plus optional strong retrieval, weak QA on the selected branch plus optional strong QA. Direct SS remains the denominator. Vectorized replay does not measure online router latency.

| Policy | Test lenient % | Cost % direct SS | Execution |
|---|---:|---:|---|
| next-control-qwen7-llama3 | 54.00 | 9.62 | direct |
| next-control-qwen7-llama70 | 62.20 | 12.60 | direct |
| next-control-qwen72-llama3 | 65.80 | 97.26 | direct |
| next-control-qwen72-llama70 | 71.80 | 100.00 | direct |
| next-cascade-always-escalate | 71.80 | 109.61 | cascade |
| next-outcome_joint-max_accuracy-learned_only | 71.20 | 76.22 | direct |
| next-outcome_joint-max_accuracy-with_fixed_fallback | 71.20 | 76.22 | direct |
| next-outcome_joint-ss_budget-learned_only | 71.20 | 76.22 | direct |
| next-outcome_joint-ss_budget-with_fixed_fallback | 71.20 | 76.22 | direct |
| next-outcome_joint-cheap95-learned_only | 71.20 | 76.22 | direct |
| next-outcome_joint-cheap95-with_fixed_fallback | 71.20 | 76.22 | direct |
| next-outcome_joint-match_ss_accuracy-learned_only | 71.20 | 76.22 | direct |
| next-outcome_joint-match_ss_accuracy-with_fixed_fallback | 71.20 | 76.22 | direct |
| next-outcome_staged-max_accuracy-learned_only | 71.40 | 76.22 | direct |
| next-outcome_staged-max_accuracy-with_fixed_fallback | 71.40 | 76.22 | direct |
| next-outcome_staged-ss_budget-learned_only | 71.40 | 76.22 | direct |
| next-outcome_staged-ss_budget-with_fixed_fallback | 71.40 | 76.22 | direct |
| next-outcome_staged-cheap95-learned_only | 71.40 | 76.22 | direct |
| next-outcome_staged-cheap95-with_fixed_fallback | 71.40 | 76.22 | direct |
| next-outcome_staged-match_ss_accuracy-learned_only | 71.40 | 76.22 | direct |
| next-outcome_staged-match_ss_accuracy-with_fixed_fallback | 71.40 | 76.22 | direct |
| next-gain_joint-max_accuracy-learned_only | 71.20 | 95.68 | direct |
| next-gain_joint-max_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-gain_joint-ss_budget-learned_only | 71.20 | 95.68 | direct |
| next-gain_joint-ss_budget-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-gain_joint-cheap95-learned_only | 70.20 | 72.27 | direct |
| next-gain_joint-cheap95-with_fixed_fallback | 70.20 | 72.27 | direct |
| next-gain_joint-match_ss_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-gain_staged-max_accuracy-learned_only | 71.20 | 95.66 | direct |
| next-gain_staged-max_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-gain_staged-ss_budget-learned_only | 71.20 | 95.66 | direct |
| next-gain_staged-ss_budget-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-gain_staged-cheap95-learned_only | 70.40 | 72.23 | direct |
| next-gain_staged-cheap95-with_fixed_fallback | 70.40 | 72.23 | direct |
| next-gain_staged-match_ss_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-pair_joint-max_accuracy-learned_only | 71.80 | 100.00 | direct |
| next-pair_joint-max_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-pair_joint-ss_budget-learned_only | 71.80 | 100.00 | direct |
| next-pair_joint-ss_budget-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-pair_joint-cheap95-learned_only | 70.60 | 81.39 | direct |
| next-pair_joint-cheap95-with_fixed_fallback | 70.60 | 81.39 | direct |
| next-pair_joint-match_ss_accuracy-learned_only | 71.80 | 100.00 | direct |
| next-pair_joint-match_ss_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-pair_staged-max_accuracy-learned_only | 71.80 | 100.00 | direct |
| next-pair_staged-max_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-pair_staged-ss_budget-learned_only | 71.80 | 100.00 | direct |
| next-pair_staged-ss_budget-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-pair_staged-cheap95-learned_only | 70.60 | 81.39 | direct |
| next-pair_staged-cheap95-with_fixed_fallback | 70.60 | 81.39 | direct |
| next-pair_staged-match_ss_accuracy-learned_only | 71.80 | 100.00 | direct |
| next-pair_staged-match_ss_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-stack_joint-max_accuracy-learned_only | 72.00 | 90.70 | direct |
| next-stack_joint-max_accuracy-with_fixed_fallback | 72.00 | 90.70 | direct |
| next-stack_joint-ss_budget-learned_only | 72.00 | 90.70 | direct |
| next-stack_joint-ss_budget-with_fixed_fallback | 72.00 | 90.70 | direct |
| next-stack_joint-cheap95-learned_only | 71.00 | 74.09 | direct |
| next-stack_joint-cheap95-with_fixed_fallback | 71.00 | 74.09 | direct |
| next-stack_joint-match_ss_accuracy-learned_only | 72.00 | 90.70 | direct |
| next-stack_joint-match_ss_accuracy-with_fixed_fallback | 72.00 | 90.70 | direct |
| next-stack_staged-max_accuracy-learned_only | 72.00 | 90.72 | direct |
| next-stack_staged-max_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-stack_staged-ss_budget-learned_only | 72.00 | 90.72 | direct |
| next-stack_staged-ss_budget-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-stack_staged-cheap95-learned_only | 71.00 | 74.11 | direct |
| next-stack_staged-cheap95-with_fixed_fallback | 71.00 | 74.11 | direct |
| next-stack_staged-match_ss_accuracy-with_fixed_fallback | 71.80 | 100.00 | direct |
| next-cascade-max_accuracy-learned_only | 71.80 | 98.21 | cascade |
| next-cascade-max_accuracy-with_fixed_fallback | 71.80 | 98.21 | cascade |
| next-cascade-ss_budget-learned_only | 71.80 | 98.21 | cascade |
| next-cascade-ss_budget-with_fixed_fallback | 71.80 | 98.21 | cascade |
| next-cascade-cheap95-learned_only | 71.40 | 74.13 | cascade |
| next-cascade-cheap95-with_fixed_fallback | 71.40 | 74.13 | cascade |
| next-cascade-match_ss_accuracy-learned_only | 71.80 | 95.17 | cascade |
| next-cascade-match_ss_accuracy-with_fixed_fallback | 71.80 | 95.17 | cascade |
| next-global_val-max_accuracy-learned_only | 71.80 | 98.21 | cascade |
| next-global_val-max_accuracy-with_fixed_fallback | 71.80 | 98.21 | cascade |
| next-global_val-ss_budget-learned_only | 71.80 | 98.21 | cascade |
| next-global_val-ss_budget-with_fixed_fallback | 71.80 | 98.21 | cascade |
| next-global_val-cheap95-learned_only | 70.40 | 72.23 | direct |
| next-global_val-cheap95-with_fixed_fallback | 70.40 | 72.23 | direct |
| next-global_val-match_ss_accuracy-learned_only | 71.20 | 76.22 | direct |
| next-global_val-match_ss_accuracy-with_fixed_fallback | 71.20 | 76.22 | direct |
| next-global_oof-max_accuracy-learned_only | 71.80 | 95.17 | cascade |
| next-global_oof-max_accuracy-with_fixed_fallback | 71.80 | 95.17 | cascade |
| next-global_oof-ss_budget-learned_only | 71.80 | 95.17 | cascade |
| next-global_oof-ss_budget-with_fixed_fallback | 71.80 | 95.17 | cascade |
| next-global_oof-cheap95-learned_only | 70.40 | 72.23 | direct |
| next-global_oof-cheap95-with_fixed_fallback | 70.40 | 72.23 | direct |
| next-global_oof-match_ss_accuracy-learned_only | 71.80 | 89.77 | direct |
| next-global_oof-match_ss_accuracy-with_fixed_fallback | 71.80 | 89.77 | direct |
| next-risk-pair_joint-max_accuracy-alpha0.005 | 71.80 | 100.00 | direct |
| next-risk-pair_joint-max_accuracy-alpha0.01 | 71.80 | 100.00 | direct |
| next-risk-pair_joint-max_accuracy-alpha0.02 | 71.80 | 100.00 | direct |
| next-risk-pair_joint-ss_budget-alpha0.005 | 71.80 | 100.00 | direct |
| next-risk-pair_joint-ss_budget-alpha0.01 | 71.80 | 100.00 | direct |
| next-risk-pair_joint-ss_budget-alpha0.02 | 71.80 | 100.00 | direct |
| next-risk-pair_joint-cheap95-alpha0.005 | 71.00 | 86.81 | direct |
| next-risk-pair_joint-cheap95-alpha0.01 | 70.60 | 82.87 | direct |
| next-risk-pair_joint-cheap95-alpha0.02 | 70.60 | 81.39 | direct |
| next-risk-pair_joint-match_ss_accuracy-alpha0.005 | 71.80 | 100.00 | direct |
| next-risk-pair_joint-match_ss_accuracy-alpha0.01 | 71.80 | 100.00 | direct |
| next-risk-pair_joint-match_ss_accuracy-alpha0.02 | 71.80 | 100.00 | direct |
| next-risk-stack_joint-max_accuracy-alpha0.005 | 72.00 | 91.36 | direct |
| next-risk-stack_joint-max_accuracy-alpha0.01 | 72.00 | 90.70 | direct |
| next-risk-stack_joint-max_accuracy-alpha0.02 | 72.00 | 90.70 | direct |
| next-risk-stack_joint-ss_budget-alpha0.005 | 72.00 | 91.36 | direct |
| next-risk-stack_joint-ss_budget-alpha0.01 | 72.00 | 90.70 | direct |
| next-risk-stack_joint-ss_budget-alpha0.02 | 72.00 | 90.70 | direct |
| next-risk-stack_joint-cheap95-alpha0.005 | 71.00 | 86.81 | direct |
| next-risk-stack_joint-cheap95-alpha0.01 | 70.80 | 84.17 | direct |
| next-risk-stack_joint-cheap95-alpha0.02 | 71.00 | 78.06 | direct |
| next-risk-stack_joint-match_ss_accuracy-alpha0.005 | 72.00 | 91.36 | direct |
| next-risk-stack_joint-match_ss_accuracy-alpha0.01 | 72.00 | 90.70 | direct |
| next-risk-stack_joint-match_ss_accuracy-alpha0.02 | 72.00 | 90.70 | direct |

## Study summary

| Study / selection | Test lenient | Cost % direct SS |
|---|---:|---:|
| Four-path pairwise, max accuracy | 71.80% | 100.00 |
| OOF stacking, max accuracy | 72.00% | 90.70 |
| Nested OOF global match-SS selection | 71.80% | 89.77 |
| Stack joint, alpha0.5% gate | 72.00% | 91.36 |
| Weak-first cascade, max accuracy | 71.80% | 98.21 |
| Weak-first cascade,95% validation target | 71.40% | 74.13 |
| Cascade always escalates, control | 71.80% | 109.61 |

All reported hyperparameters were selected on OOF/validation, and risk thresholds on separate calibration, before test construction. The stack joint result is360/500 versus359/500 SS: one rescued question, not evidence of a stable accuracy gain. The OOF match-SS selector chose outcome_joint, maintaining359/500 at89.77% cost. Validation global max accuracy selected cascade at71.8%/98.21%; OOF global max accuracy selected a different cascade weight at71.8%/95.17%. These are distinct selection objectives, not the best test row.

The paid-call audit verifies every test record, including weak QA on the selected retrieval branch. Cascade always-escalate has the same final answer as SS but109.61% cost. No weak-call cost is hidden.

Risk alpha is an unconditional additional-harm target relative to SS, not overall answer error or error conditional on switching. Some test policies have observed extra-harm rates above their nominal targets: all calibration criteria passed, but expected-risk control does not bound every finite test sample, and historical reuse means this is not certified CRC. See risk_summary.csv for every proposer/alpha, including unhelpful all-SS policies.

New models use5125 fitting questions plus1280 reserved calibration questions within the earlier6405 training pool. Thus comparisons with earlier6405-fit routers are not equal fitting-label budgets. Trees use raw deterministic statistics; no global TFIDF/scaler or old trained router is reused. Outer stacking predictions use genuine inner3-fold meta-training, and the final stack uses the same3-fold procedure.

[Accuracy/cost figure](accuracy_cost.png) · [Method table](method_summary.csv) · [Calibration and observed risk](risk_summary.csv) · [Fold isolation audit](fold_audits.json) · [Verification](verification.json). Latency remains unmeasured in this offline vectorized replay. All test comparisons remain exploratory.
