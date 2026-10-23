# Additional corrected-cache non-LLM methods

Completed experiment: `runs/nonllm_common_ws_alternatives_v1`. Reuses the corrected executor grid and the training-only TF-IDF/scaling from `nonllm_common_ws_grpo_v1`; no new executor generation or executor fine-tuning. The cohort remains common0.2 train_extra6405/val500/test500, author lenient and measured token-derived units, not the common0.3 SLR golden set.

Eight families:

1. `outcome_linear`: predict all4 route success probabilities from question+pool, then2 QA success probabilities from the paid selected evidence. BCE includes all outcomes, unlike the earlier cheapest-correct classification labels.
2. `outcome_residual`: the same target with a64-unit ReLU residual head.
3. `outcome_trees`: numeric-only multi-output ExtraTrees success predictors.
4. `gain_linear`: predict correctness(SS) minus correctness(weak retrieval,strong QA), including ties. QA is fixed strong; the retrieval threshold is validation-selected.
5. `gain_trees`: the same expected upgrade-gain target with ExtraTrees.
6. `group_centered`: group8 PPO with sampled group-mean subtraction but no standard-deviation division.
7. `expected_baseline`: group8 PPO with the exact current-policy expected cached reward as the baseline, without standard-deviation division.
8. `exact_augmented`: exact expected-reward optimization with the appended lexical features.

The two unscaled policy-gradient variants and exact control reuse the previous per-seed SFT warm weights; coefficients for new features initially zero. Policy reward remains lenient minus lambda times selected direct-call cost divided by training-mean SS. Training lambda0/.05/.15/.4, clip0.2, two PPO updates, rollout KL0.01 and entropy0.002. Changing both features and the estimator means these comparisons do not isolate standard-deviation normalization as the sole causal factor.

Additional46 features describe question type, entity-like capitalization, candidate coverage/title overlap, top-score gaps, length/coverage distributions and top-document overlap. Before retrieval these use only the supplied pool; after retrieval they use only selected evidence. Gold answers, labels, weak/strong answers and unexecuted retrieval outputs are not input features. Extra scalers fit train only. No embedding LLM is used for routing.

54 neural streams:18 supervised (three architectures/targets x two weight decays x three seeds),24 unscaled policy-gradient streams,12 exact-reward controls. Maximum300 epochs, validation BCE/MSE/task-loss early stopping.24 tree configurations (two families x four tree settings x three seeds),192 trees per estimator. All three seeds are averaged within each configuration.

All stage-penalty grids, gain thresholds and method/checkpoint settings are validation-only. Objectives: maximum accuracy, SS budget, cheapest retaining95% of validation SS accuracy, and cheapest matching validation SS accuracy. Learned-only and fixed-fallback variants are separate. Global family selection is also locked on validation before test. Validation constraints need not hold on test. The test has previously been inspected; comparisons and paired intervals are exploratory, not fresh confirmation.

Per-family maximum-accuracy test results include outcome_trees72.4%/96.77% SS cost, expected_baseline72.0%/96.54%, and exact_augmented71.8%/89.25%. These are respectively three, one and zero additional correct answers versus359/500 SS. The predeclared global validation maximum-accuracy winner is outcome_linear, which scores71.2%/85.61% on test. Report both views; do not substitute the best test family for the global validation winner.

Full results: [report](../../results/menu_ws_alternatives_v1/REPORT.md). Offline vectorized test replay leaves ms_router empty because full online latency was not measured; CPU router work is excluded from the executor FLOP proxy. No runtime speedup claim is made from cost units alone.

From the workspace containing STAIR and the local caches:

```bash
python STAIR/nonLLM/experiments/menu_ws_gguf/test_alternatives.py
python STAIR/nonLLM/experiments/menu_ws_gguf/alternatives.py --out runs/nonllm_common_ws_alternatives_v1
python STAIR/nonLLM/experiments/menu_ws_gguf/summarize_alternatives.py
```
