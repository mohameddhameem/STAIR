# Classical non-LLM MedAgentBench routers

Same Jonathan Training1/3 replay, view fields, reward/cost equations, and patient-disjoint90/60/150 split as the previous GRPO experiment. These methods use supervised reward-optimal action imitation, not GRPO: y=argmax(RW,RS), sample weight=absolute reward gap normalized by train mean. Actual classifiers run on test views once; executor outcomes are replayed. No new medical episodes or task-model training.

## Model selection and calibration

TF-IDF+view statistics, fitted on90 training tasks only; NB text-only. SVMs use patient-grouped3-fold OOF margins with fold-local feature fitting, followed by train-only logistic calibration. Hyperparameters are selected by validation mean reward at theta.50, cost tie-break. A second threshold is selected on validation reward; every choice/model hash is locked before test. Seed17 is primary;0/1/2 repeats are not an ensemble. These action probabilities are not calibrated probabilities of weak-answer correctness.

## Primary seed17, fixed theta0.50, no risk gate

| Router | T1 accuracy / cost | T3 accuracy / cost |
|---|---:|---:|
| Jonathan LLM |61.33% /54.11% |63.33% /81.47% |
| random_forest | 61.33% / 54.11% | 61.33% / 48.38% |
| extra_trees | 61.33% / 54.70% | 63.33% / 58.45% |
| decision_tree | 61.33% / 54.11% | 61.33% / 48.38% |
| linear_svm | 61.33% / 54.70% | 61.33% / 48.38% |
| rbf_svm | 61.33% / 54.70% | 63.33% / 58.45% |
| logistic_regression | 61.33% / 55.53% | 63.33% / 74.90% |
| multinomial_nb | 61.33% / 54.11% | 63.33% / 58.45% |

## Primary seed17, validation-selected threshold and common risk gate

| Router | T1 accuracy / cost | T3 accuracy / cost |
|---|---:|---:|
| Jonathan LLM, own fixed theta.50 + same gate |61.33% /80.99% |63.33% /81.47% |
| random_forest | 61.33% / 80.99% | 63.33% / 81.47% |
| extra_trees | 61.33% / 80.99% | 63.33% / 81.47% |
| decision_tree | 61.33% / 80.99% | 61.33% / 71.39% |
| linear_svm | 61.33% / 81.58% | 63.33% / 77.68% |
| rbf_svm | 61.33% / 81.58% | 63.33% / 77.68% |
| logistic_regression | 61.33% / 81.58% | 63.33% / 77.68% |
| multinomial_nb | 61.33% / 80.99% | 63.33% / 81.47% |

## Validation-selected family (not best-test family)

| Setting | Point | Family chosen on validation | Test accuracy | Cost % |
|---|---|---|---:|---:|
|T1|OP_A_fixed_050|decision_tree|61.33%|54.11%|
|T1|val_reward_selected|decision_tree|61.33%|54.11%|
|T3|OP_A_fixed_050|extra_trees|63.33%|58.45%|
|T3|val_reward_selected|linear_svm|63.33%|54.66%|

## Limits

Reported costs are measured replay FLOP units normalized by sum(full weak cost+strong cost) on the same150 tasks, including sunk weak work. They exclude router compute and the frozen risk scorer, consistently with the LLM board; CPU feature+batch prediction time is separate. These are not HotpotQA Table5 prices or direct-strong-only costs.

Ungated savings may deliver high-risk weak failures. Risk-gate results use identical frozen tau>=.8 forced escalation. Dataset small and task-type homogeneous; test already inspected in prior research, so these are exploratory paired-replay findings, not a new sealed test. Do not select method or seed based on test. Classifier probabilities model reward-optimal action, not actual weak success. Complete per-seed/point/gate rows are in results.csv and means/ranges in seed_repeats.json.

## Independent audit and repeat sensitivity

All selected feature vocabularies/scalers match the previous train-only pipeline. SVM calibration fold patient separation and90-row OOF coverage were independently reconstructed from the official dataset. No test model was run again. Paired patient-cluster intervals use2,000 resamples and are in paired_patient_comparisons.json.

| Router | T3 validation-selected threshold, same risk gate: accuracy mean (0/1/2 range) | Mean cost % |
|---|---:|---:|
|random_forest|63.33% (63.33–63.33)|81.47%|
|extra_trees|63.33% (63.33–63.33)|81.47%|
|decision_tree|61.33% (61.33–61.33)|71.39%|
|linear_svm|63.33% (63.33–63.33)|77.68%|
|rbf_svm|63.33% (63.33–63.33)|77.68%|
|logistic_regression|63.33% (63.33–63.33)|77.68%|
|multinomial_nb|63.33% (63.33–63.33)|81.47%|

## Why the paired cost changes

For seed17 Training3, validation-selected Linear SVM plus the same high-risk gate differs from the LLM on15 task7 instances. Both cached weak and strong executions fail on all15; the classifier delivers the weak result instead of paying for an ineffective strong escalation. There is zero change in correctness on those tasks, and no claim of improved clinical outcome. These instances fall outside the tau>=.8 gate. This explains the3.79 percentage-point cost saving while maintaining63.33% aggregate accuracy; it does not solve the underlying failures.

Task types are highly homogeneous in this small benchmark. These independent seed repeats are stable on the fixed replay but are not independent executor reruns or proof of generalization to new task types. Public-test reuse and multiple methods make the result exploratory.
