# Classical non-LLM routers on MedAgentBench

Run: `../runs/nonllm_medagentbench_classical_v1` from the repository.
Source: `../runs/nonllm_medagentbench_replay_v1`, pinned Jonathan campaign
`3ac054d1d72994cab5e8927c6cdba1449371aba4`. This is the same patient-disjoint
90/60/150 Training1 andTraining3 replay and same reward/cost/readout as the prior
non-LLM GRPO study; classifiers actually run on test views, executor outcomes are
replayed. No new weak/strong execution or medical environment is involved.

Seven model families: Random Forest, Extra Trees, Decision Tree, Linear SVM,
RBF SVM, Logistic Regression and Multinomial Naive Bayes. Seeds17 (primary),0,1,2
repeats; two settings;200 hyperparameter candidates;56 selected classifiers.

These models use **supervised cost-sensitive action imitation, not GRPO**.
On training90, compute the source reward for both actions, set the label to
argmax reward (W on ties), and weight the sample by abs(RW−RS), normalized by
its train mean. Misclassification regret is precisely the reward gap for two
actions, so costly wrong decisions count more. The supervision derives from
the training outcomes; test outcomes never enter input or model selection.
Probabilities describe the optimal route label, not weak-answer correctness.

Features: the same train-fitted unigram/bigram TF-IDF and ten deterministic
view statistics used previously; NB uses TF-IDF alone for nonnegative inputs.
No task ID, patient ID field, cost, risk, correctness or strong-output field is
appended. Patient references present in the shared view are retained as the LLM
sees them. No pretrained encoder is called.

SVM calibration uses patient-grouped3-fold OOF margins, with the *entire feature
pipeline refitted within each fold*. Fold weights are normalized using only
the fold's train samples. Train-only logistic calibration is fitted on those
margins, then the base SVM is refitted on90. No SVC internal probability=True
or calibration on the outer60-task validation split is used.

Per family/seed, choose hyperparameters by validation reward at theta.50, then
lower cost. Lock a second validation-reward-selected threshold over0:.01:1.
Across-family choices are also saved from validation before test. Fixed.50/.20
and both risk-gate readouts reuse a single test probability vector per selected
classifier. All selections, models and hashes are locked before test inference.
Report all methods and seeds, not the best-test family or seed.

Training1 decides after the full weak attempt. Training3 sees round2, delivers
at full weak cost or escalates at cw2+cs. Reported cost is divided by full
sum(cw+cs) on the same150 tasks for both settings; it excludes classifier CPU
work and frozen risk-score generation consistently with the source board.
Same tau>=.8 forced-escalation gate is required for risk-matched comparisons.
Cheaper ungated policies may deliver high-risk weak failures.

Commands from the repository root:

```bash
../.cascade-env/bin/python nonLLM/experiments/medagentbench/test_classical.py
../.cascade-env/bin/python nonLLM/experiments/medagentbench/audit_classical.py --source ../runs/nonllm_medagentbench_replay_v1 --out ../runs/nonllm_medagentbench_classical_v1
```

For a fresh run: `classical.py --phase prepare`, `--phase train` for every
setting/family/seed, then `--phase lock`, finally `--phase evaluate`. Every phase
requires `--source <source-run> --out <new-run>`. Evaluation refuses a second
test inference; use the audit script for stored-score checks. Prior public test
inspection and method multiplicity make these exploratory replay results.
Patient-cluster bootstrap intervals, not task-independent intervals, accompany
the paired comparisons. No raw patient views or model weights are published.
