# Original-style non-LLM classifiers: corrected EM

Three independent classifiers. All score-dependent labels use official normalized EM. Unresolved labels are skipped in fitting; all test questions are evaluated.
Costs use the common mean parameter-weighted token ruler for direct routes; CPU router overhead is excluded. No weak-first sunk calls are charged for direct routing.

| Classifier | Test exact match % | Cost % direct SS |
|---|---:|---:|
| majority | 29.03 | 21.92 |
| logreg | 36.26 | 52.47 |
| linear_svm | 35.25 | 46.93 |
| multinomial_nb | 32.75 | 41.48 |
| random_forest | 38.76 | 66.38 |
