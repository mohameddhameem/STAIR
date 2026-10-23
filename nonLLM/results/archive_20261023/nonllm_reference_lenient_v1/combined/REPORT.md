# Combined pre-routing and answer selection

Inherited train-CV models. Operating points locked on validation before test replay. Exploratory follow-up on a previously examined test set.
Strong and weak QA costs are both charged on every initially selected SS route, irrespective of the retained answer.

| Objective | Validation configuration | Val accuracy % | Val cost % SS | Test accuracy % | Strict % | Test cost % SS | Delta SS pp | Paired interval pp |
|---|---|---:|---:|---:|---:|---:|---:|---|
| max_accuracy | lambda_0.0_threshold_0.0 | 52.52 | 99.42 | 52.94 | 47.20 | 98.94 | +0.00 | [-0.337609723160027, 0.337609723160027] |
| ss_budget | lambda_0.0_threshold_0.0 | 52.52 | 99.42 | 52.94 | 47.20 | 98.94 | +0.00 | [-0.337609723160027, 0.337609723160027] |
| cheap95 | lambda_0.05_no_chooser | 51.98 | 93.44 | 52.19 | 46.52 | 93.08 | -0.74 | [-1.350438892640108, -0.2025658338960162] |
