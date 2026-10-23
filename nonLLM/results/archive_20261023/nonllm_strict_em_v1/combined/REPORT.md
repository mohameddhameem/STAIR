> Corrected run: both legacy accuracy columns represent official normalized EM. See root protocol.json.

# Combined pre-routing and answer selection

Inherited train-CV models. Operating points locked on validation before test replay. Exploratory follow-up on a previously examined test set.
Strong and weak QA costs are both charged on every initially selected SS route, irrespective of the retained answer.

| Objective | Validation configuration | Val accuracy % | Val cost % SS | Test accuracy % | Strict % | Test cost % SS | Delta SS pp | Paired interval pp |
|---|---|---:|---:|---:|---:|---:|---:|---|
| max_accuracy | direct_SS | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 | [0.0, 0.0] |
| ss_budget | direct_SS | 46.94 | 100.00 | 47.20 | 47.20 | 100.00 | +0.00 | [0.0, 0.0] |
| cheap95 | lambda_0.05_no_chooser | 46.58 | 92.64 | 46.46 | 46.46 | 91.91 | -0.74 | [-1.350438892640108, -0.1350438892640108] |
