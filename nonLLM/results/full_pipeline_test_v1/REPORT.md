# Full actual-model test benchmark

All 1,481 frozen internal test questions. Actual 1.5B Base / 7B Base Q4_K_M retrieval and answering.
2,962 retrieval calls and 5,924 QA calls are complete. There are no skipped test questions.
Retrieval sees every candidate passage and selects up to four original passages with constrained IDs.
Two synthetic retrieval demonstrations are fixed in the prompt; no gold answers or supporting annotations enter prompts.
Every QA branch uses NEW evidence produced in this run. Existing classifiers remain frozen.

| Method | Answer EM % | Answer F1 % | Diagnostic containment % | WW / WS / SW / SS |
|---|---:|---:|---:|---|
| majority | 6.21 | 11.28 | 12.36 | 1481 / 0 / 0 / 0 |
| logreg | 11.55 | 17.91 | 19.38 | 527 / 310 / 494 / 150 |
| linear_svm | 10.33 | 16.49 | 17.69 | 717 / 181 / 452 / 131 |
| multinomial_nb | 9.18 | 15.03 | 16.07 | 999 / 27 / 455 / 0 |
| random_forest | 11.21 | 18.00 | 18.70 | 569 / 280 / 306 / 326 |
| always_WW | 6.21 | 11.28 | 12.36 | 1481 / 0 / 0 / 0 |
| always_WS | 12.09 | 18.91 | 18.30 | 0 / 1481 / 0 / 0 |
| always_SW | 12.69 | 20.87 | 21.61 | 0 / 0 / 1481 / 0 |
| always_SS | 18.16 | 27.07 | 28.29 | 0 / 0 / 0 / 1481 |

Actual four-cell oracle answer EM: 25.73%.

EM and token F1 use the official HotpotQA answer-scoring functions. No supporting-fact or joint score is claimed.
Containment is only a diagnostic, not the official metric or a certified reproduction of the original lenient scorer.
Raw timings were collected with two concurrent requests and other GPU experiments. They are not isolated routed latency.
The full grid was generated once, then each frozen policy selected its actual retrieval/answering path.
This is a real-model grid evaluation, not nine separately timed online deployments.
Prompts, quantization, selected evidence, and generation settings differ from the old stored-grid experiment.
The full test result does not modify training labels, classifiers, or thresholds.

Configuration, hashes, scores and timings: [results.json](results.json).

Raw retrieval/QA outputs and resumable checkpoints are stored locally in `runs/full_pipeline_test_v1/calls.sqlite`.
Every method's selected path is stored locally in `decisions.jsonl`.
