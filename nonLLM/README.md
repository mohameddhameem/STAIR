# nonLLM

CPU routing baselines and GGUF inference setup for STAIR.
Here, non-LLM refers to the routing decision model; the upstream retrieval and
answering models in the recorded pipeline may still be LLMs.

## Local data

The archive is extracted outside this Git repository at
`/ssd1/zmcheng/cs707/extracted/`:

- `Dataset/router_data_lenient.jsonl.gz`: measured outcomes and real labels.
- `Dataset/router_data_v2_tau0.50.jsonl.gz`: alternate training labels.
- `Dataset/splits.json`: frozen train/validation/test indices (4814/1110/1481).
- `Python Codes/`: reference LLM-router scripts.

Read JSONL with `gzip.open(path, "rt")`, or decompress to a local data directory.
The original trainer expects uncompressed JSONL.

## Evaluation compatibility

Use the frozen split: train on train, select settings on validation, evaluate the
finished experiment on test. The measured outcome order is `[WW, WS, SW, SS]`,
with retrieval first and answering second. Report both lenient and strict
accuracy, routing picks, and pipeline cost (2.0/5.7/5.7/9.4 units).

Gold answers, supporting titles, measured outcomes, and judge scores are
supervision or evaluation metadata; do not use them as deployment-time features.
Features comparing weak and strong retrieval results require both retrieval
runs; account for this cost when claiming end-to-end savings.

## Synchronization

Local checkout: `/ssd1/zmcheng/cs707/STAIR`.
Remote: `https://github.com/mohameddhameem/STAIR.git`.
Working branch: `nonLLM`. Remote publishing requires an authenticated account
with repository write access, or a fork and pull request.

Keep source code and documentation here. Put local datasets, checkpoints, and
run outputs in the ignored directories listed in `.gitignore`.

## Routing baselines

`baselines.py` implements majority, TF-IDF logistic regression, TF-IDF linear SVM,
TF-IDF multinomial naive Bayes, and random forest over 20 numerical text features.
Three separate classifiers are trained per family: retrieval, QA after weak
retrieval, and QA after strong retrieval. Actual outcomes and gold annotations
are never features. Classifier scoring excludes unresolved labels; composed
routing evaluates all 1,481 test instances by selecting the corresponding grid cell.

The initial experiment uses `router_data_lenient.jsonl.gz` measured labels.
Tune only on validation balanced accuracy; TF-IDF vocabulary is fit on labeled
training rows only. Logistic regression and SVM use balanced class weights;
random forest also uses balanced weights. NB retains its learned class prior.
This reports stage accuracy, balanced accuracy, macro F1, confusion matrices,
composed lenient/strict QA accuracy, choices and pipeline cost.

Run with the existing environment (no GPU is used):

```bash
cd /ssd1/zmcheng/cs707/STAIR/nonLLM
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /ssd1/zmcheng/cs701/.env/bin/python baselines.py \
  --data /ssd1/zmcheng/cs707/extracted/Dataset/router_data_lenient.jsonl.gz \
  --splits /ssd1/zmcheng/cs707/extracted/Dataset/splits.json \
  --out /ssd1/zmcheng/cs707/runs/baselines_real_v2
```

Use a fresh output directory. Run `report_baselines.py <run-directory>` to render
the report and plot; the report does not retrain or select models on test.
`python -m unittest test_baselines.py` checks branch selection and feature leakage
using synthetic data. Dependencies for this experiment are in `requirements.txt`.

Completed experiment: [full report](results/measured_v1/REPORT.md).
Selected classifiers and vectorizers are saved locally in
`/ssd1/zmcheng/cs707/runs/baselines_real_v1/{ret,qa_W,qa_S}.joblib`.
Local results also include every test prediction in `test_predictions.csv`.
The stored-grid cost excludes CPU routing overhead; the newly downloaded GGUF
models are not run by these routing experiments.
