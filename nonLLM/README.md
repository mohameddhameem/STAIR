# nonLLM

Workspace for non-LLM routing methods in STAIR. Implementation is pending.
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
