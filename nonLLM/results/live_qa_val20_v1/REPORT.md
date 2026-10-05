# Live QA validation pilot

Actual llama.cpp CUDA inference, RTX A5000 GPU 0, Q4_K_M 1.5B Base and 7B Base.
20 questions sampled uniformly without replacement from the fixed internal validation split (seed 707).
The sample was selected before generation, without filtering by labels or outcomes. All 20 were evaluated.
Both models answered each question with the same cached strong retrieval evidence (`evidence_S`).
This tests the answering stage only; retrieval was not rerun. Servers were loaded sequentially and shut down after use.
Context 8192, one request at a time, greedy generation, up to 64 output tokens. No prompts were silently truncated.

## Scores

Answer EM and token F1 use the official HotpotQA evaluation functions. Supporting-fact and joint scores are not computed.
Whole-word containment is a diagnostic and is not the official answer metric or a certified reproduction of the project lenient scorer.

| Policy | Answer EM % | Answer F1 % | 1.5B / 7B answerer picks |
|---|---:|---:|---:|
| always_weak | 15.00 | 36.76 | 20 / 0 |
| always_strong | 35.00 | 50.71 | 0 / 20 |
| majority | 15.00 | 36.76 | 20 / 0 |
| logreg | 25.00 | 42.24 | 14 / 6 |
| linear_svm | 25.00 | 42.24 | 15 / 5 |
| multinomial_nb | 15.00 | 36.76 | 20 / 0 |
| random_forest | 35.00 | 47.82 | 6 / 14 |

The existing non-LLM QA-after-strong classifiers were used unchanged. Their decisions select between the actual generated answers above.
Both models were generated offline for every question to enable the comparison; this is not a live routed deployment cost measurement.
Base model variants, quantization and the new completion prompt can differ from the stored-grid experiment.
20 validation questions are a pilot, not a final test benchmark. Equality of EM on this sample does not establish equivalent model quality.

## Reproduce

Use the existing environment with scikit-learn/joblib installed and a fresh output directory:
```bash
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /ssd1/zmcheng/cs701/.env/bin/python \
  /ssd1/zmcheng/cs707/STAIR/nonLLM/deployment/live_qa_eval.py \
  --gpu 0 --n 20 --seed 707 \
  --out /ssd1/zmcheng/cs707/runs/live_qa_val20_v2
```

The script requires the previously saved `qa_S.joblib` baseline bundle and the downloaded official metric script.
Configuration and model hashes: [results.json](results.json). Every actual generation: [predictions.jsonl](predictions.jsonl).

## Data split provenance

The project dataset matches the AgentTTS HotpotQA dev file in all 7,405 IDs, questions, answers and supporting-title sets.
The project splits these rows into train 4814 (65%), validation 1110 (15%), and test 1481 (20%), seed 20260729.
The three index sets are disjoint. This is an internal routing split, not the original HotpotQA train/test split.
The archive lacks the split-generation script, so stratification and the interpretation of the alpha=0.6 metadata cannot be established.
Provenance checks: [data_provenance.json](../data_provenance.json).

Official HotpotQA train/dev expose answers and supporting facts; official test labels are withheld for evaluation.
Sources:
- https://github.com/hotpotqa/hotpot
- https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py
- https://github.com/FairyFali/AgentTTS/blob/3f0c2df814c059d0df57a71056c3266ff02c8559/data/HotpotQA/hotpotqa-dev.json
