# Local menu-A GGUF non-LLM routing

User-authorized local variant,18 October2026: reuse the six already verified Q4_K_M models and use up to four RTX A5000 GPUs. It uses the team's common question IDs, author lenient scorer and AgentTTS cost formula, **but is not eligible for pooling with bf16/AWQ results unless the team adopts identical GGUF copies**. Quantization/backend remain an experimental difference; do not label this as complete AgentTTS replication.

## Model sources

- Qwen2.5-7B/32B/72B-Instruct: official Qwen GGUF repositories.
- Llama-3.2-3B and Llama-3.1-8B/70B-Instruct: bartowski GGUF repositories.
- Fixed model revisions and every shard hash are in the run protocol and `models/agenttts_gguf/manifest.json` in the workspace. All118.01GB /22 shards were SHA256-verified before this run. No Unsloth files have been substituted; switching publisher/version requires a new cache identity.

## Protocol

Training-size clarification (19 October): common defines the6,405 `train_extra` IDs as an optional eligible training pool; it does not require every router to train on all6,405. Using the whole pool is this run’s declared design choice. A smaller preregistered training subset and learning curve are possible without changing the common500 validation/test IDs. Existing binary Qwen1.5B/7B labels are not labels for this six-executor9-cell GGUF setup. No training subset or active job was changed by this clarification.

- HotpotQA common `train_extra`6,405, validation500, test500, selected by committed IDs.
- Direct two-stage routing: select one of3 retrievers from question+candidate pool, execute it, then select one of3 answerers from question+actual evidence. No gold metadata or unexecuted answer is a feature.
- Executor weights frozen. K1, greedy, max512 new tokens, no answer fusion. Model-native chat templates; supplied first100 passages, each capped600 characters, no reshuffling or fresh gold insertion. Context16,384, oversize input rejected rather than silently truncated.
- llama.cpp build commit `d89651a`; GPU layers fully offloaded; flash attention on; Q8_0 K/V caches for all models. Q4_K_M weights and Q8_0 KV are separate quantization settings and both must be reported.
- The executor prompts/decoding are this local run's declared control protocol. The common package does not impose them. Router-only comparisons with another track require that track to replay the exact same executor caches; otherwise compare complete systems and state generation differences.
-9-cell offline grid:3 retrieval calls +9 answering calls per question =88,860 calls. Offline counterfactual generation is training/data-collection overhead; deployment cost is only the selected retrieval and answering calls.
- Six classical families: majority, TF-IDF logistic regression, calibrated linear SVM, multinomial NB, random forest and ExtraTrees. Router seeds0/1/2; executor generation seed0 shared across fits. Question-grouped training folds keep QA branches together. Feature vocabulary fits training only. Tree inputs use numeric text features.
- Labels use cheapest correct downstream routes on train; all-incorrect groups are masked. Stage fit settings use validation log loss. Operating-point penalties/fixed fallbacks are locked on validation before test feature construction or scoring. Counterfactual per-test costs never become routing inputs: deployment cost penalties use training-mean estimates only.
- Classical fits have no neural epoch count. GRPO is not included in this first supervised local run.
- Compare all9 fixed routes and the strongest Qwen72B+Llama70B route on the same new caches. Report common lenient/EM/F1, absolute AgentTTS units, relative cost and measured per-question router latency. Old52.79/52.94/published74.6 numbers are not substituted for this baseline.

## GPU execution and progress

A two-training-question pilot completed all24 calls (6 retrieval +18 QA), covering all six models, before full generation. Max pilot retrieval prompt11,444 tokens. Both large models fit a pair of A5000 cards with the declared KV setup. No evaluation/test question was used to tune the pilot. Four information-boundary/class-mapping tests pass.

Full generation runs model by model. Small models can have four single-GPU replicas;72B/70B can have two replicas on two GPUs each. At each stage the scheduler selects only free cards; external workloads are never terminated or adopted. If a large model lacks a free pair, it waits and records `resource_wait.json`. JSONL caches are committed per response and resumed by key, with interrupted final writes quarantined before reuse. Status/logs identify physical GPU allocation.

Current full run: workspace `runs/nonllm_common_menuA_gguf_v2`, tmux session `cs707-menuA-gguf`. This version imports only the proven pilot cache from `..._v1`; the earlier bf16/AWQ preparation directory is separate and is not running.

Artifacts: `status.json`, `active_model.json`, `workers/*.json`, `cache/*.jsonl`, `pilot.json`, `protocol.json`, `pipeline.log`; then `routers/`, `fit_records.json`, `validation_lock.json`, `results.csv`, `results.json`, `REPORT.md`. The CSV is schema-compatible but has explicit `comparison_scope`, weight/KV quantization columns and `gguf-` policy prefixes; this does not make it pooled-eligible.

Local launch from workspace root:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 .cascade-env/bin/python -u \
  STAIR/nonLLM/experiments/menu_a_gguf/run.py \
  --out runs/nonllm_common_menuA_gguf_v2 \
  --import-pilot runs/nonllm_common_menuA_gguf_v1
```

Do not run a duplicate controller while the tmux session is active. Add a `STOP` file to the run directory for a graceful cache-preserving stop at the next request boundary. No HPC jobs or external messages are submitted by this pipeline.
