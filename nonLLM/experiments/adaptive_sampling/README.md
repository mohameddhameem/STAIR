# Per-question prefix sampling on serverA

The active user-requested rerun uses **official Instruct GGUF weights with
llama.cpp on serverA only**. See [GGUF_RUN.md](GGUF_RUN.md) and
[CURRENT_RUN.json](CURRENT_RUN.json). The configuration below describes the
archived Base GGUF/SGLang run and must not be used to resume the new experiment.

User-selected alignment target: AgentTTS public HotpotQA full retrieval–merge–QA–fusion code, commit `3f0c2df814c059d0df57a71056c3266ff02c8559`. Temperatures are retrieval 0.9, QA 0.7, fusion 0.3; each generation has a 512-token limit. Prompt methods are extracted via AST from that source and executed unchanged, including Merge and evidence parsing. Fusion receives the original full reference pool twice, exactly as upstream. Even k=1 runs fusion. Native SGLang parameters explicitly match HF defaults top_k=50, top_p=1, repetition_penalty=1. Backend metadata and server RNG seed are recorded. Deterministic inference is disabled for throughput; SGLang 0.5.18 ignores per-request seeds in this mode. Forty candidates use independent draws from the server RNG. Exact replay after resume is not guaranteed. Backend-dependent random streams are not expected to reproduce HF bitwise.

W = community Qwen2.5-1.5B Base GGUF Q4_K_M. S = community Qwen2.5-7B Base GGUF Q4_K_M. These models, quantization and serving backend differ from the paper. This is a prompt/settings-aligned experiment, not reproduction of the paper's numerical results. Same-model retrieval/QA combinations bypass upstream CLI's different-name assertion.

All 7,405 bundled HotpotQA questions are processed. Retrieval is sampled once with W and once with S and reused. For each WW/WS/SW/SS combination, generate 40 QA candidates once. For every k=1..40, fuse the first k candidates with the same QA model. This isolates answer-stage sampling budget; retrieval-budget search is not part of this run. Logical calls total 7,405 × (2+4×40+4×40) = 2,384,410. Forty candidates share a native batched request. Sixteen retrieval questions or four QA/fusion questions per GPU provide continuous batching, with prefix caching enabled. The longest (k=40) fusion is evaluated first to populate the prefix cache before dispatching k=1..39.

Keep existing 65/15/20 train/validation/test IDs unchanged. Select global pair+k on validation and evaluate on the 1,481 held-out test questions. Report within-pair sampling oracle and joint pair+sampling oracle separately. Oracles use gold test answers after generation and are retrospective upper bounds. Their advantage does not establish a deployable adaptive controller. Cost-aware oracle frontier uses model-weighted token work including retrieval, QA and fusion; this is a compute proxy, not exact FLOPs. Raw fused text is scored with official HotpotQA answer EM/F1; missing upstream `evaluation_2wiki` prevents asserting identical original answer extraction. Full outputs permit later independent scoring.

Remote root is `/data1/zmcheng/cs707_sampling40`, containing transferred assets, weights, tokenizers, source, code, logs and SQLite results. The pre-existing SGLang environment is reused without modification. Each GPU runs a TP=1 replica; stages are S retrieval, W retrieval, W QA/fusion, S QA/fusion. W replicas are retained between retrieval and QA/fusion. `controller.py` shuts down only its own process groups if any external GPU process appears. Worker failure stops the experiment with `runs/FAILED.txt`; rerunning resumes successful SQLite records. `runs/*.done` marks complete shards; `analyze.py` refuses final comparisons before all 160 actions per question exist.

```bash
/home/zmcheng/miniconda3/envs/sglang518/bin/python code/controller.py
/home/zmcheng/miniconda3/envs/sglang518/bin/python code/analyze.py --root /data1/zmcheng/cs707_sampling40
```

Initial 7B benchmark: 40 QA outputs, 13,976 generated tokens in 25.46 seconds. After longest-prefix warming, all 40 prefix fusion calls completed in 56.16 seconds on one question. This warm-cache microbenchmark is not a full-run time estimate or proof of globally optimal concurrency.

## Eight-GPU continuation on serverA and serverB

On 2026-10-12 the user authorized adding serverB's four idle L40S GPUs. Both servers use SGLang 0.5.18 and Torch 2.13.0+cu130 with the identical SHA256-verified 7B GGUF and tokenizer. Prompts, sampling parameters and candidate-prefix protocol are unchanged. Each host runs four TP=1 replicas. B has Python 3.11 whereas A has Python 3.12; record this runtime difference. Serving RNG is global and not deterministic per request. A restart preserves successful SQLite records and resumes missing requests; incomplete in-flight work may be regenerated.

The immutable assignment is A indices 0–3859 (including existing completed questions) and B indices 3860–7404. These sets are disjoint and cover all 7,405 questions. `tasks_A.json` and `tasks_B.json` specify ownership; workers retain the original index modulo four shard IDs. B reuses A's finished retrieval outputs, so no retrieval or weak-answer jobs are repeated. Helper assets, CUDA 13 toolkit, weights, logs and results are under `/data1/zmcheng/cs707_sampling40_helper`. The toolkit is copied into this folder without modifying the existing B environment.

`sync_helper.py` is a CPU-only local relay. Every five minutes it exports appended helper rows from SQLite read snapshots, transfers them and merges them into A's canonical databases. Merge retries are idempotent and conflicting payloads are rejected. It persists row offsets only after successful merge. `*.assigned.done` means a host finished its own assignment; final `*.done` markers and full evaluation require 82 saved actions for all 7,405 questions after merging. Final reports remain on A and are also copied locally to `/ssd1/zmcheng/cs707/runs/sampling40_helper_sync`.

A controller uses `CS707_INDEX_FILE=/data1/zmcheng/cs707_sampling40/code/tasks_A.json`. B uses `CS707_ROOT=/data1/zmcheng/cs707_sampling40_helper`, `CS707_HELPER=1`, `CS707_INDEX_FILE=/data1/zmcheng/cs707_sampling40_helper/code/tasks_B.json`, and `CS707_CUDA_HOME=/data1/zmcheng/cs707_sampling40_helper/runtime/cuda-13.0`. Both retain the external-GPU-process watchdog. `status.py --root ...` reports saved requests and assignment markers.

Validation: merge insertion, idempotent retransmission, conflicting-payload rejection, assignment coverage and preserved source model SHA256. All experiment code passes Python compilation.

The local relay is hosted in the detached tmux session `cs707_helper_sync`, with its log at `/ssd1/zmcheng/cs707/runs/sampling40_helper_sync/sync.log`. This keeps synchronization running after the interactive command exits. Both hosts passed health checks and began assigned 7B requests on all four GPUs.
