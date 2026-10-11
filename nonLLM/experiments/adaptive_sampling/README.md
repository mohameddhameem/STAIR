# Per-question prefix sampling on serverA

User-selected alignment target: AgentTTS public HotpotQA full retrieval–merge–QA–fusion code, commit `3f0c2df814c059d0df57a71056c3266ff02c8559`. Temperatures are retrieval 0.9, QA 0.7, fusion 0.3; each generation has a 512-token limit. Prompt methods are extracted via AST from that source and executed unchanged, including Merge and evidence parsing. Fusion receives the original full reference pool twice, exactly as upstream. Even k=1 runs fusion. Native SGLang parameters explicitly match HF defaults top_k=50, top_p=1, repetition_penalty=1. Sampling seeds and backend metadata are persisted. Backend-dependent random streams are not expected to reproduce HF bitwise.

W = community Qwen2.5-1.5B Base GGUF Q4_K_M. S = community Qwen2.5-7B Base GGUF Q4_K_M. These models, quantization and serving backend differ from the paper. This is a prompt/settings-aligned experiment, not reproduction of the paper's numerical results. Same-model retrieval/QA combinations bypass upstream CLI's different-name assertion.

All 7,405 bundled HotpotQA questions are processed. Retrieval is sampled once with W and once with S and reused. For each WW/WS/SW/SS combination, generate 40 QA candidates once. For every k=1..40, fuse the first k candidates with the same QA model. This isolates answer-stage sampling budget; retrieval-budget search is not part of this run. Logical calls total 7,405 × (2+4×40+4×40) = 2,384,410. Forty candidates share a native batched request. Four question workers per GPU provide continuous batching, with prefix caching enabled.

Keep existing 65/15/20 train/validation/test IDs unchanged. Select global pair+k on validation and evaluate on the 1,481 held-out test questions. Report within-pair sampling oracle and joint pair+sampling oracle separately. Oracles use gold test answers after generation and are retrospective upper bounds. Their advantage does not establish a deployable adaptive controller. Cost-aware oracle frontier uses model-weighted token work including retrieval, QA and fusion; this is a compute proxy, not exact FLOPs. Raw fused text is scored with official HotpotQA answer EM/F1; missing upstream `evaluation_2wiki` prevents asserting identical original answer extraction. Full outputs permit later independent scoring.

Remote root is `/data1/zmcheng/cs707_sampling40`, containing transferred assets, weights, tokenizers, source, code, logs and SQLite results. The pre-existing SGLang environment is reused without modification. Each GPU runs a TP=1 replica; stages are W retrieval, S retrieval, W QA/fusion, S QA/fusion. `controller.py` shuts down only its own process groups if any external GPU process appears. Worker failure stops the experiment with `runs/FAILED.txt`; rerunning resumes successful SQLite records. `runs/*.done` marks complete shards; `analyze.py` refuses final comparisons before all 160 actions per question exist.

```bash
/home/zmcheng/miniconda3/envs/sglang518/bin/python code/controller.py
/home/zmcheng/miniconda3/envs/sglang518/bin/python code/analyze.py --root /data1/zmcheng/cs707_sampling40
```
