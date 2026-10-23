# First common-standard non-LLM run: HotpotQA, AgentTTS menu A

This is preparation for the six-model **menu**, not a claim to reproduce AgentTTS's full stochastic sampling/search algorithm. The old binary W/S experiment cannot be relabeled as menuA.

The common-pinned executors are three Qwen retrievers (7B/32B bf16,72B AWQ) and three Llama answerers (3.2-3B/3.1-8B bf16,3.1-70B AWQ). Use the exact revisions in `common/stair_common/models.toml`; no GGUF substitution. Executors are frozen for this routing-only baseline.

Prepare from repository root after installing common:

```bash
python nonLLM/experiments/common_menu_a/prepare.py
```

It writes an isolated protocol and copied common ID manifests under the sibling workspace's `runs/nonllm_common_menuA_v1`. This does not train or submit jobs. A sibling workspace with the existing HotpotQA source is required.

## First experimental protocol

- Train routers on common HotpotQA `train_extra` (6,405), validate on500, test on500. All splits are selected by committed question IDs. Do not reuse old router weights.
- Direct two-stage routing: question+candidate pool selects one of3 retrievers before execution; question+selected evidence selects one of3 answerers before execution. No weak-answer features or mandatory weak calls.
- First control uses K1, proposed greedy generation without fusion. **These decoding choices are our proposed router-control protocol, not yet a team-wide prompt/decoding standard.** Original AgentTTS uses sampling and fusion; it needs a separate properly charged comparator if included.
- Train majority, TF-IDF logistic regression, linear SVM, multinomial NB, random forest and ExtraTrees. Retain question-grouped training folds for QA's3 evidence branches. Seeds0/1/2. Iterative additions use validation-loss stopping/common history; do not fabricate neural epochs for tree fits.
- For complete offline label generation, execute3 retrievers and3 answerers on each retrieved branch:9 outcome cells per question,12 model calls per question,88,860 calls over7,405 questions. At deployment only the chosen route is charged.
- Recompute the strongest baseline: Qwen72B AWQ retrieval + Llama70B AWQ answering, K1. Also report all9 fixed policies and accuracy/cost operating points. Do not reuse52.79%,52.94% or published74.6% as this run's measured baseline.
- Standard lenient/EM/F1, measured per-question AgentTTS cost units and routing latency; export `stair_common.schema` columns. All model/threshold selection uses train/validation, never test.

## Comparing with other tracks

For a router-only comparison, all tracks must share identical executor caches, revisions, candidate pool/order, prompts, evidence formatting, decoding, seeds and cost/score/split definitions. Common currently pins model/split/scorer/cost/output contracts but does not freeze a universal prompt/decoding/candidate-pool contract. Ask for a matching shared cache before repeating expensive generation. If executor calls differ, label the comparison as whole-system rather than attributing it solely to the router. No other track's common-standard results have been imported here.

## Preflight, 18 October 2026

- Six pinned model endpoints checked. Both official Llama3B and Llama8B weight URLs return HTTP401 (`GatedRepoError`); an account with the model licences accepted must log in locally, or supply the exact pinned local copies. Never paste tokens into a chat or commit them.
- vLLM is not installed in the current environment. Use an isolated vLLM environment rather than changing an unrelated existing inference stack.
- Qwen32B bf16 weights alone are65.53GB. Two local24GB A5000 GPUs cannot hold them, even before KV cache. A two-GPU cluster run needs sufficiently large GPUs; using more local GPUs or CPU offload would require an explicitly recorded hardware/backend setting. Do not quantize the model as a workaround and call it the same common run.
- The existing Qwen7B HF download metadata names the common-pinned revision; metadata/weight checks and bf16 re-generation are still required before cache reuse. Old FP16 inference outputs are not automatically the common vLLM outputs.
- Available six GGUF copies remain exploratory; they are not loaded by this proposed pooled protocol.

Current status: **prepared, not trained**. Models/access, backend/resources and the matched executor-cache contract remain prerequisites. No cluster jobs were submitted and no expensive full grid generation was launched by preparation.
