> Complete23 October2026. [Results and limitations](../../results/README.md); this run uses common0.2, not the new common0.3 SLR golden set.

# Four-model weak/strong GGUF experiment

Active run: `runs/nonllm_common_ws_gguf_v2`; tmux: `cs707-ws-fixed`.

User requested the same four executor model families as the teammate: Qwen2.5-7B/72B-Instruct retrieval and Llama-3.2-3B / Llama-3.1-70B-Instruct answering. Each router stage selects between weak and strong. This is the WS2x2 subset of common menu A, not the complete three-by-three AgentTTS search menu. Results carry `menu_variant=WS2x2` and explicit comparison scope.

Generation uses the existing verified Q4_K_M GGUF files, Q8 KV, K=1, greedy decoding and no fusion. Shared common IDs: train_extra 6405, val 500, test 500. Metrics: author lenient match and AgentTTS normalized FLOPs. GGUF results remain separate from BF16/AWQ pooled results unless the team adopts identical weights and execution settings.

The previous six-model run was stopped via its STOP file. Migration validates data, splits, generation, scoring and model provenance before copying committed relevant records. It excludes QA records conditioned on Qwen32. Old records and intermediate model downloads remain preserved. Migration provenance is in `migration.json`.

There are 7405 questions x (2 retrieval + 4 answering) = 44430 model calls. The scheduler uses up to four local GPUs. Following generation, six classical router families (majority, logistic regression, calibrated linear SVM, multinomial NB, random forest, extra trees) train automatically with seeds 0/1/2. Hyperparameters and routing penalties are chosen on validation before test scoring. All four fixed combinations, including Qwen72+Llama3 and Qwen72+Llama70, are evaluated as controls. This K=1 experiment does not reproduce the original 40-answer-sample AgentTTS recipe. GRPO is not part of this initial baseline stage.

Run from the project root:

```bash
.cascade-env/bin/python STAIR/nonLLM/experiments/menu_ws_gguf/status.py
.cascade-env/bin/python STAIR/nonLLM/experiments/menu_ws_gguf/test_contract.py
```

## Template repair

The first four-model run is invalidated for model/router comparisons: Llama70's embedded legacy template receives an empty tool list from the backend, and tests `tools is not none`, injecting a request to emit function-call JSON even though no tools were requested. Observed JSON-call signatures: 12416/12810 train, 969/1000 validation, 972/1000 test answers. Llama3: zero across all splits.

The corrected run passes `chat_template_kwargs.tools=null` explicitly to both `/apply-template` and `/v1/chat/completions`. The Llama70 template date is fixed to `22 Oct 2026` to prevent changes during multiday generation. A preflight guard rejects injected tool instructions or an ipython environment. Weights, references, task messages, decoding and scoring are unchanged. Only the Llama70 output cache is regenerated; all router labels, fits and test evaluations are rebuilt afterward. Do not use the old REPORT.md as final evidence.

Audit files: `runs/nonllm_ws_template_audit_v1/`; provenance in corrected run `migration.json` and frozen `code/` snapshot.

## Concurrent generation

The initial concurrency optimization used two two-GPU replicas with up to four active requests each (superseded by the 16-request setting below). The server uses four slots and a unified 16384-token KV pool. Client-side admission reserves rendered prompt tokens plus the maximum 512 output tokens; long requests wait for capacity. Each request retains the original 16384-token context limit. New cache records include `server_parallel` and `kv_unified`. Existing corrected serial records are reused. Batch execution can change floating-point results and final text, so serial bitwise equivalence is not claimed. All router policies use the same combined cache.

The eight-training-case diagnostic benchmark took 70.83 seconds serial, 41.36 seconds at concurrency two, and 35.78 seconds at concurrency four. Four concurrent requests delivered 17.08 generated tokens/s versus 11.94 serial. See `runs/nonllm_ws_concurrency_audit_v1/REPORT.md`. Two of eight texts differed from the serial baseline. These figures are not whole-run speed or accuracy guarantees.

Concurrency-specific checks cover bounded in-flight tasks, one result per job, and shared KV admission/release including errors. Runtime provenance and the old source snapshot are retained in the run directory.

## Current runtime: 16 requests per replica

The active setting is `CS707_REQUESTS_PER_REPLICA=16`: two model replicas, each on two GPUs, for up to32 simultaneous requests. Unified KV capacity remains16384 tokens per replica; the existing prompt+512 token admission budget may reduce actual concurrency for long inputs. Context and decoding limits remain unchanged.

On the same sixteen training questions, GPU0/1 improved from24.53 tokens/s at4 requests to27.60 at8 and35.28 at16. GPU2/3 delivered22.33 at8 and34.53 at16. These small-sample measurements include numerical/output-length variation and are not whole-run guarantees. See `runs/nonllm_ws_concurrency_audit_v2/REPORT.md`. Runtime configuration and prior source snapshots are stored in the run directory.
