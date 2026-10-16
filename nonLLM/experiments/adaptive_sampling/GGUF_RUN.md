# ServerA-only Instruct GGUF prefix-sampling run

The user requested a new full run with GGUF Instruct models on all four serverA
L40S GPUs, without using serverB. This supersedes the BF16 diagnostic deployment
and the earlier Base GGUF experiment. Old outputs remain separate.

At launch on 2026-10-15, physical GPU2 returned an uncorrectable ECC error
(volatile DRAM count 2; row remapping pending). The four-card attempt stopped
before full workers started. GPU reset was blocked by dcgm-exporter/nvtop
monitoring handles; permission to close those monitoring processes was requested.
The full workload then started on healthy physical GPUs **0, 1, 3**, without
serverB. All 7,405 questions remain assigned. Three-GPU work queues partition
each of the four logical SQLite shards into disjoint subsets, with each physical
GPU processing its subsets sequentially. Final logical-shard counts are checked
before marking a phase complete. This can resume on four GPUs after recovery.

On 2026-10-15 the user authorized pausing the run to repair GPU2. The controller
stopped, preserving all committed requests. After releasing monitoring clients,
the reset was still blocked until the idle `nvidia_drm`/`nvidia_modeset` modules
were temporarily unloaded. A targeted UUID reset succeeded at 18:49 +08; the
modules and system monitoring were restored without rebooting the server.
GPU2's row remapping is no longer pending, recovery action is None and volatile
DRAM uncorrectable errors are zero. A 36 GiB two-pattern memory check and 30
4096-square matrix checks passed. Historical aggregate error counts remain.
The workload resumed on **all four physical GPUs 0, 1, 2, 3**, reusing its caches.
The four-card controller now uses disjoint rank subsets within every logical
shard, as the three-card queues did. This distributes the earlier run's cache
skew across all four GPUs instead of assigning the mostly cached shard to one
GPU and leaving that GPU idle early. Three- and four-rank assignments were
checked to cover all 7,405 questions exactly once.

Training-only fusion throughput measurements led to a separate
`runtime_schedule.json`: W uses one fusion lane, S retains two; both keep two
question lanes. For the same 24 W fusion requests on training indices 0 and 12,
one lane took 17.11 s, two took 24.80 s, and four took 45.24 s. Eight and twelve
lanes were much slower and stopped before completion. For S, one/two lanes
took 61.74/64.35 s with different output lengths; that small difference did not
justify changing the established two-lane setting. Higher utilization alone
does not indicate higher request throughput. This small benchmark does not
establish the speedup over the whole corpus. `gguf_throughput.py` records the
workload and raw diagnostic replies outside experiment databases.

Only worker scheduling and logging changed. Model/prompt/sampler/parser files,
the frozen profile and database fingerprint are unchanged. The scheduling-only
worker lock update and original source/lock are archived under
`maintenance/20261015_184550` in the run root. No existing results were deleted.

Remote run root: `/data1/zmcheng/cs707_sampling40_instruct_gguf`.
On 2026-10-16 the user requested an early Strong reference batch before the
Weak stage finishes. See [STRONG_PILOT.md](STRONG_PILOT.md): twelve training
questions get matched Strong sampling/fusion and answer extraction, using the
same full-run caches, followed by automatic full-controller restoration.
After the reference batch finished, the full run was paused for review at
2026-10-16 15:50 +08. All generation checkpoints are preserved. A separate
evaluation_v2 report re-extracts the existing twelve-question replies with the
same grounded reader for narrative responses; original caches remain unchanged.
The current root state and CURRENT_RUN.json indicate this pause.
All runtime files, weights, tokenizers, source snapshots, logs and caches are
under this folder. Unchanged assets and libraries are hard-linked to avoid extra
disk allocation; they are never modified in place.

| Role | Official GGUF repository | Revision | File |
|---|---|---|---|
| W | Qwen/Qwen2.5-1.5B-Instruct-GGUF | 91cad51170dc346986eccefdc2dd33a9da36ead9 | qwen2.5-1.5b-instruct-q4_k_m.gguf |
| S | Qwen/Qwen2.5-7B-Instruct-GGUF | bb5d59e06d9551d752d08b292a50eb208b07ab1f | qwen2.5-7b-instruct-q4_k_m-00001/00002-of-00002.gguf |

Both strong-model shards and the weak-model file were checked against their
Hub LFS SHA-256 values. `profile.json` records the exact names and hashes.
The serving backend is llama.cpp commit
`d89651a7b205c03c4a0b13cd0646d400dc929f79`, with CUDA full model offload.
No SGLang GGUF loader or BF16 model fallback is used.

## Frozen experiment

* All 7,405 questions and the existing 4,814/1,110/1,481 train/validation/test split.
* Four pairs WW/WS/SW/SS. First letter is retrieval model, second is answering.
* One retrieval per model/question. Forty QA candidates generated once per pair.
  Prefix k always contains exactly candidates 0 through k-1 from that batch.
* All prefixes k=1..40 are fused, including k=1. Public generation prompts and
  merge parsing come from pinned AgentTTS commit
  `3f0c2df814c059d0df57a71056c3266ff02c8559` via `protocol.py`.
* Retrieval 0.9, QA 0.7, fusion 0.3; maximum 512 new tokens. **No temperature grid.**
  Other values come from the matching Instruct generation configurations:
  top_k=20, top_p=0.8, repetition penalty W=1.1/S=1.05. The native sampler order
  is penalties, temperature, top-k, top-p, and its repetition window covers the
  complete prompt/output. Quantization and engine differences still preclude
  identical token draws or a numerical reproduction of AgentTTS.
* QA draw order is native task-index order, not completion order. Stable seeds
  derive from request keys; native child draws increment the seed. Seeds and
  output order are saved. The retrospective oracle remains optimistic because
  it selects among stochastic outcomes using gold answers.

## Serving and safety

Each GPU runs one independent replica. The four GPUs process the same model
phase together: W retrieval, S retrieval, W answers, S answers, S answer reading.
The controller reuses the strong replicas between answers and reading.
Every question belongs to exactly one GPU shard by index modulo four. There is
no serverB helper, cross-server model endpoint, or partial assignment.
When a GPU is unavailable, `--gpus 0 1 3` selects the healthy devices; logical
shard identity and per-request settings/seeds remain unchanged.

Each replica has 64 native slots and a 524,288-token unified FP16 KV pool.
Native `n_cmpl=40` forks share prompt KV. Long fusion calls use two concurrent
lanes per GPU to preserve prefix locality; admitting many long fusions at once
replicates prefill and slows this backend. A separate weighted admission limit
bounds active token/slot reservations. Retrieval and answer generation each
have two question pipelines per GPU. An initial 16-way retrieval batch took
about 60 seconds; its unrelated long prompts expand the unified attention pool.
The scheduler was reduced to two lanes without changing request parameters,
and committed outputs remain valid.

Per-request context is 131,072 with explicit YaRN factor 4, original context
32,768. A runtime context metadata override prevents the existing server's
training-context cap from silently restricting requests. Extension of the 1.5B
checkpoint is experimental. Both actual GGUF models passed a 60,244-token needle
canary; this is limited validation, not a general long-context capability claim.
A conservative bound over all 7,405 fusion prompts, including 40 maximum-length
answers and fusion output, was 66,965 tokens. Actual QA prompts are checked again
after retrieval. Requests fail rather than silently truncating or shifting context.

The controller refuses occupied GPUs and watches for external GPU processes.
On conflicts or worker/server errors it stops only its own processes and writes
`runs/FAILED.txt`. SQLite commits preserve completed requests. The profile,
inference code and input files are locked to prevent mixing changed protocols.
HTTP connections close after requests to avoid stale server keep-alive sockets.

## Scoring and gates

Raw normalized EM/F1 and a locally defined guarded lenient containment metric
are saved separately from extracted-answer metrics. The upstream scorer was not
supplied, so the lenient implementation is not certified identical to the original.

Answer extraction first uses conservative parsing, then a fixed 7B **GGUF**
greedy span reader when needed. It receives the question and model reply only,
never gold answers or reference evidence. Quoted evidence must occur in the
reply. Copied questions cannot support a yes/no answer. This explicit guard was
added after an initial GGUF reader control incorrectly answered a copied math
question; that failed control is retained in `gates/initial_reader_echo_failure.json`.
Abstentions count as wrong. Reader tokens/time are additional recorded costs.
Grounding and controls do not prove perfect semantic faithfulness.

Before full launch, `gguf_gate.py` checks short instructions, 40 distinct seeded
draws, reader controls, long input, and actual retrieval/40-candidate fusion on
two predetermined training questions. `gguf_smoke.py` then exercises the actual
resumable worker, all 160 pair/budget outputs, and extraction for training index
0. Its valid cache can be reused by the full run. No test answers choose a model,
prompt, threshold or generation setting.

`gguf_analyze.py` selects the globally fixed pair/budget on validation, then
reports held-out test fixed scores, per-pair budget oracles and the joint oracle.
It reports low-budget/large-budget win counts and an evaluated cost-constrained
oracle frontier. Extracted metrics include 7B reader token cost. The proxy is
parameter-weighted token work, not measured FLOPs, wall time or monetary cost.
Fusion at k=1 and reader costs are included. The cost frontier is an evaluated
Lagrangian set, not an exact constrained optimum.

Run and status, on serverA with the run root as working directory:

```bash
/home/zmcheng/miniconda3/envs/sglang518/bin/python code/gguf_controller.py --root "$PWD"
/home/zmcheng/miniconda3/envs/sglang518/bin/python code/gguf_controller.py --root "$PWD" --gpus 0 1 3
/home/zmcheng/miniconda3/envs/sglang518/bin/python code/gguf_status.py --root "$PWD"
```

`state.json` identifies the live phase and PIDs. Worker logs and committed SQLite
counts provide progress. `analysis/report.json`, `analysis/scores.npz` and
`analysis/per_question_oracle.csv` are written after all generation and reading
complete. `COMPLETE` is written only after final analysis succeeds.
