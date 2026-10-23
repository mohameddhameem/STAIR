# AgentTTS model and execution audit

Audited local upstream commit: `3f0c2df814c059d0df57a71056c3266ff02c8559`.
Source repository: https://github.com/FairyFali/AgentTTS

## Models and stage-specific choices

HotpotQA / 2WikiMultiHopQA have **six**, not four, executable model candidates:

| Stage | Small | Medium | Large |
|---|---|---|---|
| Retrieval | Qwen2.5-7B-Instruct-AWQ | Qwen2.5-32B-Instruct-AWQ | Qwen2.5-72B-Instruct-AWQ |
| Answering | Llama-3.2-3B-Instruct | Llama-3.1-8B-Instruct | Llama-3.1-70B-Instruct |

The requested complete six-model set is being downloaded. For comparison, a four-model endpoint subset would drop both middle sizes: retrieval Qwen7B/72B; answering Llama3B/70B. This is a subset of the original HotpotQA search space, not the complete original six-model setup. CWQ/WebQSP instead use exactly four candidates: Qwen7B/72B and Llama8B/70B. Do not silently substitute Llama8B for HotpotQA's smallest Llama3B.

A separate search controller defaults to `o3-mini-2025-01-31`, `reasoning_effort=high`. This controller is not a retrieval/answering candidate and is not included in the six local GGUF model downloads.

## Where decisions come from

1. An offline LLM search agent proposes a model and sample count **for each stage**, subject to a total compute budget. It uses task-level evaluation history; it is not the per-question W-first review router used by the October SLR reference.
2. Candidate generation offers samples `[1,100]` in its prompt, but the implementation is not a universal hard 100 cap: initialization derives counts from budget and random/mutation code uses other rules. Defaults: budget1000, five iterations, ten candidates per iteration.
3. The bundled `Environment` reads CSV experiment results, filters by stage-specific model names and budget, and falls back to the nearest sample counts when an exact configuration is absent. The proposal can therefore be mapped to an existing measured configuration.
4. The selected strategy is then a stage-model/sample allocation evaluated on questions. No requirement to run a smaller model before selecting a larger one exists in this execution pipeline.

## What the retrieval model selects

The provided data generator builds an Elasticsearch corpus from HotpotQA context paragraphs. For each question it starts with gold supporting paragraphs, appends unique paragraphs from a 100-hit title/text query, and shuffles the candidate list. Consequently this construction is gold-injected and can contain more than100 paragraphs; the task description rounds it to a100-chunk pool.

The retrieval LLM sees the question plus the supplied numbered `ctxs_candidate` passages and emits selected reference IDs/text. It is selecting from this supplied pool, not browsing the web and not choosing its own candidate model list.

For repeated retrieval, `Merge` extracts all reference IDs, uses the most common returned list length, and keeps that many most frequent IDs. This is deterministic frequency aggregation, not a separate model call. With no parsed IDs it falls back to reference1.

The QA model sees only the selected original reference texts and question. With multiple answers, the configured `fuser` uses the same QA model to combine answers. This additional generation call must be counted.

## HotpotQA runtime settings

Main combined retrieval–merge–answer–fusion branch (`--generate`, without `--no_retrieve`):

| Component | Temperature | Max new tokens | Count |
|---|---:|---:|---|
| Retrieval |0.9|512|selected retrieval sample count|
| Merge |N/A, deterministic Python|N/A|no model call|
| Answering |0.7|512|selected answer sample count|
| Answer fusion |0.3|512|one call, configured even at answer sample count1|

The no-retrieval diagnostic branch uses answer temperature0.3 for one answer or0.9 for multiple answers; multi-answer fusion uses temperature0.3 and max1024 new tokens. These branch settings are different and must not be conflated.

The custom Transformers generator uses `do_sample=True`, chat templates, input truncation, and batches up to10 return sequences. It does not explicitly set `top_p` or `top_k`; effective values depend on each model's generation configuration/library defaults. This differs from our earlier greedy deterministic SLR cache generation. Budget-planning lengths (`Np=[2048,256]`, `Nd=[128,64]`) are cost assumptions, not the runtime512-token limits.

The runtime invokes an imported `evaluation_2wiki.metric_evaluation` and reports `Gen_EM`, token F1/precision/recall and retrieval metrics; search optimizes `Gen_EM`. The missing evaluator source prevents certifying equality with the SLR lenient grader. Do not call them identical.

## GGUF correspondence and precision

Downloaded files use Q4_K_M, a GGUF quantization of the same original Instruct checkpoint families. Original Qwen execution loads AWQ checkpoints, whereas the Llama loader uses bitsandbytes4-bit NF4, double quantization and FP16 compute. GGUF Q4_K_M is **not bitwise identical** to either original quantization/backend; matching model identity does not guarantee matching answers or scores.

Pinned repositories, filenames, expected sizes and SHA256 hashes are saved in `models/agenttts_gguf/available_files.json`. Download only Q4_K_M shards and keep every shard for a model together. Pass the first shard to llama.cpp.

## Reproduction gaps in the released code

The local release lacks the referenced HotpotQA generated `tts_*.json` files, `CSV_files/ret_then_gen_hotpotqa2.csv`, and `evaluation_2wiki.py`. In addition, `unit_budgets['hotpotqa']` is empty while initialization indexes its stage entries, so the HotpotQA search entry needs repair/configuration before it runs. The documented defaults alone do not constitute a fully runnable exact reproduction.

## Source locations

- `tools/AgentTTS/code_others/AgentTTS.py:403` — HotpotQA stage model lists.
- `tools/AgentTTS/code_others/AgentTTS.py:415` — CWQ/WebQSP four-model lists.
- `tools/AgentTTS/code_others/agent/agent_class.py:277` — search controller.
- `tools/AgentTTS/code_others/agent/agent_class.py:534` — cached HotpotQA environment.
- `tools/AgentTTS/code_others/archon_hotpotqa.py:441` — combined execution configuration.
- `tools/AgentTTS/code_others/archon_hotpotqa.py:143` — retrieval vote aggregation.
- `tools/AgentTTS/code_others/archon/completions/utils.py:601` — AWQ loader.
- `tools/AgentTTS/code_others/archon/completions/utils.py:630` — Llama NF4 loader.
- `tools/AgentTTS/code_others/archon/completions/utils.py:951` — sampling implementation.
- `tools/AgentTTS/data/retrieval_QA_data_generator.py:124` — HotpotQA pool generation.
