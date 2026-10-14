# Instruct repair and serving validation

The completed Base GGUF run is preserved in `../../results/adaptive_sampling40_base`.
Its raw EM of zero combines a generation failure with an answer-format mismatch.
It cannot establish whether per-question sampling budgets improve accuracy.

## Scope and evidence

The user selected preservation of the public AgentTTS prompts and permitted Instruct
models. Retrieval, QA, merge parsing and fusion prompts still come directly from
AgentTTS commit `3f0c2df814c059d0df57a71056c3266ff02c8559`, through `protocol.py`.
Retrieval/QA/fusion temperatures remain 0.9/0.7/0.3, with 512 generated tokens.
QA candidates are generated once; budget k uses exactly their first k entries.
Fusion is still performed at k=1.

Two separate problems were observed in the old run.

* Many Base generations repeated instructions, changed the question, or invented
  reference content. A diagnostic sending identical token IDs and the same 7B
  GGUF to llama.cpp and SGLang produced different greedy completions. This does
  not identify a specific loader/kernel defect. Switching to official BF16
  Safetensors avoids that unvalidated GGUF serving path.
* The public prompts permit explanatory paragraphs. Even sensible Instruct
  answers therefore fail exact match against a short gold answer. The upstream
  `evaluation_2wiki` module is missing, so identical scoring with AgentTTS cannot
  be claimed.

`diagnose_generation.py` and `run_native_controls.py` include gold-evidence and
gold-candidate controls to isolate generation behavior, plus modified-prompt
controls. Those are explicitly diagnostics. `repair_pilot.py` instead runs
actual model retrieval and model-generated candidates; it does not substitute
gold references or answers into the public generation pipeline.

Official downloaded checkpoints on serverA are:

| Role | Model | Hub revision |
|---|---|---|
| W | Qwen/Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 |
| S | Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 |

Source-style `model.generate` inherits each model's generation configuration.
For these Instruct checkpoints that means top_k=20, top_p=0.8, repetition penalty
1.1 (W) or 1.05 (S), and EOS IDs 151645/151643. Stage temperatures override the
checkpoint temperature. An earlier diagnostic used 50/1/1 and is retained
separately; it is not the source-defaults pilot.

Inspection of this SGLang installation showed that its native repetition
penalizer tracks generated tokens, whereas Hugging Face's default processor
includes prompt tokens too. The initial pilots therefore matched parameter
values but not that rule. `hf_repetition.py` supplies an optional custom logit
processor over both prompt and output tokens, with the native penalty disabled
to prevent double application. `check_hf_repetition.py` matched the installed HF
processor exactly in 16 CPU checks. Use `--hf-repetition-penalty` on a service
started with `--enable-custom-logit-processor`; the module must be on the server's
PYTHONPATH. The corrected diagnostic launcher sets both. Different engines and
batching still do not reproduce identical random draws; these tests do not prove
full backend equivalence.

## Answer extraction is an added evaluation step

`answer_parser.py` first tries conservative deterministic parsing.
`grounded_answer.py` optionally uses a fixed 7B Instruct reader on the question
and generated response only. Gold answers and reference documents are never
passed to this reader. It must quote evidence from the response; non-polar
answers must also be literal spans in that evidence. A yes/no stance may be
inferred from quoted response text. This is not a proof of semantic faithfulness,
especially for implied yes/no answers, so inspection and abstention remain
necessary. Invalid extractions count as incorrect.

Raw EM/F1 and extracted EM/F1 are reported separately. Reader tokens and time
are additional costs. This evaluation extension must be applied consistently to
all compared configurations; it is not the original AgentTTS scorer. Synthetic
checks deliberately include wrong assertions, question echoes and unanswerable
responses. The reader must preserve wrong assertions or abstain, never correct
them using its own knowledge. Ten safety checks passed with one abstention on
an answer-bearing example; this is not ten successful answer extractions.

## Long-context configuration

Both checkpoint configurations specify 32,768 positions. Forty QA responses,
plus the public fuser's repeated reference pool, can exceed that. The 7B model
card documents YaRN extension. The 1.5B card specifies a 32,768-token context;
its extension here is experimental and not a validated 128K capability claim.

In this installation (SGLang 0.5.18, Transformers 5.12.1), `get_rope_config`
prefers `rope_parameters`. Setting only the older `rope_scaling` field left
the effective rotary configuration at `default`. Both models then generated
garbage on a 60,244-token canary, despite receiving the full input. The corrected
diagnostic launcher explicitly sets:

```json
{
  "max_position_embeddings": 131072,
  "rope_parameters": {
    "rope_type": "yarn",
    "rope_theta": 1000000.0,
    "factor": 4.0,
    "original_max_position_embeddings": 32768
  }
}
```

Both corrected services recovered a synthetic code placed near the beginning
of 60,244 tokens of varied article text under the unchanged public QA prompt.
Both reported receiving all 60,244 tokens. W also hallucinated unrelated
explanation, so this is a context/serving canary, not a long-context accuracy
benchmark. Earlier failed controls remain archived. A single canary does not
establish reliable retrieval across the full context window.

## Running isolated checks

All remote paths are under `/data1/zmcheng/cs707_sampling40`; diagnostic models,
logs and results are in `diagnostics`. The local raw diagnostic backup is
`/ssd1/zmcheng/cs707/runs/sampling40_repair_diagnostics`.

`quality_gate.py` checks short chat instructions. `long_context_check.py` checks
the full long input and exits nonzero on failure. `repair_pilot.py` exercises
actual retrieval, QA sampling, prefix fusion and answer extraction on training
questions only. Model revision and deployment profile must be recorded alongside
results. Example diagnostic invocations, with services already running:

```bash
python repair_pilot.py --label new_short_diagnostic
python long_context_check.py --label yarn_v5
python repair_pilot.py --label new_40_diagnostic --long-context \
  --questions 2 --candidates 40 --budgets 1 10 20 40
python repair_pilot.py --label new_hf_penalty_diagnostic --long-context \
  --hf-repetition-penalty
```

`worker.py` now accepts `--tokenizer-path`, `--generation-config`, `--runs-dir`
and `--context-limit`, plus `--hf-repetition-penalty`. Both model arguments must be supplied together. The
Instruct profile requires a separate runs directory, stores its configuration in
SQLite metadata, and refuses to resume a cache with different generation
settings. Pilot caches also bind their profile; archived unversioned pilot caches
require a fresh label to rerun. Existing Base behavior remains the default. This worker records raw
generations; the original analysis script still scores raw text. Answer
extraction is currently implemented in the repair pilot and must be integrated
with full-run scoring before making extracted-score oracle comparisons.

Validation included real W/S retrieval calls using the new worker options and
an intentional context-profile mismatch that was rejected before generation.
The final 8-question training pilot also exercised actual retrieval, QA and
fusion with the HF-compatible repetition processor and YaRN configuration.
Its 64 outputs contained 34 extracted exact matches; this is not a held-out
score or the performance of a selected router. A separate native-penalty check
completed 40 candidates and prefixes 1/10/20/40 for two training questions.
Temporary serving processes were stopped after verification; a separate GPU2
training process on serverA was left untouched.
This repair does not automatically launch a full 7,405-question rerun.

References: [Qwen Base usage](https://huggingface.co/Qwen/Qwen2.5-7B),
[7B Instruct and YaRN](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct),
[1.5B Instruct context specification](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
