# Generation repair diagnostics

These are training-only serving and scoring checks, not held-out results.
Original AgentTTS generation prompts remain unchanged. The user approved
switching W/S to official Qwen2.5-1.5B/7B Instruct BF16 checkpoints.

The final 8-question check used corrected YaRN deployment and the optional
HF-compatible prompt-and-output repetition penalty. All four pairs and prefixes
1/4 completed, with four candidates generated once per pair. Of 64 outcomes,
34 matched the gold answer after extraction (EM 53.125%, F1 54.7123%). Raw paragraph
EM remained zero. Reader cost was 29,009 prompt tokens and 2,246 generated tokens;
the sum of individual request times was 48.27 seconds, not wall-clock compute.
The report and cache profile are `pilot_8_questions_hf_penalty_yarn.json` and
`hf_pilot_profile.json`. This average across configurations is not the accuracy
of one selected policy. Differences from earlier pilots are not a controlled
ablation; deployment profiles and random draws changed.

The initial 8-question check generated four QA candidates for every model pair
and fused prefixes 1 and 4. Across its 64 pair/budget outcomes, raw paragraph EM
was 0%; short-answer extraction produced EM 46.875% and F1 47.5694%.
This check used the native SGLang repetition-penalty scope. It demonstrates
recoverable answers, not equivalent sampling behavior or research effectiveness.

A second check used two training questions, all four model pairs, 40 QA
candidates, and prefixes 1/10/20/40. All 32 fusion outcomes completed. The longest
actual fusion input had 33,625 tokens, exceeding the checkpoint's 32,768-position
configuration. This check also used native SGLang repetition penalty; its scores
must not be pooled with other profiles.

`serving_checks.json` retains both failed and repaired long-context results.
The old `rope_scaling` override did not activate YaRN under Transformers 5.
After setting `rope_parameters`, both models recovered a code placed near the
start of 60,244 tokens with no input truncation. This is a limited canary; W also
hallucinated unrelated explanation. Extension of the 1.5B checkpoint beyond its
documented 32,768-token context remains experimental.

`hf_repetition_conformance.json` compares the optional replacement processor
against the installed Hugging Face implementation. All 16 positive/negative
logit, repeated-ID, prompt/output-overlap and empty-input checks matched exactly.
This does not establish equivalence of all sampling behavior across backends.
Real W/S retrieval calls and all three pipeline generation stages also completed
using this processor.

Answer extraction is additional postprocessing. It receives only the question
and generated response, with a deterministic parser followed by a fixed greedy
7B reader if needed. It never receives gold answers. Grounding checks require
quoted response evidence; implied yes/no extraction remains a semantic judgment.
Abstentions count as errors. Reader tokens/time are reported separately and must
be included in full comparisons. `extractor_safety_checks.json` retains deliberately
wrong assertions and unanswerable controls. `audited_examples.json` shows three
correct short answers inside longer fusion paragraphs from the final pilot.

The original Base experiment remains preserved separately. Its raw results
cannot be converted into valid adaptive-budget evidence by fixing EM formatting
alone. No repaired full-dataset or test-set results are reported here.

Protocol details and reproduction instructions are in
[`REPAIR.md`](../../experiments/adaptive_sampling/REPAIR.md).
