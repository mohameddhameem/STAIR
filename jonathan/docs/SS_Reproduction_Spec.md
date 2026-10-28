# Always-Strong (SS) Baseline — Reproduction Specification
*Jonathan's Stage-Level Routing, Config B · spec frozen 25 Oct 2026 · for team reproduction (Andy)*

## The number being reproduced
**SS lenient accuracy 76.77 on the held-out test split (n = 1,481)** — and, from the same stored run:
**78.80 on team-val (= AgentTTS's 500)**. Full four-path ladder for cross-checking any partial rerun
(test split, standard ruler): **WW 50.30 · WS 64.01 · SW 61.78 · SS 76.77**.
Per-question reference rows: `results/common/jonathan-ss-hotpotqa-{val,test}.csv` (STAIR branch `jonathan`).

## 1 · Dataset
The AgentTTS HotpotQA release file = the official HotpotQA dev-distractor set, **7,405 questions**, each with its
**100-passage pool**, order as-given (no shuffle). Verified identities (25 Oct, by sorted-ID md5):
`dev[0:500]` = STAIR `splits/hotpotqa/val.txt` = **AgentTTS's published 500**; `dev[500:1000]` = `test.txt`.
NOTE for the team: AgentTTS's published 0.74 therefore lives on team-VAL, not team-test.

## 2 · Split
Ours: 65/15/20 single draw, seed 20260729, drawn once and frozen (15 Oct). Committed as
`splits/hotpotqa/slr-{train,val,test}.txt` (4,814 / 1,110 / 1,481). The 76.77 is on `slr-test.txt`.

## 3 · Policy
Always-strong, two stages, no router, no weak draft, K = 1:
1. **Retrieval**: the strong retriever reads the question + the 100-passage pool and selects references.
2. **Answering**: the strong answerer reads the question + the selected references and produces one short answer.

## 4 · Models and serving
- Retrieval: **Qwen2.5-72B-Instruct** · Answering: **Llama-3.1-70B-Instruct**
  (team pins incl. revisions: `common/stair_common/models.toml`, menu A).
- Generated July 2026 on the SMU cluster; stores copied to the lab (`configb_stores.tgz`).
- dtype fp16 family; pool prompts up to ~14K tokens (measured per call in the meta stores); greedy decoding.
- **UNCONFIRMED (checked 28 Oct): quantization of the two large models in the July cluster run.** The lab holds no
  record of it (the store archive has answers and per-call token counts only); the lab-side scripts load the strong
  models with bitsandbytes 4-bit, the team pin (`models.toml`) is the AWQ copies. A reproduction on the AWQ pins
  with the corrected answering prompt that lands within ±0.5 of 76.77 settles it; a residual gap points here.
  Flagged rather than assumed: a quantization difference moved a
  comparable benchmark by ~5 points in our 24 Oct MedAgentBench audit, so a reproduction must pin it.

## 5 · Decoding
Greedy, temperature 0, K = 1 (deterministic; our repeat check was byte-identical 60/60).

## 6 · Prompts
**CORRECTION (28 Oct 2026, after Andy's rerun found 0.716 and no textual matches):** the two stages differ.
- **Retrieval prompt: AgentTTS's, byte-verbatim** (system line + selection instruction), as in
  `agenttts_archon_hotpotqa_reference.py`; `verify_prompt_bytes.py` is the byte-equality check.
- **Answering prompt: AgentTTS's answer prompt PLUS one added instruction** —
  "Answer with ONLY the exact answer phrase - no explanation, no full sentence." — i.e. `qa_prompt()` in
  `lab_package/run_labels4.py`. Evidence: the July strong answers are ~2 output tokens each
  (`meta_qa_strong.jsonl`, `n_out`), the fingerprint of that instruction; AgentTTS's prompt yields sentence-length
  answers. The July cluster script itself was not preserved; the stores and meta records are the artifact.
  A reproduction must include the added instruction or it will not match the 76.77 (nor the stored answers).
- The earlier wording "executor prompts byte-verbatim" was wrong for the answering stage and is withdrawn.
  Every row on our HotpotQA boards shares this answering prompt, so comparisons among our rows are unaffected;
  comparisons with AgentTTS's published numbers carry it as a stated difference.

## 7 · Scoring
`stair_common.score.lenient` — the certified standard ruler (identical to `lenient_match_agenttts.py`,
confirmed by the author of AgentTTS as their own scoring rule). `em`/`f1` reported alongside per the schema.

## 8 · Costs
Two bases, per `stair_common.cost` and the schema: `units` = measured tokens (from per-call token logs);
`price` = AgentTTS Table-5 prices (SS policy = 814 + 115 = 929 — the golden-set convention the team follows).
Never mix the two in one comparison.

## 9 · Acceptance for a cluster rerun
Reproduce the ladder on `slr-test.txt` within ±0.5 (greedy determinism holds within one serving stack;
larger drift = serving difference — diagnose per-question against the reference CSVs before quoting).
