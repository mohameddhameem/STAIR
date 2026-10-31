# GPAgentBench-2K prompts — provenance

| File | Source | Status |
|---|---|---|
| `agent_prompt_C1.txt` | GPAgentBench-2K paper (arXiv 2608.30188, EMNLP 2026), Appendix C.1 "Zero-Shot Six-Action Evaluation Prompt" | **verbatim** transcription (31 Oct 2026, full-text read). The paper prints the prompt hard-wrapped in a narrow column; line breaks inside sentences were removed, nothing else changed. Bracketed placeholders in the paper (`[case id]`, `[consultation transcript]`, …) are written here as `{named}` fields for `str.format`; the paper's own example shows the initial `collecting_evidence` state with allowed = ask / body_exam / test / diagnosis / refer and masked = treatment — NOTE the paper's allowed/masked lists spell the two terminal-side names "diagnosis" / "treatment" while the action types are "diagnose" / "treat"; we reproduce the paper's lists as printed. |
| `judge_dx_C2.txt` | Appendix C.2 "Diagnostic Accuracy Evaluation Prompt" | **verbatim** |
| `judge_tx_C2.txt` | Appendix C.2 "Treatment Score Evaluation Prompt" | **verbatim** |
| `patient_simulator_v1.txt` | OURS — the paper does not publish its simulator prompt (GPT-4o-mini) | DRAFT, written from §4.1 (four persona dimensions) and Appendix C.4 (audit criteria: factual consistency with the record, clinical realism, persona adherence); reviewed line by line by Jonathan before any episode |

Not available anywhere (authors' GitHub `jianing-lab/gpagentbench` is an empty placeholder, re-checked 31 Oct 2026 21:09): harness code, simulator prompt, decoding settings. Decoding used here = temperature 0, max_tokens 2048 (agent, simulator), temperature 0 / 16 (judge) — our choice, recorded, not the paper's.
