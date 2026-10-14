# Completed Base GGUF prefix-sampling run

Execution, cross-server merging and scoring completed on 2026-10-14 at approximately 17:00 Singapore time. All 7,405 questions have both retrieval outputs and all four model pairs' 40 QA candidates and 40 prefix-fusion outputs. Both servers released their four L40S GPUs after finishing their assignments.

The protocol follows the pinned public AgentTTS HotpotQA full-pipeline prompt methods and temperatures 0.9/0.7/0.3, with 512 new tokens per generation. W and S are community Qwen2.5-1.5B Base and Qwen2.5-7B Base GGUF Q4_K_M. Models, quantization and serving backend differ from the paper. Source code and execution details are in `nonLLM/experiments/adaptive_sampling`.

The unchanged split has 4,814 training, 1,110 validation and 1,481 test questions. Validation EM selected WS with k=6 (cost breaks ties). Its test raw-text exact match is 0%, and its test raw-text F1 is 0.6652%. Across all 160 pair/budget actions, test exact match is also 0%; therefore the retrospective joint EM oracle is 0% and its gain is 0 percentage points.

| Pair | Test raw F1 at k=1 | Test raw F1 at k=40 |
|---|---:|---:|
| WW | 0.9839% | 0.7963% |
| WS | 0.6719% | 0.5101% |
| SW | 1.0398% | 0.7369% |
| SS | 0.6768% | 0.5463% |

These scores use official HotpotQA normalization and EM/F1 on unmodified fusion text. No answer extraction is applied. The upstream `evaluation_2wiki` scorer was absent, so metric identity with the paper is not established. Long outputs can fail exact match even when they contain correct information. Inspected samples also repeated instructions, rewrote the question or invented reference content; this run has a generation validity problem beyond answer formatting. Output differences alone do not identify whether the cause is deployment, tokenizer/input construction, or suitability of the Base models for these prompts. The intended adaptive-budget hypothesis cannot be assessed reliably from this run. These results should not be used as evidence against adaptive sampling or as a meaningful comparison of model capabilities.

`report.json` contains the fixed validation-selected configuration, full test action grid and retrospective oracle statistics. `per_question_oracle.csv` breaks equal exact-match scores by approximate model-weighted token cost. An oracle k=1 on an all-wrong question therefore means a cost tie-break, not that one sample answered correctly.

Canonical raw outputs and final `scores.npz` remain under `/data1/zmcheng/cs707_sampling40/runs` on serverA. A local copy of the final scores is at `/ssd1/zmcheng/cs707/runs/sampling40_helper_sync/scores.npz`. Helper source outputs are under `/data1/zmcheng/cs707_sampling40_helper/runs` on serverB.
