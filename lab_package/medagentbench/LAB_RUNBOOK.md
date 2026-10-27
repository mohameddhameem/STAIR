# Stage-Level Routing — Lab Runbook & Session Handoff (27 Oct 2026)

Operational state for continuing the SLR/RightRoute research in any new chat.
Read alongside MedAgentBench_Preregistration_DRAFT_22Oct2026.md (the campaign's
decision log) and MedAgentBench_Results_Writeup_27Oct2026.md (the narrative).

## Working rules (Jonathan's standing instructions)
- Jonathan personally validates prompts AND Python line-by-line BEFORE any launch;
  present code for review, never launch directly. Never assume configurations —
  confirm one by one.
- Security: no tokens/credentials in chat or any committed file. Commit only to
  branch `jonathan` of github.com/mohameddhameem/stair; never merge to main.
  Large stores (the 4.1 GB grid, .jsonl episode stores) are NEVER committed.
- Naming: "held-out test" (never "sealed"); "AgentTTS (reproduced)" vs "published";
  "SFT router prompt" (never "stock prompt"); "task type" (each has 30 instances);
  descriptive experiment names, no run codes on slides; native PowerPoint tables;
  data for Jonathan's checking = Excel. Trainings are "Training 1 — risk-free
  training, risk tested as a guardrail at readout", "Training 2 — risk in the
  reward", "Training 3 — the round-2 router".
- File transfers: Claude sends files in chat -> Jonathan saves to
  C:\Documents\Research\slr -> scp to the lab (IP 10.4.0.73, user jonathan).
- Every experiment is pre-registered (design, predictions, kill bar, evaluation
  scope) in the registration log BEFORE it runs; each trained router gets exactly
  ONE held-out test read.

## Lab environment (host hmcs-ada600, 46 GB GPU, shared — exclusive windows for 72B)
- Harness: /mnt/hdd/jonathan/medagentbench (clone of stanfordmlgroup/MedAgentBench
  @ 9926011; refsol.py installed from the authors' README link — untracked by
  design). FHIR server = their Docker image. Task stack: `python -m src.start_task -a`
  (controller + workers; parent Exit 1 is harmless), then
  `python -m src.assigner -c configs/assignments/<name>.yaml`. The assigner RESUMES
  from its output dir — move old outputs aside before reruns.
- Envs: training/tokenizer work -> /mnt/hdd/jonathan/slr-env/bin/python with
  PYTHONPATH=/mnt/hdd/jonathan/pyshim:/mnt/hdd/jonathan/STAIR/common HF_HUB_OFFLINE=1
  (torchao .so load warnings are harmless). vLLM serving -> vllm-env2
  (vLLM 0.19.0 + torch 2.10.0+cu128). mab-env = harness only (transformers too old
  for Qwen2 tokenizers).
- Serve 72B (needs GPU exclusively, ~44.5 GB; kill other servers first):
  HF_HOME=/mnt/hdd/jonathan/hf_cache nohup /mnt/hdd/jonathan/vllm-env2/bin/vllm \
  serve Qwen/Qwen2.5-72B-Instruct-AWQ --max-model-len 11264 \
  --gpu-memory-utilization 0.97 --enforce-eager --port 8000 > <log> 2>&1 &
- Serve 7B (weak pilot): same pattern, Qwen/Qwen2.5-7B-Instruct,
  --gpu-memory-utilization 0.50 --max-model-len 16384, port 8000.
- Scripts live in /mnt/hdd/jonathan/slr/stage-level-routing/lab_package/
  (mirrored on the branch under lab_package/medagentbench/).

## Data & result stores (all under /mnt/hdd/jonathan/runs/mab/ unless noted)
- training1_table.json (replay table, 300 rows; sha 186545f0...), training3_table.json
  (round-2 views + cw2; sha 4eff23f3...)
- ss_costs_per_instance.csv, weak_costs_by_round.csv (per-round weak costs)
- tau_scores.json (v1, superseded) and tau_scores_v2.json (FROZEN; sha 2d971783...)
- training1_router/ , training2_router_lt{0.5,1,2}/ , training3_router/ — each with
  best_adapter, curve.jsonl, val_scores.jsonl, manifest.json, and (T1/T2-lt0.5/T3)
  test_scores.jsonl + test_read_manifest.json (the run-once guards: their existence
  BLOCKS any second test read — by design)
- training{1,2,3}_calibration.json; episode stores under
  medagentbench/outputs/{MedAgentBenchv1, WeakPilot7B_run1, WeakPilot7B, DeepInfraAnchor}
- Splits: STAIR/common/stair_common/splits/medagentbench/{calib,test,calib_sweep}.txt
  (patient-disjoint, seed 20260729)

## Campaign status (26-27 Oct 2026): COMPLETE — no pending experiments
Held-out test frontier (150 instances, each router read once): always-weak 12.00 |
always-strong 54.00 (all-300: 56.67/56.00) | oracle 63.33 | T1 61.33 @ 54.1% (R' 0.3766)
| T1+gate 61.33 @ 81.0% | T2 63.33 @ 94.5% | T3 63.33 @ 81.5% (R' 0.4588).
Verdicts: risk belongs at the decision layer, not in the gradient; early escalation
wins the accuracy corner; hopeless-set delivery is emergent. Partial-credit check
deferred (trigger unmet; mechanical scheme on file).

## Deliverables (current versions)
- Deck: Research_Direction_For_Stage_Level_Routing_V14-137.pptx (results board =
  slide 117; three-environments = 115; prompts = 90; risk design = 70/73/87/88)
- Workbook: MedAgentBench_Campaign_Summary_27Oct2026.xlsx (verified numbers source)
- Write-up: MedAgentBench_Results_Writeup_27Oct2026.md (this folder) + the live
  Claude doc "MedAgentBench Routing Campaign — Results Write-up"

## Open items
- Dhameem's review of branch `jonathan` (Jonathan sends the note; source text drafted)
- FUTURE CHAPTER (scoped, needs own registration before anything runs): mid-episode
  HANDOFF — strong continues from the weak model's partial state; live rollouts,
  cannot be replayed; restart baseline to beat = T3's 63.33 @ 81.5%
- Other declared openings: third benchmark (recipe transfer), clinician review of
  the rubric's fine calibration, seed repeats of headline runs
