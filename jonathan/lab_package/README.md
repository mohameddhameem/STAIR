# SLR Code Package — the experiment chain (Qwen 1.5B / 7B, Config A)

Stage-Level Routing (SLR) research, SMU EngD — Dohyung (Jonathan) Ahn, 8 Oct 2026.
These scripts build and evaluate a learned two-stage router on HotpotQA: each
instance passes through retrieval then answering, and at each stage a router
decides whether the weak model (Qwen2.5-1.5B-Instruct) suffices or the strong
model (Qwen2.5-7B-Instruct) must be used. Companion package: the label bundle
(`slr_labels_configA.tar.gz`), whose README defines the shared vocabulary
(retrieval choice W/S, the four cells WW/WS/SW/SS, pool, lenient EM, splits).

## The chain, in run order

Data generation and labels (run once; outputs are in the label bundle):

| Script | Role |
|---|---|
| `run_labels4.py` | THE GRID RUNNER: executes all four cells (weak/strong retrieval x weak/strong answering) on every instance, greedy decoding, and caches raw outputs + per-call token counts in append-only, resume-safe stores. Everything downstream reads this cache; no model is ever re-run |
| `build_ret_labels.py` | Converts grid outcomes into retrieval-stage labels (W/S/X) |
| `build_qa_labels.py` | Converts grid outcomes into answering-stage labels, two rows per instance (one per retrieval choice), honest label + training target |
| `make_label_review.py` | Emits the human-readable review CSVs |
| `check_router_budget.py` | Measured the input budget: 600-character passage cap keeps the gold answer visible for 97.7% of instances; call budget 16,384 tokens, zero truncation |
| `probe_ret_coverage.py` | Learnability probe: can an untrained judge see the retrieval labels at all? (AUC 0.766 — the floor the trained router must beat) |
| `chooser_ceiling.py` | Measured and RETIRED an alternative label scheme (comparing the two models' outputs): upside 3.4% of instances vs a 4x downside. Kept as the audit trail for why labels are outcome-based |
| `zeroshot_judge.py` | Untrained-judge baselines (0.477 / 0.568) that training must beat |

Router training and evaluation:

| Script | Role |
|---|---|
| `router_prompts.py` | THE PROMPT CONTRACT: the frozen router prompts and the single assembly function. Every training and evaluation script imports from here, so prompts are byte-identical across the chain by construction |
| `sft_router5.py` | The one trainer, two stages (`--stage ret` / `--stage qa`): LoRA on Qwen2.5-1.5B, class-weighted, checkpoints chosen by validation AUC; refuses to start if the prompt contract has drifted. Results: retrieval router AUC 0.771, answering router AUC 0.707 |
| `token_costs.py` | Measures the cost ruler from the grid's recorded token counts: six prices (weak/strong retrieval; answering priced conditionally on which retrieval produced its evidence). No hand-typed costs anywhere downstream |
| `gate_eval.py` | Threshold selection on VALIDATION only (CPU): sweeps every distinct threshold pair, builds the accuracy-cost frontier, and locks the operating point by a rule signed before the sweep ran (cheapest pair at >= 95% of strong-strong validation accuracy). Refuses to run without the measured costs |
| `compose_eval.py` | Evaluation of the runtime (by replay): routing decisions are made live by the trained routers; each decision's outcome and cost are looked up in the deterministic grid cache. `--mode val` cross-checks the locked rows via an independent code path; `--mode test` is the ONE sealed-test read, behind three pre-declared guards (lock must exist; locked validation accuracy must clear the bar; an existing result file blocks any re-read). Sealed-test result: 50.10 accuracy at 43% of the always-strong cost |
| `validate_compose.py` | The corner proof: forces the thresholds to +/- infinity so the composition must reproduce each fixed cell's accuracy and cost exactly (4/4 on all 1,110 validation instances) |
| `guard_test.py` | Sandboxed test of the sealed-test guards (5/5 refusals/pass-through correct); never touches real run directories |

## Conventions the chain enforces (in code, not prose)

1. **One prompt source.** Training, judging, and test-time scoring all import
   `router_prompts.py`; `sft_router5.py` runs a drift check at startup and
   exits if the contract changed.
2. **Pinned data order.** Pool size 100, shuffling off — pinned via
   environment defaults inside the scripts (a pool-order bug caught in review
   is the reason).
3. **Deterministic replay.** The grid was generated with greedy decoding, so
   cached outcomes equal live outcomes; evaluation replays the cache instead
   of re-running models.
4. **Measured costs only.** `gate_eval.py` and `compose_eval.py` read prices
   from `token_costs.py`'s output and refuse to run without it.
5. **Sealed test discipline.** Thresholds are locked on validation under a
   pre-registered rule; the test split is read once, and the result file's
   existence blocks any second read.
6. **Review before run.** Every script was reviewed line-by-line before its
   first run; the empirical checks above (probe, corner proof, guard test,
   independent cross-checks) are part of the package so a reader can re-run
   the verification, not just trust it.

## Environment

Python 3.10 venv with torch, transformers, peft, numpy. GPU needed only for
`run_labels4.py`, `sft_router5.py`, and `compose_eval.py --mode test`;
everything else is CPU. Paths at the top of each script point at the lab's
run directories — adjust for your machine. Reading order for a new reader:
this README -> `router_prompts.py` -> `run_labels4.py` -> the label builders
-> `sft_router5.py` -> `gate_eval.py` -> `compose_eval.py`.
