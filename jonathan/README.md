# jonathan: Stage-Level Routing (SLR), lab code and results

Stage-Level Routing on HotpotQA: each question runs retrieval, then answering, and a learned router at each stage
decides whether the weak model's output is enough or the strong model must run (weak-first cascade). Pulled from the
lab box on 19 Oct 2026.

| Config | Retrieval weak / strong | Answering weak / strong | Router |
|---|---|---|---|
| A | Qwen2.5-1.5B / Qwen2.5-7B (fp16) | Qwen2.5-1.5B / Qwen2.5-7B | Qwen2.5-1.5B + LoRA |
| B | Qwen2.5-7B-AWQ / Qwen2.5-72B-AWQ | Llama-3.2-3B / Llama-3.1-70B-AWQ | Qwen2.5-1.5B + LoRA |

## Layout

- `lab_package/`: the lab code as it ran, including exploratory scripts. Paths point at the lab box
  (`/mnt/hdd/jonathan/...`). `lab_package/README.md` is the reading guide to the main chain (`run_labels4.py` ->
  label builders -> `sft_router5.py` -> `gate_eval.py` -> `compose_eval.py`); `configa_*` / `configb_*` are the
  sealed-test scorers and readouts.
- `runs/`: results, mirroring the lab's run directory.
  - `full_run_p100/`: Config A grid. Every model call's output (`retr_*`, `qa_*`) and token counts (`meta_*`).
  - `config_b/`: Config B grid stores, the standard-ruler grid (`router_data_standard.jsonl`), labels, router
    training curves and val scores, ablations, and the sealed-test router scores (`test_scores/`).
  - `ret_labels/`, `qa_labels/`, `sft_router5_*`, `gate_eval*`, `token_costs/`, `compose_eval/`, `joint_train_j/`:
    Config A labels, router runs, threshold sweeps and costs.
- `env_freeze.txt`: the lab Python environment (Python 3.10, torch 2.11, transformers 5.14.1, peft 0.19.1).

## Certified sealed-test numbers (n = 1,481, standard lenient ruler)

| Config | Policy | Accuracy | Cost |
|---|---|---|---|
| B | SFT (label) | 76.70 | 835 |
| B | SFT (label) + reward | 75.96 | 665 |
| B | always-strong | 76.77 | 929 |
| A | SFT (label) | 59.55 | 182.3 |
| A | SFT (label) + reward | 58.81 | 163.3 |
| A | always-strong | 59.55 | 250.1 |

Costs are not on the team's AgentTTS-unit scale: B uses AgentTTS Table 5 per-pass prices (`runs/config_b/token_costs`),
and A uses the measured token-aware family normalised to its cheapest call (`runs/token_costs`).

Reproduced from these files:
- **Config B:** all three rows. Run `configb_test_readout.py` on `runs/config_b/router_data_standard.jsonl` and
  `runs/config_b/test_scores/`: 76.70 at 834.8, 75.96 at 664.7, 76.77 at 929.
- **Config A always-strong:** 59.55 at 250.06, from the standard-ruler grid.
- **Config A router rows: not reproducible from this folder.** `configa_test_readout.py` needs
  `runs/test_scores/*.jsonl` and `runs/reward_train_qa4/val_scores.jsonl`, which are still on the lab box.
- `runs/compose_eval/test_result.json` (50.10 at 106.95) is from an earlier scoring era (grid lenient rule, CPT-95
  lock), superseded by the rows above.

## Not in the repo (size or type)

- `tts_test.json` (422 MB): the 7,405 HotpotQA dev questions with their 100-passage pools, in AgentTTS's order.
  Positions 0–500 / 500–1000 / 1000– equal the team's `val` / `test` / `train_extra` splits exactly.
- The `router_data_*.jsonl` grid views of Config A (about 437 MB each). Their verdicts can be recomputed from
  `runs/full_run_p100/qa_*.jsonl` and `tts_test.json` with the standard ruler; that was checked on 19 Oct and all
  7,405 questions came out identical.
- `*.npz` feature dumps, `*.log` files (ignored by the repo), `__pycache__`, `.bak` copies, and router adapters.
