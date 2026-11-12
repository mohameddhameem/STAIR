#!/usr/bin/env python3
"""
twowiki_pair_test_store.py -- ONE store script for the three pending 2Wiki test600 reads (12 Nov 2026; DRAFT for review).
= twowiki_sft_test_store.py (reviewed, run 11 Nov 15:07, gate passed exactly) with the pair chosen by --pair; lines marked # PAIR:
  --pair reward         SFT (label) + Reward         adapters reward_2w_{ret,qa}/best_adapter          SFT router prompt
  --pair warm-sft       SFT (label) + GRPO (Reward)  adapters grpo_2w_warm_sft_ep3_standard/{ret,qa}_best_adapter   SFT router prompt
  --pair warm-hotpotqa  HotpotQA-warmed GRPO (Reward) adapters grpo_2w_warm_hotpotqa_ep3_objective/{ret,qa}_best_adapter  OBJECTIVE prompt
The prompt is activated to match training (objective prompt only for warm-hotpotqa). Gate: reproduce the pair's own retrieval
val_scores (25 rows, 0 flips at 0.50/0.80, mean|d| < 0.01, max|d| < 0.05) -- a prompt mismatch fails here, before any test row.
Output <pair folder>/test600/{ret,qa}_scores.jsonl + manifest.json; one-run guard. Everything else byte-identical.
"""

import os, sys, json, hashlib, datetime

os.environ["SLR_POOL_K"] = "110"                                                   # T600: as the 2Wiki trainer
os.environ.setdefault("SLR_SHUFFLE", "off")
LAB_PKG = "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package"   # T600: the folder grpo_trainer_v15_2w.py ran from
sys.path.insert(0, LAB_PKG)
import torch
import run_labels4 as R
import router_prompts as RP
import argparse                                                                              # PAIR
_ap = argparse.ArgumentParser(); _ap.add_argument("--pair", choices=["reward", "warm-sft", "warm-hotpotqa"], required=True)
_ap.add_argument("--eval-scores", action="store_true", help="gate against *_val_scores_eval.jsonl and read LOCKED_thresholds_eval.json (12 Nov fix)")
_args = _ap.parse_args(); PAIRCHOICE, EVALS = _args.pair, _args.eval_scores
S5 = "/mnt/hdd/jonathan/runs/twowiki"
PAIRS = {  # name: (ret adapter, qa adapter, ret val_scores file, sweep/lock folder, objective prompt?)
    "reward":        (f"{S5}/reward_2w_ret/best_adapter", f"{S5}/reward_2w_qa/best_adapter",
                      f"{S5}/reward_2w_ret/val_scores.jsonl", f"{S5}/reward_2w_pair", False),
    "warm-sft":      (f"{S5}/grpo_2w_warm_sft_ep3_standard/ret_best_adapter", f"{S5}/grpo_2w_warm_sft_ep3_standard/qa_best_adapter",
                      f"{S5}/grpo_2w_warm_sft_ep3_standard/ret_val_scores.jsonl", f"{S5}/grpo_2w_warm_sft_ep3_standard", False),
    "warm-hotpotqa": (f"{S5}/grpo_2w_warm_hotpotqa_ep3_objective/ret_best_adapter", f"{S5}/grpo_2w_warm_hotpotqa_ep3_objective/qa_best_adapter",
                      f"{S5}/grpo_2w_warm_hotpotqa_ep3_objective/ret_val_scores.jsonl", f"{S5}/grpo_2w_warm_hotpotqa_ep3_objective", True),
}
AD_RET, AD_QA, VALREF, PAIR, OBJ = PAIRS[PAIRCHOICE]                                        # PAIR
if EVALS:                                                                                    # 12 Nov: eval-mode re-dump + its lock
    VALREF = VALREF.replace("ret_val_scores.jsonl", "ret_val_scores_eval.jsonl")
LOCKNAME = "LOCKED_thresholds_eval.json" if EVALS else "LOCKED_thresholds.json"
if OBJ:
    import router_prompts_objective
    router_prompts_objective.activate()  # trainer parity for the objective-prompt pair
    print("objective prompt ACTIVE:", router_prompts_objective.OBJECTIVE_TEMPLATE, flush=True)
else:
    print("SFT router prompt (router_prompts as imported; objective prompt NOT activated)", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

GRID = "/mnt/hdd/jonathan/runs/twowiki/grpo_grid"                                          # T600: staged train+val (gate only)
A1 = {"ret": AD_RET, "qa": AD_QA}                                                            # PAIR: adapter folders
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
CELLS = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged/test600"                        # T600
DATA_2W = os.path.join(CELLS, "tts_test_2wiki_adapted.json")
OUT = os.path.join(PAIR, "test600")                                                        # SFT


def load_store(path):
    d = {}
    for line in open(path):
        r = json.loads(line); d[r["k"]] = r["v"]
    return d


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if os.path.exists(os.path.join(OUT, "manifest.json")):                                 # one declared read
        raise SystemExit(f"REFUSING: {OUT}/manifest.json exists -- the test600 store has already run")
    data = json.load(open(DATA_2W))                                                        # 2W
    assert len(data) == 600, len(data)                                                    # T600
    retr = {"W": load_store(os.path.join(CELLS, "retr_weak.jsonl")),                      # 2W
            "S": load_store(os.path.join(CELLS, "retr_strong.jsonl"))}
    qa_w = load_store(os.path.join(CELLS, "qa_weak.jsonl"))                                # 2W
    rows = []
    for i, inst in enumerate(data):                                                        # all 600, store index = position
        pool = inst["ctxs_candidate"]                                                      # 2W: full pool (cells ran untrimmed)
        sel_w = R.parse_selection(retr["W"][str(i)], len(pool))
        ret_user = RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel_w),
                       selections=RP.format_selections(sel_w, pool))
        qa_user = {}
        for rc, sfx in (("W", "weak"), ("S", "strong")):
            sel = R.parse_selection(retr[rc][str(i)], len(pool))
            qa_user[rc] = RP.INSTR_QA1.format(q=inst["question"],
                              evidence=RP.format_evidence(R.build_block(inst, sel)),
                              weak_answer=str(qa_w[f"{i}|{sfx}"]))
        rows.append({"idx": i, "ret_user": ret_user, "qa_user": qa_user})
    print(f"2Wiki test rows: {len(rows)} (expect 600)")                                    # T600

    tok = AutoTokenizer.from_pretrained(MODEL)
    idW, idS = RP.letter_ids(tok)

    def load_router(sub):
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
        return PeftModel.from_pretrained(base, A1[sub]).eval()                                 # PAIR

    def pW(model, user):
        ids = RP.assemble(tok, user).to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        return float(torch.softmax(torch.stack([out[idW], out[idS]]), 0)[0])

    os.makedirs(OUT, exist_ok=True)
    ret_model = load_router("ret")                                                           # SFT
    # ---- ACCEPTANCE GATE (T600: re-pointed): reproduce the 2Wiki trainer's own val scores ----
    hdata = json.load(open(os.path.join(GRID, "tts_2wiki_trainval.json")))                 # T600
    hretr_w = load_store(os.path.join(GRID, "retr_weak.jsonl"))
    vrows = []
    for rline in open(os.path.join(GRID, "ret_labels", "ret_labels.jsonl")):
        r = json.loads(rline)
        if r["split"] != "val":
            continue
        i = r["idx"]; inst = hdata[i]
        pool = inst["ctxs_candidate"]                                                       # T600: = trim_pool at K=110
        sel_w = R.parse_selection(hretr_w[str(i)], len(pool))
        vrows.append((i, RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel_w),
                          selections=RP.format_selections(sel_w, pool))))
        if len(vrows) == 25:
            break
    ref = {}
    for line in open(VALREF):                                                                 # PAIR: the pair's own val scores
        rr = json.loads(line); ref[rr["idx"]] = rr["pW"]
    with torch.no_grad():
        pairs = [(pW(ret_model, u), ref[i]) for i, u in vrows]
    diffs = [abs(p - r) for p, r in pairs]
    flips = sum(((p >= t) != (r >= t)) for p, r in pairs for t in (0.50, 0.80))
    gate = {"rows": len(diffs), "mean_abs_d": round(sum(diffs) / len(diffs), 6), "max_abs_d": round(max(diffs), 6), "flips": flips}
    print("GATE (2Wiki val reproduction):", gate)                                          # T600
    assert flips == 0 and gate["mean_abs_d"] < 0.01 and gate["max_abs_d"] < 0.05, \
        "GATE FAILED -- the loaded router does not reproduce the trained one; TEST600 NOT SCORED"
    with torch.no_grad(), open(os.path.join(OUT, "ret_scores.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"idx": r["idx"], "pW": round(pW(ret_model, r["ret_user"]), 6)}) + "\n")
    del ret_model; torch.cuda.empty_cache()
    qa_model = load_router("qa")                                                             # SFT
    with torch.no_grad(), open(os.path.join(OUT, "qa_scores.jsonl"), "w") as f:
        for r in rows:
            for rc in ("W", "S"):
                f.write(json.dumps({"idx": r["idx"], "ret_choice": rc, "pW": round(pW(qa_model, r["qa_user"][rc]), 6)}) + "\n")
    man = {"version": "twowiki_pair_test_store v1.0 2026-11-12", "pair": PAIRCHOICE, "router": A1, "base": MODEL,   # PAIR
           "prompt": "objective prompt" if OBJ else "SFT router prompt (objective NOT activated)",
           "gate": gate, "data_sha256": sha(DATA_2W),
           "stores_sha256": {k: sha(os.path.join(CELLS, k)) for k in ("retr_weak.jsonl", "retr_strong.jsonl", "qa_weak.jsonl")},
           "locked": json.load(open(os.path.join(PAIR, LOCKNAME))), "eval_scores": EVALS,   # 12 Nov
           "created": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("wrote ret_scores.jsonl, qa_scores.jsonl, manifest.json -- scores only, no outcomes touched")


if __name__ == "__main__":
    main()
