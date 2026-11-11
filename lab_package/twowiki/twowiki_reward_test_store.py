#!/usr/bin/env python3
"""
twowiki_reward_test_store.py -- STORE pass of the 2Wiki SFT (label) + Reward routers on the held-out test600.
REWARD COPY (11 Nov 2026) of twowiki_sft_test_store.py: adapters reward_2w_{ret,qa}/best_adapter; gate reference reward_2w_ret/val_scores.jsonl;
lock from reward_2w_pair; output reward_2w_pair/test600. Same SFT router prompt (the reward pass keeps it). Lines marked # RW differ.
SFT COPY (11 Nov 2026) of twowiki_fresh_test_store.py (the reviewed fresh-adapter store, run 10 Nov); lines marked # SFT differ:
  adapters = sft_2w_router5_{ret,qa}/best_adapter; NO objective-prompt activation (the SFT adapters were trained on the SFT
  router prompt, router_prompts as imported); gate reference = sft_2w_router5_ret/val_scores.jsonl; lock from sft_2w_pair;
  output sft_2w_pair/test600. Everything else byte-identical.
Original header follows.
Writes per-instance router probabilities ONLY -- never reads an outcome (no answers, no scores). DRAFT v1.0 2026-11-10
for Jonathan's line-by-line review.

COPY of twowiki_zeroshot_store.py (the reviewed zero-shot store, 1 Nov); lines that differ are marked # T600:
  * router = the 2Wiki fresh run's best adapters (grpo_2w_a1_fresh_ep6_objective), not the HotpotQA champion
  * data/stores = labels6000_merged/test600 (600 rows), not their 100
  * imports from the folder the 2Wiki trainer ran from (byte-identical prompts/glue to training)
  * ACCEPTANCE GATE re-pointed: before any test row is scored, the store must reproduce the trainer's OWN val scores
    (ret_val_scores.jsonl, 25 staged val rows, same thresholds as before: 0 flips at 0.50 / 0.80, mean|d| < 0.01,
    max|d| < 0.05) -- this proves the loaded router, adapters and prompt are the trained ones
UNCHANGED: full candidate pool (labels ran untrimmed), INSTR_RET2 / INSTR_QA1 / format_* / assemble with default model
names, router_prompts_objective.activate(), the W/S logit-pair reader, scores-only output, one-run guard.
Output: /mnt/hdd/jonathan/runs/twowiki/grpo_2w_fresh_test600/{ret_scores.jsonl, qa_scores.jsonl, manifest.json}
Usage (lab, slr-env, a few minutes GPU):  python3 twowiki_fresh_test_store.py
"""
import os, sys, json, hashlib, datetime

os.environ["SLR_POOL_K"] = "110"                                                   # T600: as the 2Wiki trainer
os.environ.setdefault("SLR_SHUFFLE", "off")
LAB_PKG = "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package"   # T600: the folder grpo_trainer_v15_2w.py ran from
sys.path.insert(0, LAB_PKG)
import torch
import run_labels4 as R
import router_prompts as RP
print("SFT router prompt (router_prompts as imported; objective prompt NOT activated)", flush=True)   # SFT
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

GRID = "/mnt/hdd/jonathan/runs/twowiki/grpo_grid"                                          # T600: staged train+val (gate only)
S5 = "/mnt/hdd/jonathan/runs/twowiki"                                                      # SFT: the two SFT (label) runs
A1 = {"ret": f"{S5}/reward_2w_ret", "qa": f"{S5}/reward_2w_qa"}                            # RW
PAIR = f"{S5}/reward_2w_pair"                                                              # RW: sweep + lock folder
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
        return PeftModel.from_pretrained(base, os.path.join(A1[sub], "best_adapter")).eval()   # SFT

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
    for line in open(os.path.join(A1["ret"], "val_scores.jsonl")):                           # SFT
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
    man = {"version": "twowiki_reward_test_store v1.0 2026-11-11", "router": A1, "base": MODEL,        # SFT
           "prompt": "SFT router prompt (router_prompts, objective NOT activated), as in SFT training",
           "gate": gate, "data_sha256": sha(DATA_2W),
           "stores_sha256": {k: sha(os.path.join(CELLS, k)) for k in ("retr_weak.jsonl", "retr_strong.jsonl", "qa_weak.jsonl")},
           "locked": json.load(open(os.path.join(PAIR, "LOCKED_thresholds.json"))),          # SFT
           "created": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("wrote ret_scores.jsonl, qa_scores.jsonl, manifest.json -- scores only, no outcomes touched")


if __name__ == "__main__":
    main()
