#!/usr/bin/env python3
"""
twowiki_zeroshot_store.py -- STORE pass of the HotpotQA label-free champion router, applied ZERO-SHOT (no retraining)
to AgentTTS's 2Wiki 100, over the four-cell stores of job 346550. Writes per-instance router probabilities ONLY --
never reads an outcome (no answers, no scores). DRAFT v1.0 2026-11-01 for Jonathan's line-by-line review.

DELTA on lab_package/hotpotqa/grpo_fresh_test_store.py (the champion's held-out store, 28 Oct) -- marked # 2W:
  * data  = the adapted 2Wiki 100 (twowiki_cells_adapt.py output), not HotpotQA's tts_test.json
  * stores = retr_weak / retr_strong / qa_weak of the 2Wiki cells run (copied from the cluster to the lab)
  * pool  = the instance's full candidate list (the cells ran with SLR_POOL_K=110 = no trimming; HotpotQA's
            store called trim_pool at POOL_K=100, which on 2Wiki would cut pools of 101-104)
  * rows  = all 100 instances (no split file: the 100 are one declared read)
UNCHANGED, byte for byte: router base (Qwen2.5-1.5B-Instruct), the two best adapters of
grpo_ret_qa_a1_fresh_ep6_objective, router_prompts (INSTR_RET2 / INSTR_QA1 / format_* / assemble with the DEFAULT model
names, exactly as the champion's store and trainer called it), router_prompts_objective.activate() (the reward router
prompt), parse_selection / build_block from run_labels4, the W/S logit-pair reader, and the ACCEPTANCE GATE: before
any 2Wiki row is scored, the store must reproduce the trainer's own HotpotQA validation scores (25 rows, 0 decision
flips at 0.50 and 0.80, mean|d| < 0.01, max|d| < 0.05) -- this proves the router, adapters and prompt are the champion.
Known wording carried over verbatim (not edited, to keep the prompt byte-identical): INSTR_RET2 says "100-passage pool"
and "of 100 passages"; 2Wiki pools are 100-104 passages.

Inputs (lab):  /mnt/hdd/jonathan/runs/twowiki/cells_their100/{tts_test_2wiki_adapted.json, retr_weak.jsonl,
               retr_strong.jsonl, qa_weak.jsonl}   (scp from the cluster run dir)
               /mnt/hdd/jonathan/runs/config_b/{grpo_ret_qa_a1_fresh_ep6_objective/, ret_labels/ret_labels.jsonl,
               retr_weak.jsonl, retr_strong.jsonl}, /mnt/hdd/jonathan/data/tts_test.json   (HotpotQA, for the gate only)
Output:        /mnt/hdd/jonathan/runs/twowiki/zeroshot_their100/{ret_scores.jsonl, qa_scores.jsonl, manifest.json}
Usage (lab, slr-env, ~2 min GPU, 1.5B fits beside other users' jobs):  python3 twowiki_zeroshot_store.py
"""
import os, sys, json, hashlib, datetime

os.environ.setdefault("SLR_POOL_K", "100")
os.environ.setdefault("SLR_SHUFFLE", "off")
LAB_PKG = "/mnt/hdd/jonathan/STAIR/jonathan/lab_package"            # run_labels4, router_prompts (champion's copies)
HOTPOTQA_PKG = "/mnt/hdd/jonathan/STAIR/lab_package/hotpotqa"       # router_prompts_objective
sys.path.insert(0, HOTPOTQA_PKG); sys.path.insert(0, LAB_PKG)
import torch
import run_labels4 as R
import router_prompts as RP
import router_prompts_objective
router_prompts_objective.activate()  # trainer v1.5 parity: patches RP.SYS_CASCADE_TEMPLATE
print("objective prompt ACTIVE:", router_prompts_objective.OBJECTIVE_TEMPLATE, flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---- champion (unchanged) ----
GRID = "/mnt/hdd/jonathan/runs/config_b"
A1 = os.path.join(GRID, "grpo_ret_qa_a1_fresh_ep6_objective")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# ---- 2Wiki (# 2W) ----
CELLS = "/mnt/hdd/jonathan/runs/twowiki/cells_their100"                                   # 2W
DATA_2W = os.path.join(CELLS, "tts_test_2wiki_adapted.json")                               # 2W
OUT = "/mnt/hdd/jonathan/runs/twowiki/zeroshot_their100"                                   # 2W


def load_store(path):
    d = {}
    for line in open(path):
        r = json.loads(line); d[r["k"]] = r["v"]
    return d


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if os.path.exists(os.path.join(OUT, "manifest.json")):                                 # one declared read
        raise SystemExit(f"REFUSING: {OUT}/manifest.json exists -- the zero-shot store has already run")
    data = json.load(open(DATA_2W))                                                        # 2W
    assert len(data) == 100, len(data)
    retr = {"W": load_store(os.path.join(CELLS, "retr_weak.jsonl")),                      # 2W
            "S": load_store(os.path.join(CELLS, "retr_strong.jsonl"))}
    qa_w = load_store(os.path.join(CELLS, "qa_weak.jsonl"))                                # 2W
    rows = []
    for i, inst in enumerate(data):                                                        # 2W: all 100, store index = position
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
    print(f"2Wiki rows: {len(rows)} (expect 100)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    idW, idS = RP.letter_ids(tok)

    def load_router(sub):
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
        return PeftModel.from_pretrained(base, os.path.join(A1, sub)).eval()

    def pW(model, user):
        ids = RP.assemble(tok, user).to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        return float(torch.softmax(torch.stack([out[idW], out[idS]]), 0)[0])

    os.makedirs(OUT, exist_ok=True)
    ret_model = load_router("ret_best_adapter")
    # ---- ACCEPTANCE GATE (unchanged from the champion's store): reproduce the trainer's own HotpotQA val scores ----
    hdata = json.load(open("/mnt/hdd/jonathan/data/tts_test.json"))
    hretr_w = load_store(os.path.join(GRID, "retr_weak.jsonl"))
    vrows = []
    for rline in open(os.path.join(GRID, "ret_labels", "ret_labels.jsonl")):
        r = json.loads(rline)
        if r["split"] != "val":
            continue
        i = r["idx"]; inst = hdata[i]
        pool = R.trim_pool(inst)
        sel_w = R.parse_selection(hretr_w[str(i)], len(pool))
        vrows.append((i, RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel_w),
                          selections=RP.format_selections(sel_w, pool))))
        if len(vrows) == 25:
            break
    ref = {}
    for line in open(os.path.join(A1, "ret_val_scores.jsonl")):
        rr = json.loads(line); ref[rr["idx"]] = rr["pW"]
    with torch.no_grad():
        pairs = [(pW(ret_model, u), ref[i]) for i, u in vrows]
    diffs = [abs(p - r) for p, r in pairs]
    flips = sum(((p >= t) != (r >= t)) for p, r in pairs for t in (0.50, 0.80))
    gate = {"rows": len(diffs), "mean_abs_d": round(sum(diffs) / len(diffs), 6), "max_abs_d": round(max(diffs), 6), "flips": flips}
    print("GATE (HotpotQA val reproduction):", gate)
    assert flips == 0 and gate["mean_abs_d"] < 0.01 and gate["max_abs_d"] < 0.05, \
        "GATE FAILED -- the loaded router does not reproduce the champion; 2WIKI NOT SCORED"
    with torch.no_grad(), open(os.path.join(OUT, "ret_scores.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"idx": r["idx"], "pW": round(pW(ret_model, r["ret_user"]), 6)}) + "\n")
    del ret_model; torch.cuda.empty_cache()
    qa_model = load_router("qa_best_adapter")
    with torch.no_grad(), open(os.path.join(OUT, "qa_scores.jsonl"), "w") as f:
        for r in rows:
            for rc in ("W", "S"):
                f.write(json.dumps({"idx": r["idx"], "ret_choice": rc, "pW": round(pW(qa_model, r["qa_user"][rc]), 6)}) + "\n")
    man = {"version": "twowiki_zeroshot_store v1.0 2026-11-01", "router": A1, "base": MODEL,
           "prompt": "router_prompts_objective (reward router prompt), default model names as in the champion's store",
           "gate": gate, "data_sha256": sha(DATA_2W),
           "stores_sha256": {k: sha(os.path.join(CELLS, k)) for k in ("retr_weak.jsonl", "retr_strong.jsonl", "qa_weak.jsonl")},
           "frozen_points": {"greedy": [0.50, 0.50, 0.50], "cap<=70%": [0.80, 0.05, 0.05]},
           "created": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("wrote ret_scores.jsonl, qa_scores.jsonl, manifest.json -- scores only, no outcomes touched")


if __name__ == "__main__":
    main()
