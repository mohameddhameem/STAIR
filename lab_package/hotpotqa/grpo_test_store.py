#!/usr/bin/env python3
# grpo_test_store.py -- VERSION: v1.0 2026-10-21
# STORE pass of the GRPO (A1) best adapters over the SEALED TEST split:
# writes per-instance probabilities ONLY. This script never touches outcomes
# (no em_standard, no gold answers) -- it produces scores; the separate
# read-only readout applies the pre-declared frozen thresholds.
# Inputs byte-identical to training/validation (same build as the certified
# trainers). Output file names mirror the certified test_scores convention.
#
# Usage (lab, slr-env):  python3 grpo_test_store.py        (~5 min GPU)
import os, sys, json

os.environ.setdefault("SLR_POOL_K", "100")
os.environ.setdefault("SLR_SHUFFLE", "off")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import run_labels4 as R
import router_prompts as RP
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DATA = "/mnt/hdd/jonathan/data/tts_test.json"
GRID = "/mnt/hdd/jonathan/runs/config_b"
ROWS = os.path.join(GRID, "ret_labels", "ret_labels.jsonl")      # rows + splits only
A1   = os.path.join(GRID, "grpo_ret_qa_a1")
OUT  = os.path.join(GRID, "test_scores")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_store(path):
    d = {}
    for line in open(path):
        r = json.loads(line); d[r["k"]] = r["v"]
    return d


def main():
    data = json.load(open(DATA))
    retr = {"W": load_store(os.path.join(GRID, "retr_weak.jsonl")),
            "S": load_store(os.path.join(GRID, "retr_strong.jsonl"))}
    qa_w = load_store(os.path.join(GRID, "qa_weak.jsonl"))
    rows = []
    for rline in open(ROWS):
        r = json.loads(rline)
        if r["split"] != "test":
            continue
        i = r["idx"]; inst = data[i]
        pool = R.trim_pool(inst)
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
    print(f"test rows: {len(rows)} (expect 1,481)")

    tok = AutoTokenizer.from_pretrained(MODEL)
    idW, idS = RP.letter_ids(tok)

    def load_router(sub):
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                    device_map="cuda")
        return PeftModel.from_pretrained(base, os.path.join(A1, sub)).eval()

    def pW(model, user):
        ids = RP.assemble(tok, user).to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        return float(torch.softmax(torch.stack([out[idW], out[idS]]), 0)[0])

    os.makedirs(OUT, exist_ok=True)
    ret_model = load_router("ret_best_adapter")
    with torch.no_grad(), open(os.path.join(OUT, "ret_grpo_a1.jsonl"), "w") as f:
        for k, r in enumerate(rows):
            f.write(json.dumps({"idx": r["idx"], "pW": round(pW(ret_model, r["ret_user"]), 6)}) + "\n")
            if (k + 1) % 250 == 0:
                print(f"  ret {k+1}/{len(rows)}", flush=True)
    del ret_model; torch.cuda.empty_cache()

    qa_model = load_router("qa_best_adapter")
    with torch.no_grad(), open(os.path.join(OUT, "qa_grpo_a1.jsonl"), "w") as f:
        for k, r in enumerate(rows):
            for rc in ("W", "S"):
                f.write(json.dumps({"idx": r["idx"], "ret_choice": rc,
                                    "pW": round(pW(qa_model, r["qa_user"][rc]), 6)}) + "\n")
            if (k + 1) % 250 == 0:
                print(f"  qa {k+1}/{len(rows)}", flush=True)
    print("wrote test_scores/ret_grpo_a1.jsonl and qa_grpo_a1.jsonl -- scores only, no outcomes touched")


if __name__ == "__main__":
    main()
