#!/usr/bin/env python3
"""
twowiki_warm_gate_diag.py -- WHY did the warm-SFT store fail its acceptance gate? VALIDATION ONLY (val400), read-only, no test
contact, ~2 min GPU. DRAFT 12 Nov 2026 for Jonathan's review.
For the pair given by --pair (warm-sft | warm-hotpotqa) it scores ALL 400 val rows of the RETRIEVAL router with
  (a) <outdir>/ret_best_adapter   (what the store loads)      (b) <outdir>/ret_last_adapter
and compares each to the trainer's own <outdir>/ret_val_scores.jsonl (written after reload-best). Also re-scores 5 rows twice
to show run-to-run noise. Prints mean|d|, max|d|, flips at 0.50/0.80, and the 5 worst rows. Reads nothing from test600.
Row construction, prompt module and scoring are byte-identical to twowiki_pair_test_store.py (which is byte-identical to the
reviewed stores). Usage (lab, slr-env, from the lab_package folder):  python3 -I twowiki_warm_gate_diag.py --pair warm-sft
"""
import argparse, json, os, sys, torch
LAB_PKG = "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package"
sys.path.insert(0, LAB_PKG)
import run_labels4 as R
import router_prompts as RP
ap = argparse.ArgumentParser(); ap.add_argument("--pair", choices=["warm-sft", "warm-hotpotqa"], required=True)
a = ap.parse_args()
S5 = "/mnt/hdd/jonathan/runs/twowiki"
OUTDIR = {"warm-sft": f"{S5}/grpo_2w_warm_sft_ep3_standard", "warm-hotpotqa": f"{S5}/grpo_2w_warm_hotpotqa_ep3_objective"}[a.pair]
if a.pair == "warm-hotpotqa":
    import router_prompts_objective; router_prompts_objective.activate(); print("objective prompt ACTIVE")
else:
    print("SFT router prompt (standard)")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
GRID, MODEL = f"{S5}/grpo_grid", "Qwen/Qwen2.5-1.5B-Instruct"


def load_store(path):
    d = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


hdata = json.load(open(os.path.join(GRID, "tts_2wiki_trainval.json")))
hretr_w = load_store(os.path.join(GRID, "retr_weak.jsonl"))
vrows = []
for rline in open(os.path.join(GRID, "ret_labels", "ret_labels.jsonl")):
    r = json.loads(rline)
    if r["split"] != "val":
        continue
    i = r["idx"]; inst = hdata[i]; pool = inst["ctxs_candidate"]
    sel_w = R.parse_selection(hretr_w[str(i)], len(pool))
    vrows.append((i, RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel_w), selections=RP.format_selections(sel_w, pool))))
assert len(vrows) == 400, len(vrows)
ref = {}
for line in open(os.path.join(OUTDIR, "ret_val_scores.jsonl")):
    r = json.loads(line); ref[r["idx"]] = r["pW"]
tok = AutoTokenizer.from_pretrained(MODEL); idW, idS = RP.letter_ids(tok)


def scorer(sub):
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
    m = PeftModel.from_pretrained(base, os.path.join(OUTDIR, sub)).eval()
    def pW(user):
        ids = RP.assemble(tok, user).to("cuda")
        out = m(input_ids=ids, logits_to_keep=1).logits[0, -1]
        return float(torch.softmax(torch.stack([out[idW], out[idS]]), 0)[0])
    return pW


def report(name, pW):
    got = {}
    with torch.no_grad():
        for i, user in vrows:
            got[i] = pW(user)
    flips = sum(((got[i] >= t) != (ref[i] >= t)) for i, _ in vrows for t in (0.5, 0.8))
    dd = [abs(got[i] - ref[i]) for i, _ in vrows]
    print(f"{name:18s} n=400  mean|d| {sum(dd)/400:.5f}  max|d| {max(dd):.5f}  flips(0.50,0.80) {flips}")
    worst = sorted(((abs(got[i] - ref[i]), i, ref[i], got[i]) for i, _ in vrows), reverse=True)[:5]
    for w in worst:
        print(f"    idx {w[1]}  trainer {w[2]:.4f}  now {w[3]:.4f}  |d| {w[0]:.4f}")
    rep = [abs(pW(vrows[k][1]) - got[vrows[k][0]]) for k in range(5)]
    print(f"    re-score noise on 5 rows (same adapter, same process): max {max(rep):.6f}")


print("pair:", a.pair, "| reference:", os.path.join(OUTDIR, "ret_val_scores.jsonl"))
report("ret_BEST_adapter", scorer("ret_best_adapter"))
report("ret_LAST_adapter", scorer("ret_last_adapter"))
