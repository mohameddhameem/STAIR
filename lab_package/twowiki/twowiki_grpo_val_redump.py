#!/usr/bin/env python3
"""
twowiki_grpo_val_redump.py -- EVAL-MODE re-dump of a GRPO run's val400 router scores. VALIDATION ONLY, read-only on test, ~2 min GPU.
12 Nov 2026 (DRAFT for Jonathan's review). WHY: grpo_trainer v1.5 (and its 2W / 2W-warm copies) reloads the best adapters, calls
evaluate(), and evaluate() ends with model.train() -- so ret/qa_val_scores.jsonl were written with LoRA dropout (0.05) ACTIVE:
a stochastic forward (confirmed 12 Nov 09:46-09:49: store-vs-dump mean|d| 0.012, max 0.19, 12 flips/400 on the warm-SFT pair;
re-score noise in eval mode exactly 0). The stores score in eval mode and are right; the sweeps/locks used the noisy dumps.
This script writes <outdir>/ret_val_scores_eval.jsonl and qa_val_scores_eval.jsonl (deterministic, eval mode, same prompt module
as training: --prompts objective activates the objective prompt) and prints the deviation of the old dump from the new one.
v1.1 (09:55): sets SLR_POOL_K=110 before importing run_labels4 and asserts the pool is untrimmed -- v1.0 omitted this and
scored trimmed/reordered pools (mean|d| 0.32); its outputs were deleted. Original files are NOT touched. Usage: python3 -I twowiki_grpo_val_redump.py --run <outdir> --prompts standard|objective
"""
import argparse, json, os, sys, torch
os.environ["SLR_POOL_K"] = "110"   # BEFORE importing run_labels4 -- exactly as the 2Wiki trainers (pools are 100-104: trim is a no-op)
LAB_PKG = "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package"
sys.path.insert(0, LAB_PKG)
import run_labels4 as R
assert R.POOL_K == 110, R.POOL_K
import router_prompts as RP
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--prompts", choices=["standard", "objective"], required=True)
a = ap.parse_args()
if a.prompts == "objective":
    import router_prompts_objective; router_prompts_objective.activate(); print("objective prompt ACTIVE")
else:
    print("SFT router prompt (standard)")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
GRID, MODEL = "/mnt/hdd/jonathan/runs/twowiki/grpo_grid", "Qwen/Qwen2.5-1.5B-Instruct"
for f in ("ret_val_scores_eval.jsonl", "qa_val_scores_eval.jsonl"):
    if os.path.exists(os.path.join(a.run, f)):
        raise SystemExit(f"REFUSING: {f} exists in {a.run}")


def load_store(path):
    d = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


data = json.load(open(os.path.join(GRID, "tts_2wiki_trainval.json")))
retr = {"W": load_store(os.path.join(GRID, "retr_weak.jsonl")), "S": load_store(os.path.join(GRID, "retr_strong.jsonl"))}
qa_w = load_store(os.path.join(GRID, "qa_weak.jsonl"))
rows = []
for rline in open(os.path.join(GRID, "ret_labels", "ret_labels.jsonl")):
    r = json.loads(rline)
    if r["split"] != "val":
        continue
    i = r["idx"]; inst = data[i]; pool = R.trim_pool(inst)                      # as the trainer's build_rows
    assert pool == inst["ctxs_candidate"], f"pool altered for idx {i} -- SLR_POOL_K not in effect"   # 12 Nov 09:52 guard
    sel_w = R.parse_selection(retr["W"][str(i)], len(pool))
    ret_user = RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel_w), selections=RP.format_selections(sel_w, pool))
    qa_user = {}
    for rc, sfx in (("W", "weak"), ("S", "strong")):
        sel = R.parse_selection(retr[rc][str(i)], len(pool))
        qa_user[rc] = RP.INSTR_QA1.format(q=inst["question"], evidence=RP.format_evidence(R.build_block(inst, sel)),
                                          weak_answer=str(qa_w[f"{i}|{sfx}"]))
    rows.append((i, ret_user, qa_user))
assert len(rows) == 400, len(rows)
tok = AutoTokenizer.from_pretrained(MODEL); idW, idS = RP.letter_ids(tok)


def load(sub):
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
    return PeftModel.from_pretrained(base, os.path.join(a.run, sub)).eval()


def pW(m, user):
    ids = RP.assemble(tok, user).to("cuda")
    out = m(input_ids=ids, logits_to_keep=1).logits[0, -1]
    return float(torch.softmax(torch.stack([out[idW], out[idS]]), 0)[0])


old_ret = {json.loads(l)["idx"]: json.loads(l)["pW"] for l in open(os.path.join(a.run, "ret_val_scores.jsonl"))}
old_qa = {(json.loads(l)["idx"], json.loads(l)["ret_choice"]): json.loads(l)["pW"] for l in open(os.path.join(a.run, "qa_val_scores.jsonl"))}
with torch.no_grad():
    m = load("ret_best_adapter"); d = []
    with open(os.path.join(a.run, "ret_val_scores_eval.jsonl"), "w") as f:
        for i, ru, _ in rows:
            p = pW(m, ru); f.write(json.dumps({"idx": i, "pW": round(p, 6)}) + "\n"); d.append(abs(p - old_ret[i]))
    print(f"RET  old(train-mode) vs new(eval): mean|d| {sum(d)/len(d):.5f}  max|d| {max(d):.5f}  n=400")
    del m; torch.cuda.empty_cache()
    m = load("qa_best_adapter"); d = []
    with open(os.path.join(a.run, "qa_val_scores_eval.jsonl"), "w") as f:
        for i, _, qu in rows:
            for rc in ("W", "S"):
                p = pW(m, qu[rc]); f.write(json.dumps({"idx": i, "ret_choice": rc, "pW": round(p, 6)}) + "\n"); d.append(abs(p - old_qa[(i, rc)]))
    print(f"QA   old(train-mode) vs new(eval): mean|d| {sum(d)/len(d):.5f}  max|d| {max(d):.5f}  n=800")
print("wrote ret_val_scores_eval.jsonl, qa_val_scores_eval.jsonl in", a.run, "-- originals untouched")
