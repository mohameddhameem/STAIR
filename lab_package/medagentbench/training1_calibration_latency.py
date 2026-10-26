#!/usr/bin/env python3
# training1_calibration_latency.py — v1.0 2026-10-26
# The REGISTERED calibration audit + decision latency, for Training 1.
#   * Calibration: reliability of pDeliver vs actual weak success (qw), on the
#     VALIDATION split (60) — before training (base model; a fresh LoRA's B=0
#     start IS the base model) and after (best_adapter). Reports reliability
#     bins, ECE, AUC, and the saturation fraction (pDeliver < 0.05 or > 0.95).
#   * Latency: wall time per routing decision (one forward pass), measured on
#     the same 60 calls — mean / median / p95 in milliseconds.
# Read-only over all stores; writes runs/mab/training1_calibration.json.
# Usage (lab, GPU free): /mnt/hdd/jonathan/slr-env/bin/python training1_calibration_latency.py
import json, time, statistics as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

TABLE   = "/mnt/hdd/jonathan/runs/mab/training1_table.json"
ADAPTER = "/mnt/hdd/jonathan/runs/mab/training1_router/best_adapter"
OUT     = "/mnt/hdd/jonathan/runs/mab/training1_calibration.json"
MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
WEAK_NAME, STRONG_NAME = "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"

# prompt: IDENTICAL to training1_trainer.py v1.0
SYS = ("You route a medical record-system task. The weak model {weak_name} has "
       "already attempted it; choose whether to deliver its result or escalate "
       "to {strong_name}. Your objective: the task should be completed correctly "
       "while spending as little compute as possible - correctness first, cost "
       "second. Based on the information shown below, choose one option."
       ).format(weak_name=WEAK_NAME, strong_name=STRONG_NAME)
USER_TAIL = ("\n\nOptions:\nW) Deliver the weak model's result\n"
             "S) Escalate to the strong model\nAnswer with a single letter: W or S.")

val = [t for t in json.load(open(TABLE)) if t["split"] == "val"]
assert len(val) == 60

tok = AutoTokenizer.from_pretrained(MODEL)
def one_id(s):
    ids = tok.encode(s, add_special_tokens=False); assert len(ids) == 1
    return ids[0]
idW, idS = one_id("W"), one_id("S")

def score_all(model, time_it=False):
    out, times = [], []
    with torch.no_grad():
        for t in val:
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": t["view"] + USER_TAIL}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                          return_tensors="pt")
            if not torch.is_tensor(ids):
                ids = ids["input_ids"]
            ids = ids.to("cuda")
            if time_it: torch.cuda.synchronize(); t0 = time.perf_counter()
            logits = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
            lp = torch.log_softmax(torch.stack([logits[idW], logits[idS]]), 0)
            pW = float(lp[0].exp())
            if time_it:
                torch.cuda.synchronize(); times.append((time.perf_counter() - t0) * 1000)
            out.append({"tid": t["tid"], "p": pW, "qw": t["qw"]})
    return out, times

def calibration(scored):
    n = len(scored)
    bins = [(0.0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.01)]
    table, ece = [], 0.0
    for lo, hi in bins:
        grp = [s for s in scored if lo <= s["p"] < hi]
        if grp:
            mp = st.mean(s["p"] for s in grp); emp = st.mean(s["qw"] for s in grp)
            ece += len(grp)/n * abs(mp - emp)
            table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": len(grp),
                          "mean_pDeliver": round(mp,3), "empirical_weak_success": round(emp,3)})
        else:
            table.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0,
                          "mean_pDeliver": None, "empirical_weak_success": None})
    pos = [s["p"] for s in scored if s["qw"] == 1]
    neg = [s["p"] for s in scored if s["qw"] == 0]
    auc = (sum((p > q) + 0.5*(p == q) for p in pos for q in neg) / (len(pos)*len(neg))
           if pos and neg else None)
    sat = st.mean((s["p"] < 0.05 or s["p"] > 0.95) for s in scored)
    return {"reliability": table, "ECE": round(ece, 4),
            "AUC_pDeliver_vs_weak_success": (round(auc, 4) if auc is not None else None),
            "n_weak_correct": len(pos), "saturation_frac_p_outside_5pct": round(sat, 3)}

print("scoring BEFORE training (base model = fresh router's zero-effect start)...", flush=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda").eval()
before_scores, _ = score_all(base)
before = calibration(before_scores)

print("scoring AFTER training (best_adapter) + timing decisions...", flush=True)
model = PeftModel.from_pretrained(base, ADAPTER).eval()
after_scores, times = score_all(model, time_it=True)
after = calibration(after_scores)
lat = {"mean_ms": round(st.mean(times), 1), "median_ms": round(st.median(times), 1),
       "p95_ms": round(sorted(times)[int(0.95*len(times))-1], 1), "n_timed": len(times),
       "note": "one routing decision = one forward pass of the 1.5B router (model resident; load time excluded)"}

report = {"split": "val (60) — the registered calibration scope", "prompt_matches_trainer": True,
          "before_training": before, "after_training": after, "decision_latency": lat}
json.dump(report, open(OUT, "w"), indent=2)
print(json.dumps(report, indent=1))
