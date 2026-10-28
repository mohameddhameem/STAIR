#!/usr/bin/env python3
# =============================================================================
# training4_test_read.py — VERSION: v1.0 2026-10-29
# THE SINGLE HELD-OUT TEST READ for "Training 4 — action-based risk penalty"
# (the CONFIRMATION READ registered 28 Oct: with wrong-action flags 0/0 on
# train and val the penalty is identically zero, so the router is predicted to
# reproduce Training 1 at both operating points: 61.33 @ 54.1 % and 60.00 @ 51.2 %).
# DELTA on the validated training2_test_read.py (changes marked # T4):
#   * Training 4 adapter and outdir
#   * BOTH Training 1 operating points declared (theta 0.50 primary, 0.20 cost-lean)
#   * the reported R' is R'4 = R'1 - P*1[tau>=0.8]*1[wrong action] on the deliver
#     branch, with the flags read from the committed 28 Oct exposure replay
#     (the test 30 are in that file; flags are 0 there too, so R'4 == R'1)
# Table, views, costs and tau are Training 1's. Each trained router gets exactly
# one test read; the manifest guard refuses a second run.
#
# Inputs:  runs/mab/training1_table.json      (replay table, 300 rows)
#          runs/mab/training4_router_P19/best_adapter
#          runs/mab/tau_scores_v2.json         (frozen rubric scores)
#          runs/mab/weak_exposure_high_tier.json (28 Oct replay; per-instance flags)
# Outputs: runs/mab/training4_router_P19/test_scores.jsonl   (pDeliver per test row)
#          runs/mab/training4_router_P19/test_read_manifest.json
#
# GPU note: needs the card for the 1.5B router — stop the 72B server first.
# Usage (lab, slr-env):  python3 training4_test_read.py
# =============================================================================
import os, json, hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

VERSION = "training4_test_read v1.0 2026-10-29"                                  # T4
TABLE   = "/mnt/hdd/jonathan/runs/mab/training1_table.json"
TAU     = "/mnt/hdd/jonathan/runs/mab/tau_scores_v2.json"
EXPOSURE = "/mnt/hdd/jonathan/runs/mab/weak_exposure_high_tier.json"            # T4
ADAPTER = "/mnt/hdd/jonathan/runs/mab/training4_router_P19/best_adapter"        # T4
OUTDIR  = "/mnt/hdd/jonathan/runs/mab/training4_router_P19"                     # T4
MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
WEAK_NAME, STRONG_NAME = "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"
LAMBDA  = 0.10
P       = 19.0                                                                  # T4
OPS     = {"primary operating point (theta=0.50)": 0.50,                        # T4
           "cost-lean operating point (theta=0.20)": 0.20}
TAU_GATE = 0.80

# ---- prompt: IDENTICAL to training4_trainer.py v1.0 (same self-check) -------
SYS = ("You route a medical record-system task. The weak model {weak_name} has "
       "already attempted it; choose whether to deliver its result or escalate "
       "to {strong_name}. Your objective: the task should be completed correctly "
       "while spending as little compute as possible - correctness first, cost "
       "second. Based on the information shown below, choose one option."
       ).format(weak_name=WEAK_NAME, strong_name=STRONG_NAME)
USER_TAIL = ("\n\nOptions:\nW) Deliver the weak model's result\n"
             "S) Escalate to the strong model\nAnswer with a single letter: W or S.")
assert ("Your objective: the task should be completed correctly while spending "
        "as little compute as possible - correctness first, cost second.") in SYS
print("[test read prompt] matches the trained router's prompt", flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def hinge(x): return max(0.0, x - 1.0)

guard = os.path.join(OUTDIR, "test_read_manifest.json")
if os.path.exists(guard):
    raise SystemExit(f"REFUSING TO RUN: {guard} already exists — "
                     "the single test read has already happened.")
man = json.load(open(os.path.join(OUTDIR, "manifest.json")))                    # T4
assert man["stage"] == "training4_action_penalty" and not man.get("smoke"), man.get("stage")
assert float(man["P"]) == P, f"manifest P={man['P']}, script P={P}"

table = [t for t in json.load(open(TABLE)) if t["split"] == "test"]
assert len(table) == 150, f"expected 150 test rows, got {len(table)}"
tauj = json.load(open(TAU))
taus = {s["tid"]: s["tau"] for s in tauj["scores"]}
assert all(t["tid"] in taus for t in table), "tau score missing for a test instance"
expo = json.load(open(EXPOSURE))                                                # T4
wrong = {k: (v["category"] == "wrong_order") for k, v in expo["per_instance"].items()}   # T4 same rule as the trainer
assert all(t["tid"] in wrong for t in table if taus[t["tid"]] >= TAU_GATE), "exposure flag missing on a test tier row"   # T4
n_high = sum(taus[t["tid"]] >= TAU_GATE for t in table)
n_flag = sum(wrong.get(t["tid"], False) for t in table if taus[t["tid"]] >= TAU_GATE)   # T4
print(f"test rows: 150 | high-risk tier (tau>={TAU_GATE}): {n_high} | wrong-action flags on tier: {n_flag} | "
      f"tau scorer: {tauj['scorer']}", flush=True)

# ---- the one router pass over test ------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL)
def one_id(s):
    ids = tok.encode(s, add_special_tokens=False); assert len(ids) == 1
    return ids[0]
idW, idS = one_id("W"), one_id("S")
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                            device_map="cuda")
model = PeftModel.from_pretrained(base, ADAPTER).eval()

scores = {}
with torch.no_grad(), open(os.path.join(OUTDIR, "test_scores.jsonl"), "w") as fv:
    for t in table:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": t["view"] + USER_TAIL}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        lp = torch.log_softmax(torch.stack([out[idW], out[idS]]), 0)
        pW = float(lp[0].exp())
        scores[t["tid"]] = pW
        fv.write(json.dumps({"idx": t["idx"], "tid": t["tid"],
                             "pDeliver": round(pW, 6)}) + "\n")
print("router pass over test 150 complete", flush=True)

# ---- tallies (pure lookups from here) ----------------------------------------
def tally(theta, gate):
    """gate=False: tally 1 (without guardrail). gate=True: tau>=0.8 forced S."""
    acc = cost = rsum = rsum1 = nd = 0
    hi_acc = hi_cost = hi_n = 0
    for t in table:
        deliver = scores[t["tid"]] >= theta
        if gate and taus[t["tid"]] >= TAU_GATE:
            deliver = False
        q = t["qw"] if deliver else t["qs"]
        c = t["cw"] if deliver else t["cw"] + t["cs"]
        x = c / t["cs"]
        r1 = q - LAMBDA * x - hinge(x)
        pen = P if (deliver and taus[t["tid"]] >= TAU_GATE and wrong.get(t["tid"], False)) else 0.0   # T4
        rsum1 += r1; rsum += r1 - pen
        acc += q; cost += c; nd += deliver
        if taus[t["tid"]] >= TAU_GATE:
            hi_n += 1; hi_acc += q; hi_cost += c
    n = len(table)
    return {"deliver": f"{nd}/{n}", "acc_pct": round(100 * acc / n, 2),
            "mean_cost": round(cost / n, 1), "mean_Rprime4": round(rsum / n, 4),     # T4
            "mean_Rprime1": round(rsum1 / n, 4),                                     # T4
            "high_tier": {"n": hi_n,
                          "acc_pct": (round(100 * hi_acc / hi_n, 2) if hi_n else None),
                          "mean_cost": (round(hi_cost / hi_n, 1) if hi_n else None)}}

results = {}
for name, th in OPS.items():
    results[name] = {"tally1_without_guardrail": tally(th, gate=False),
                     "tally2_with_guardrail":    tally(th, gate=True)}

# reference anchors on test (same replay lookups)
def anchor(deliver_all):
    acc = cost = rsum = 0
    for t in table:
        q = t["qw"] if deliver_all else t["qs"]
        c = t["cw"] if deliver_all else t["cw"] + t["cs"]
        x = c / t["cs"]; rsum += q - LAMBDA * x - hinge(x); acc += q; cost += c
    return {"acc_pct": round(100 * acc / 150, 2), "mean_cost": round(cost / 150, 1),
            "mean_Rprime": round(rsum / 150, 4)}
oracle_acc = sum(max(t["qw"], t["qs"]) for t in table)
refs = {"always_deliver": anchor(True), "always_escalate": anchor(False),
        "oracle_acc_pct": round(100 * oracle_acc / 150, 2),
        "training1_registered": {"primary": "61.33 @ 54.1 %", "cost_lean": "60.00 @ 51.2 %"}}   # T4

manifest = {"version": VERSION, "adapter": ADAPTER, "P": P,                     # T4
            "table_sha256": sha256(TABLE), "tau_sha256": sha256(TAU),
            "exposure_sha256": sha256(EXPOSURE), "wrong_action_flags_on_test_tier": n_flag,   # T4
            "tau_scorer": tauj["scorer"], "tau_gate": TAU_GATE,
            "declared_operating_points": OPS,
            "router_prompt_sys": SYS,
            "references_on_test": refs, "results": results}
json.dump(manifest, open(guard, "w"), indent=2)
print(json.dumps({"TEST_READ": True, **manifest}, indent=1), flush=True)
