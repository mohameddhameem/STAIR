#!/usr/bin/env python3
# =============================================================================
# training3_test_read.py — VERSION: v1.0 2026-10-26
# THE SINGLE HELD-OUT TEST READ for "Training 3 — the round-2 router."
# DELTA on the validated training1_test_read.py: Training 3 table (round-2
# views), Training 3 adapter, ONE declared operating point (theta = 0.50),
# Training 3 economics (deliver = full cw; escalate = cw2 + cs), and costs
# reported both under round-2 accounting and against the board's full-episode
# always-escalate for cross-row comparability. Each trained router gets exactly
# one test read; Training 1's read stands untouched.
#
# This script is run ONCE. Everything it reports was pre-declared on val:
#   * Operating points:  OP-A theta=0.50 (primary, the trained greedy policy)
#                        OP-B theta=0.20 (cost-match claim)
#   * Two readout tallies of the SAME router pass (no second run):
#       Tally 1 (without guardrail): deliver iff pDeliver >= theta
#       Tally 2 (with guardrail):    same, but tau >= 0.8 -> forced ESCALATE
#   * Primary endpoint: high-risk-tier (tau >= 0.8) accuracy at matched cost.
# All outcomes are replay lookups from the Training 1 table — no new episodes.
#
# Inputs:  runs/mab/training1_table.json      (replay table, 300 rows)
#          runs/mab/training1_router/best_adapter
#          runs/mab/tau_scores.json           (frozen rubric scores)
# Outputs: runs/mab/training1_router/test_scores.jsonl   (pDeliver per test row)
#          runs/mab/training1_router/test_read_manifest.json
#
# GPU note: needs the card for the 1.5B router — stop the 72B server first.
# Usage (lab, slr-env):  python3 training1_test_read.py
# =============================================================================
import os, json, hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

VERSION = "training1_test_read v1.0 2026-10-26"
TABLE   = "/mnt/hdd/jonathan/runs/mab/training3_table.json"
TAU     = "/mnt/hdd/jonathan/runs/mab/tau_scores_v2.json"
ADAPTER = "/mnt/hdd/jonathan/runs/mab/training3_router/best_adapter"
OUTDIR  = "/mnt/hdd/jonathan/runs/mab/training3_router"
MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
WEAK_NAME, STRONG_NAME = "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"
LAMBDA  = 0.10
OPS     = {"OP-A (primary, theta=0.50)": 0.50, "OP-B (cost-match, theta=0.20)": 0.20}
TAU_GATE = 0.80

# ---- prompt: IDENTICAL to training1_trainer.py v1.0 (same self-check) -------
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

table = [t for t in json.load(open(TABLE)) if t["split"] == "test"]
assert len(table) == 150, f"expected 150 test rows, got {len(table)}"
tauj = json.load(open(TAU))
taus = {s["tid"]: s["tau"] for s in tauj["scores"]}
assert all(t["tid"] in taus for t in table), "tau score missing for a test instance"
n_high = sum(taus[t["tid"]] >= TAU_GATE for t in table)
print(f"test rows: 150 | high-risk tier (tau>={TAU_GATE}): {n_high} | "
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
    acc = cost = rsum = nd = 0
    hi_acc = hi_cost = hi_n = 0
    for t in table:
        deliver = scores[t["tid"]] >= theta
        if gate and taus[t["tid"]] >= TAU_GATE:
            deliver = False
        q = t["qw"] if deliver else t["qs"]
        c = t["cw"] if deliver else t["cw2"] + t["cs"]   # round-2 escalation
        x = c / t["cs"]
        rsum += q - LAMBDA * x - hinge(x)
        acc += q; cost += c; nd += deliver
        if taus[t["tid"]] >= TAU_GATE:
            hi_n += 1; hi_acc += q; hi_cost += c
    n = len(table)
    return {"deliver": f"{nd}/{n}", "acc_pct": round(100 * acc / n, 2),
            "mean_cost": round(cost / n, 1), "mean_Rprime": round(rsum / n, 4),
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
        c = t["cw"] if deliver_all else t["cw2"] + t["cs"]   # round-2 escalation
        x = c / t["cs"]; rsum += q - LAMBDA * x - hinge(x); acc += q; cost += c
    return {"acc_pct": round(100 * acc / 150, 2), "mean_cost": round(cost / 150, 1),
            "mean_Rprime": round(rsum / 150, 4)}
oracle_acc = sum(max(t["qw"], t["qs"]) for t in table)
full_ss_mean = round(sum(t["cw"] + t["cs"] for t in table) / 150, 1)
refs = {"always_deliver": anchor(True),
        "always_escalate_round2": anchor(False),
        "always_escalate_full_episode_mean_cost": full_ss_mean,
        "oracle_acc_pct": round(100 * oracle_acc / 150, 2)}

manifest = {"version": VERSION, "adapter": ADAPTER,
            "table_sha256": sha256(TABLE), "tau_sha256": sha256(TAU),
            "tau_scorer": tauj["scorer"], "tau_gate": TAU_GATE,
            "declared_operating_points": OPS,
            "router_prompt_sys": SYS,
            "references_on_test": refs, "results": results}
json.dump(manifest, open(guard, "w"), indent=2)
print(json.dumps({"TEST_READ": True, **manifest}, indent=1), flush=True)
