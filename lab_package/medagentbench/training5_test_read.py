#!/usr/bin/env python3
# =============================================================================
# training5_test_read.py — VERSION: v1.0 2026-10-28
# THE SINGLE HELD-OUT TEST READ for one Training 5 router (three-way router:
# deliver / escalate / hand to a person; H available on tau >= 0.8 only).
# DELTA on the validated training2_test_read.py: three-letter softmax on the
# high-risk rows, H's replayed outcome (v1.1: a hand-off counts as correct for
# accuracy-with-H; its reward is 1 iff both models fail, minus the spent weak
# cost), the hand-off metrics (needed / unnecessary / fail with no human), and
# the run directory / theta_H passed as flags (theta_H selected on VALIDATION
# before this script is run).
#
# Run ONCE per router (the manifest guard refuses a second run). Reports:
#   * Point A (primary): the trained greedy policy = argmax over allowed actions
#   * Point B: hand over iff P(H) >= theta_H (declared on validation), else the
#              greedy choice between W and S
#   * References on the same 150: always-deliver, always-escalate, oracle
#     (two-way), and the three-way oracle (a person on every both-fail row).
# All outcomes are replay lookups from the Training 1 table — no new episodes.
# The router's input is the task text + weak attempt only (the "view" field).
#
# Usage (lab, slr-env, GPU free):
#   python3 training5_test_read.py --run-dir /mnt/hdd/jonathan/runs/mab/training5v13_router_lt0 --theta-h 0.50
# =============================================================================
import os, json, hashlib, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

VERSION = "training5_test_read v1.3 2026-10-28 (H = +1 if both fail / -1 if strong succeeds; risk = lambda_t*tau on any AI failure, high tier)"
TABLE   = "/mnt/hdd/jonathan/runs/mab/training1_table.json"
TAU     = "/mnt/hdd/jonathan/runs/mab/tau_scores_v2.json"
MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
WEAK_NAME, STRONG_NAME = "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"
LAMBDA  = 0.10
TAU_GATE = 0.80
Q_H = 1.0

# ---- prompt: IDENTICAL to training5_trainer.py v1.0 (same self-check) -------
SYS = ("You route a medical record-system task. The weak model {weak_name} has "
       "already attempted it; choose whether to deliver its result or escalate "
       "to {strong_name}. Your objective: the task should be completed correctly "
       "while spending as little compute as possible - correctness first, cost "
       "second. Based on the information shown below, choose one option."
       ).format(weak_name=WEAK_NAME, strong_name=STRONG_NAME)
USER_TAIL_2 = ("\n\nOptions:\nW) Deliver the weak model's result\n"
               "S) Escalate to the strong model\nAnswer with a single letter: W or S.")
USER_TAIL_3 = ("\n\nOptions:\nW) Deliver the weak model's result\n"
               "S) Escalate to the strong model\n"
               "H) Hand the task to a person\nAnswer with a single letter: W, S or H.")
assert ("Your objective: the task should be completed correctly while spending "
        "as little compute as possible - correctness first, cost second.") in SYS
assert USER_TAIL_3.count("\n") == USER_TAIL_2.count("\n") + 1
print("[test read prompt] matches the trained router's prompt", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True, help="training5v13b_router_lt<lt> (holds best_adapter)")
ap.add_argument("--theta-h", type=float, required=True, dest="theta_h",
                help="Point B threshold on P(H), chosen on validation before this read")
a = ap.parse_args()
ADAPTER = os.path.join(a.run_dir, "best_adapter")
LT = None  # lambda_t of the trained router, read from its manifest below
OUTDIR = a.run_dir
man = json.load(open(os.path.join(a.run_dir, "manifest.json")))
assert man["stage"] == "training5_three_way_router" and "v1.3" in man["version"], \
    f"run dir manifest: stage={man.get('stage')}, version={man.get('version')}"
assert not man.get("smoke"), "refusing to read a smoke run"
LT = float(man["lambda_t"])

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def hinge(x): return max(0.0, x - 1.0)

guard = os.path.join(OUTDIR, "test_read_manifest.json")
if os.path.exists(guard):
    raise SystemExit(f"REFUSING TO RUN: {guard} already exists — the single test read has already happened.")

table = [t for t in json.load(open(TABLE)) if t["split"] == "test"]
assert len(table) == 150, f"expected 150 test rows, got {len(table)}"
tauj = json.load(open(TAU))
taus = {s["tid"]: s["tau"] for s in tauj["scores"]}
assert all(t["tid"] in taus for t in table)
for t in table:
    t["high"] = taus[t["tid"]] >= TAU_GATE
    t["both_fail"] = (t["qw"] == 0 and t["qs"] == 0)
n_high = sum(t["high"] for t in table)
print(f"test rows: 150 | high-risk tier (tau>={TAU_GATE}): {n_high} | both-fail on tier: "
      f"{sum(t['both_fail'] for t in table if t['high'])} | tau scorer: {tauj['scorer']}", flush=True)

# ---- the one router pass over test ------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL)
def one_id(s):
    ids = tok.encode(s, add_special_tokens=False); assert len(ids) == 1
    return ids[0]
idW, idS, idH = one_id("W"), one_id("S"), one_id("H")
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
model = PeftModel.from_pretrained(base, ADAPTER).eval()

ACT = ["W", "S", "H"]
probs = {}
with torch.no_grad(), open(os.path.join(OUTDIR, "test_scores.jsonl"), "w") as fv:
    for t in table:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": t["view"] + (USER_TAIL_3 if t["high"] else USER_TAIL_2)}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        logits = [out[idW], out[idS]] + ([out[idH]] if t["high"] else [])
        pr = torch.softmax(torch.stack(logits), 0).tolist()
        probs[t["tid"]] = pr
        fv.write(json.dumps({"idx": t["idx"], "tid": t["tid"], "high": t["high"],
                             "pDeliver": round(pr[0], 6), "pEscalate": round(pr[1], 6),
                             "pHuman": (round(pr[2], 6) if t["high"] else None)}) + "\n")
print("router pass over test 150 complete", flush=True)

# ---- decisions and tallies (pure lookups from here) -------------------------
def decide(t, point):
    pr = probs[t["tid"]]
    if point == "A":                                   # trained greedy policy
        return ACT[max(range(len(pr)), key=lambda i: pr[i])]
    if t["high"] and pr[2] >= a.theta_h:               # Point B: threshold on P(H)
        return "H"
    return "W" if pr[0] >= pr[1] else "S"

def outcome(t, c):
    """(correct, ai_cost, human_cost[unused in v1.1], reward) for action c on row t (replay)."""
    xw = t["cw"] / t["cs"]
    risk = lambda q: (LT * taus[t["tid"]] if (t["high"] and q == 0) else 0.0)   # v1.3 risk-weighted failure
    if c == "W":
        return t["qw"], t["cw"], 0.0, t["qw"] - LAMBDA * xw - hinge(xw) - risk(t["qw"])
    if c == "S":
        xs = (t["cw"] + t["cs"]) / t["cs"]
        return t["qs"], t["cw"] + t["cs"], 0.0, t["qs"] - LAMBDA * xs - hinge(xs) - risk(t["qs"])
    return Q_H, t["cw"], 0.0, (1.0 if t["both_fail"] else 0.0) - (1.0 if t["qs"] else 0.0) - LAMBDA * xw

def tally(dec):
    n = len(table); acc = ai_n = ai_acc = 0; ai_cost = hum_cost = rsum = 0.0
    hi = {"n": 0, "deliver": 0, "escalate": 0, "to_person": 0, "H_on_both_fail": 0, "both_fail": 0,
          "correct": 0, "ai_correct": 0, "ai_n": 0, "fail_no_human": 0, "ai_cost": 0.0, "human_cost": 0.0}
    for t in table:
        c = dec(t); q, cai, chum, r = outcome(t, c)
        acc += q; ai_cost += cai; hum_cost += chum; rsum += r
        if c != "H":
            ai_n += 1; ai_acc += q
        if t["high"]:
            hi["n"] += 1; hi[{"W": "deliver", "S": "escalate", "H": "to_person"}[c]] += 1
            hi["both_fail"] += t["both_fail"]; hi["correct"] += q
            hi["ai_cost"] += cai; hi["human_cost"] += chum
            if c == "H":
                hi["H_on_both_fail"] += t["both_fail"]
            else:
                hi["ai_n"] += 1; hi["ai_correct"] += q; hi["fail_no_human"] += (1 - q)
    H = hi["to_person"]
    return {"acc_pct_with_H": round(100 * acc / n, 2),
            "ai_subset_acc_pct": (round(100 * ai_acc / ai_n, 2) if ai_n else None), "ai_handled": ai_n,
            "mean_ai_cost": round(ai_cost / n, 1),
            "mean_Rprime": round(rsum / n, 4),
            "high_tier": {"n": hi["n"], "deliver": hi["deliver"], "escalate": hi["escalate"],
                          "to_person": H, "to_person_needed": hi["H_on_both_fail"],
                          "to_person_unnecessary": H - hi["H_on_both_fail"], "human_load": round(H / hi["n"], 4),
                          "H_precision": (round(hi["H_on_both_fail"] / H, 4) if H else None),
                          "H_recall": round(hi["H_on_both_fail"] / hi["both_fail"], 4),
                          "fail_with_no_human": hi["fail_no_human"],
                          "acc_pct_with_H": round(100 * hi["correct"] / hi["n"], 2),
                          "ai_subset_acc_pct": (round(100 * hi["ai_correct"] / hi["ai_n"], 2) if hi["ai_n"] else None),
                          "mean_ai_cost": round(hi["ai_cost"] / hi["n"], 1)}}

results = {"point_A_greedy": tally(lambda t: decide(t, "A")),
           f"point_B_thetaH_{a.theta_h:g}": tally(lambda t: decide(t, "B"))}

# references on the same 150
refs = {"always_deliver": tally(lambda t: "W"), "always_escalate": tally(lambda t: "S"),
        "oracle_two_way": tally(lambda t: "W" if t["qw"] else "S"),
        "oracle_three_way": tally(lambda t: "W" if t["qw"] else ("S" if t["qs"] else ("H" if t["high"] else "S")))}

manifest = {"version": VERSION, "run_dir": a.run_dir, "adapter": ADAPTER, "H_reward": man.get("H_reward"),
            "lambda_t": man["lambda_t"], "theta_h": a.theta_h, "Q_H": Q_H,
            "table_sha256": sha256(TABLE), "tau_sha256": sha256(TAU), "tau_scorer": tauj["scorer"],
            "tau_gate": TAU_GATE, "router_prompt_sys": SYS, "references_on_test": refs, "results": results}
json.dump(manifest, open(guard, "w"), indent=2)
print(json.dumps({"TEST_READ": True, **manifest}, indent=1), flush=True)
