# training3_build_table.py — "Training 3 — the round-2 router" data preparation.
# DELTA on training1_build_table.py (declared 26 Oct, Step A headroom 5.7pp):
#   * The router's VIEW is cut at ROUND 2 of the weak attempt: episodes that end
#     within 2 rounds show the Training 1 summary (finished / stopped); longer
#     ones show only "still working after 2 rounds" + the round-2 action. The
#     grader's verdict appears nowhere, as before.
#   * NEW cost field cw2 = cumulative weak units after round min(2, rounds).
#     Deliver still pays FULL cw (delivering requires letting weak finish);
#     escalate now pays cw2 + cs (the early-escalation saving).
# Original header follows.
# Builds the replay table from the CANONICAL RECORDS (declared here):
#   weak  = outputs/WeakPilot7B_run1   (7B,  26 Oct, noise floor +/-0.33 pts)
#   strong= outputs/MedAgentBenchv1    (72B, 24 Oct run 2, noise floor +/-0.67 pts)
# Output: /mnt/hdd/jonathan/runs/mab/training1_table.json  (one row per instance)
# Also prints: always-weak / always-strong reward anchors per split, and the
# weak-first-accounting band (deliver = C_w; escalate = C_w + C_s).
# Read-only over run artifacts; writes only the table. No GPU.
import json, collections, statistics as st
from transformers import AutoTokenizer
from stair_common import cost as C

MAB   = "/mnt/hdd/jonathan/medagentbench"
WEAK  = f"{MAB}/outputs/WeakPilot7B_run1/qwen2.5-7b-local/medagentbench-std/overall.json"
STRONG= f"{MAB}/outputs/MedAgentBenchv1/qwen2.5-72b-local/medagentbench-std/overall.json"
SPLITS= "/mnt/hdd/jonathan/STAIR/common/stair_common/splits/medagentbench"
SSCSV = "/mnt/hdd/jonathan/runs/mab/ss_costs_per_instance.csv"
DATA  = f"{MAB}/data/medagentbench/test_data_v2.json"
OUT   = "/mnt/hdd/jonathan/runs/mab/training3_table.json"
LAMBDA = 0.10
VIEW_ANSWER_CAP = 400   # chars of the weak attempt's final answer shown to the router

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def load(p):
    return {r["index"]: r for r in json.load(open(p))["custom"]["raw_results"]}

W, S = load(WEAK), load(STRONG)
data = {x["id"]: x for x in json.load(open(DATA))}

# splits: train = calib minus calib_sweep; val = calib_sweep; test = test
sweep = set(open(f"{SPLITS}/calib_sweep.txt").read().split())
calib = set(open(f"{SPLITS}/calib.txt").read().split())
test  = set(open(f"{SPLITS}/test.txt").read().split())
def split_of(tid):
    if tid in test: return "test"
    if tid in sweep: return "val"
    if tid in calib: return "train"
    raise KeyError(tid)

s_units = {}
for line in open(SSCSV):
    if line.startswith("tid"): continue
    tid, sp, ni, no, u = line.strip().split(",")
    s_units[tid] = float(u)

def weak_view_and_cost(rec):
    """Round-2 view + full and round-2 costs. The view NEVER includes anything
    past the weak model's second round, and never the grader's verdict."""
    h = rec["history"]
    n_in = n_out = 0; ctx = ""
    agent_msgs, cum = [], []
    for m in h:
        if m["role"] == "agent":
            n_in += len(tok.encode(ctx)); n_out += len(tok.encode(m["content"]))
            agent_msgs.append(m["content"]); cum.append(C.units(7.0, n_in, n_out))
        ctx += m["content"]
    rounds = len(agent_msgs)
    last = agent_msgs[-1] if agent_msgs else ""
    cw_full = cum[-1] if cum else 0.0
    cw2 = cum[min(2, rounds) - 1] if cum else 0.0
    if rounds <= 2:
        if last.startswith("FINISH"):
            summary = f"The weak model finished after {rounds} round(s). Its final answer: {last[:VIEW_ANSWER_CAP]}"
        else:
            summary = (f"The weak model did NOT finish: it stopped after {rounds} round(s) "
                       f"without calling FINISH. Its last action: {last[:VIEW_ANSWER_CAP]}")
    else:
        a2 = agent_msgs[1]
        summary = (f"The weak model is still working: after 2 of its rounds it has not "
                   f"called FINISH. Its most recent action: {a2[:VIEW_ANSWER_CAP]}")
    return summary, cw_full, cw2, rounds

rows = []
for idx in sorted(W):
    tid = f"task{idx//30+1}_{idx%30+1}"
    d = data[tid]
    view_task = d["instruction"] + (("\nContext: " + d["context"]) if d.get("context","").strip() else "")
    wsum, cw, cw2, rounds = weak_view_and_cost(W[idx])
    rows.append({
        "tid": tid, "idx": idx, "split": split_of(tid),
        "view": view_task + "\n\n" + wsum,
        "qw": 1 if W[idx]["status"] == "completedCorrect" else 0,
        "qs": 1 if S[idx]["status"] == "completedCorrect" else 0,
        "cw": round(cw, 2), "cw2": round(cw2, 2), "cs": s_units[tid], "weak_rounds": rounds,
    })
json.dump(rows, open(OUT, "w"), indent=1)
print(f"wrote {len(rows)} rows -> {OUT}")
cnt = collections.Counter(r["split"] for r in rows)
print("split sizes:", dict(cnt))

# rewards under the registered form (per-instance C_SS_i = cs; weak-first costs)
def R_deliver(r):  x = r["cw"] / r["cs"];               return r["qw"] - LAMBDA*x - max(0.0, x-1.0)
def R_escalate(r): x = (r["cw2"] + r["cs"]) / r["cs"];  return r["qs"] - LAMBDA*x - max(0.0, x-1.0)

for sp in ("train", "val", "test"):
    sub = [r for r in rows if r["split"] == sp]
    mW = st.mean(R_deliver(r) for r in sub); mS = st.mean(R_escalate(r) for r in sub)
    print(f"{sp:5s} n={len(sub):3d} | mean R' always-deliver {mW:+.4f} | always-escalate {mS:+.4f} | bar = {max(mW,mS):+.4f}")

# weak-first band: accuracy and cost of the oracle under weak-first accounting
acc = cost = ss_cost = 0
for r in rows:
    ss_cost += r["cw2"] + r["cs"]         # always-escalate at round 2 (early-escalation accounting)
    if r["qw"]: acc += 1; cost += r["cw"]                      # deliver wins or ties -> deliver (full weak)
    elif r["qs"]: acc += 1; cost += r["cw2"] + r["cs"]         # only escalation fixes it (cut at round 2)
    else: cost += r["cw"]                                      # hopeless -> deliver (cheapest)
n = len(rows)
print(f"\nWEAK-FIRST BAND (all 300): always-deliver {100*st.mean(r['qw'] for r in rows):.2f} | "
      f"always-escalate {100*st.mean(r['qs'] for r in rows):.2f} | "
      f"oracle {100*acc/n:.2f} at {100*cost/ss_cost:.1f}% of always-escalate cost")
