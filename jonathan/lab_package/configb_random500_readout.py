# configb_random500_readout.py -- a like-sized (n=500) clean sample from the SEALED
# TEST split (read-only, CPU). Purpose: a sample the same SIZE as AgentTTS's 500,
# drawn entirely outside our training data. Seed-locked: seed 20261015, registered
# 15 Oct. Thresholds frozen from validation (no tuning). Costs: weak-first paths,
# budget = 929.
import json, random

B = "/mnt/hdd/jonathan/runs/config_b"
PRICE = [80, 195, 894, 1009]; C_SS = 929
SEED, N = 20261015, 500

test = [g for g in map(json.loads, open(f"{B}/router_data_standard.jsonl")) if g["split"] == "test"]
sample = random.Random(SEED).sample(test, N)

def load(fn):
    return {(r["idx"], r.get("ret_choice")): r["pW"] for r in map(json.loads, open(f"{B}/test_scores/{fn}.jsonl"))}
ret_l, qa_l, qa_r = load("ret_label"), load("qa_label"), load("qa_reward")

RUNS = [
 ("label only, peak point",        ret_l, qa_l, 0.95, 1.00, 1.00),
 ("label only, parity point",      ret_l, qa_l, 0.80, 0.90, 1.00),
 ("reward answering, parity point", ret_l, qa_r, 0.80, 0.55, 0.60),
]
for name, rows in (("full test n=1,481", test), (f"random 500 (seed {SEED})", sample)):
    ss = sum(g["em_standard"][3] for g in rows) / len(rows)
    print(f"{name}: always-SS {100*ss:.2f} at 100%")
    for pname, ret, qa, tr, tw, ts in RUNS:
        acc = cost = 0
        for g in rows:
            i, em = g["idx"], g["em_standard"]
            if ret[(i, None)] >= tr: c = 0 if qa[(i, "W")] >= tw else 1
            else:                    c = 2 if qa[(i, "S")] >= ts else 3
            acc += em[c]; cost += PRICE[c]
        print(f"  {pname}: {100*acc/len(rows):.2f} at {100*(cost/len(rows))/C_SS:.1f}%")
