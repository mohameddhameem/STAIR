# configb_label_readout.py -- quick label-only operating point (VALIDATION split).
# Read-only. Decisions at threshold 0.5 from the SFT val_scores files; outcomes
# replayed from the grid file (standard ruler); costs = Table 5 path prices.
import json

GRID = "/mnt/hdd/jonathan/runs/config_b/router_data_standard.jsonl"
RET  = "/mnt/hdd/jonathan/runs/config_b/sft_router5_ret/val_scores.jsonl"
QA   = "/mnt/hdd/jonathan/runs/config_b/sft_router5_qa/val_scores.jsonl"

PRICE = {("W","W"): 80, ("W","S"): 192, ("S","W"): 817, ("S","S"): 929}  # Table 5
CELL  = {("W","W"): 0,  ("W","S"): 1,  ("S","W"): 2,  ("S","S"): 3}     # em order
C_SS  = 929

ret = {r["idx"]: r["pW"] for r in map(json.loads, open(RET))}
qa  = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(QA))}
val = [g for g in map(json.loads, open(GRID)) if g["split"] == "val"]

n = acc = cost = ww = ss = 0
for g in val:
    i, em = g["idx"], g["em_standard"]
    assert i in ret and (i, "W") in qa and (i, "S") in qa, f"missing scores for idx {i}"
    rc = "W" if ret[i] >= 0.5 else "S"
    qc = "W" if qa[(i, rc)] >= 0.5 else "S"
    acc  += em[CELL[(rc, qc)]]
    cost += PRICE[(rc, qc)]
    ww += em[0]; ss += em[3]; n += 1

print(f"validation n={n}  (threshold 0.5, Table 5 prices)")
print(f"  always-WW : {100*ww/n:.2f} at {100*80/C_SS:.1f}% of SS cost")
print(f"  always-SS : {100*ss/n:.2f} at 100% of SS cost")
print(f"  label-routed (ret+qa): {100*acc/n:.2f} at {100*(cost/n)/C_SS:.1f}% of SS cost")
