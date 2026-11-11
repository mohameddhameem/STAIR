#!/usr/bin/env python3
"""
grpo_fresh_random500_view.py -- SUBSET VIEW of the GRPO(Reward) label-free champion's existing held-out test read on the
registered size-matched sample (seed 20261015, n = 500 of the 1,481). Read-only, CPU, seconds. No new thresholds, no new
test exposure: the per-instance router probabilities were stored on 21 Oct (test_scores/) and the two points are the
frozen ones of grpo_fresh_test_readout.py (greedy 0.50/0.50/0.50; cap<=70% 0.80/0.05/0.05). Same precedent as the
21 Oct size-matched check of the mixed pair (registration, 21 Oct). Fills slide 128 row 2's GRPO cells.
Usage (lab):  python3 -I grpo_fresh_random500_view.py
"""
import json, random

B = "/mnt/hdd/jonathan/runs/config_b"
GRID, TS = f"{B}/router_data_standard.jsonl", f"{B}/test_scores"
PRICE, C_SS = [80, 195, 894, 1009], 929
POINTS = [("greedy (router's own choice)", 0.50, 0.50, 0.50), ("cap<=70% point (frozen 28 Oct)", 0.80, 0.05, 0.05)]

rows = [json.loads(l) for l in open(GRID) if l.strip()]
test = [g for g in rows if g["split"] == "test"]; assert len(test) == 1481
sample = random.Random(20261015).sample(test, 500)            # identical call to random500_seed20261015_write.py
idx = sorted(g["idx"] for g in sample)
try:
    on_disk = [int(x) for x in open(f"{B}/random500_seed20261015_idx.txt").read().split()]
    print("idx file on disk:", "MATCH" if on_disk == idx else "MISMATCH -- STOP")
    assert on_disk == idx
except FileNotFoundError:
    print("idx file not on disk; using the registered draw directly")


def load(fn):
    return {(r["idx"], r.get("ret_choice")): r["pW"] for r in map(json.loads, open(f"{TS}/{fn}.jsonl"))}


ret, qa = load("ret_grpo_fresh_objective"), load("qa_grpo_fresh_objective")
for label, R in (("random 500", sample), ("full 1,481", test)):
    ss = sum(g["em_standard"][3] for g in R) / len(R)
    print(f"{label}: always-SS {100*ss:.2f} at 100%")
    for pname, tr, tw, ts in POINTS:
        acc = cost = 0
        for g in R:
            i, em = g["idx"], g["em_standard"]
            if ret[(i, None)] >= tr: c = 0 if qa[(i, "W")] >= tw else 1
            else:                    c = 2 if qa[(i, "S")] >= ts else 3
            acc += em[c]; cost += PRICE[c]
        print(f"  GRPO(Reward) label-free, {pname}: {100*acc/len(R):.2f} at {100*(cost/len(R))/C_SS:.1f}%")
