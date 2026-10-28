#!/usr/bin/env python3
"""
training5_select_theta.py -- choose theta_H for Point B on VALIDATION (registered rule, v1.1 amendment):
the smallest theta on {0.3, 0.4, ..., 0.9} with 0 unnecessary hand-offs; ties -> fewest fails with no human.
Point B: on high-tier rows, hand over iff pHuman >= theta, else the greedy choice between W and S.
Read-only over <run-dir>/val_scores.jsonl and the committed Training 1 table; prints the grid and the choice.

Usage: python3 -I training5_select_theta.py --run-dir /mnt/hdd/jonathan/runs/mab/training5v13c_router_lt0 \
                                            --table /mnt/hdd/jonathan/runs/mab/training1_table.json
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--table", required=True)
a = ap.parse_args()
tab = {t["tid"]: t for t in json.load(open(a.table))}
rows = [json.loads(l) for l in open(os.path.join(a.run_dir, "val_scores.jsonl"))]
hi = [r for r in rows if r["high"]]
assert hi and all(r["pHuman"] is not None for r in hi)
print(f"{os.path.basename(a.run_dir)}: validation high-tier rows = {len(hi)}")
print(f"{'theta':>6} {'toPerson':>9} {'needed':>7} {'unnec':>6} {'failsNoHuman':>13}")
best = None
for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    H = need = unnec = fails = 0
    for r in hi:
        t = tab[r["tid"]]; both_fail = (t["qw"] == 0 and t["qs"] == 0)
        if r["pHuman"] >= th:
            H += 1; need += both_fail; unnec += (not both_fail)
        else:
            c = "W" if r["pDeliver"] >= r["pEscalate"] else "S"
            fails += 1 - (t["qw"] if c == "W" else t["qs"])
    print(f"{th:>6.1f} {H:>9d} {need:>7d} {unnec:>6d} {fails:>13d}")
    key = (unnec, fails, th)
    if best is None or key < best[0]:
        best = (key, th)
print(f"CHOSEN theta_H = {best[1]}  (rule: 0 unnecessary if attainable, then fewest fails with no human, then smallest theta)")
