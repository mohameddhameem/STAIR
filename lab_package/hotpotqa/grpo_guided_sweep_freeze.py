# grpo_guided_sweep_freeze.py -- F6G COPY (11 Nov 2026) of grpo_pair_sweep.py (the certified sweep): PAIRS re-pointed to the
# three-insights run (F6g, grpo_ret_qa_a1_fresh_ep6_guided) and, added at the end (# F6G), the <=70%-cap point (best validation
# accuracy at cost <= 70% of C_SS; ties -> cheapest) FROZEN to FROZEN_cap70.json before any test read (refuses to overwrite).
# Consistency: must reproduce the 25 Oct registration -- cap<=70% 75.77 @ 67.6%, cheapest >= SS 76.76 @ 99.1%. Validation only.
# grpo_pair_sweep.py -- threshold sweep for the GRPO (A1) routers (read-only).
# VERBATIM COPY of the certified configb_pair_sweep.py logic -- same grid, same
# PRICE family, same 21^3 sweep, same two printed points (best accuracy /
# cheapest >= SS) -- with ONLY the PAIRS table changed. The "both reward" row
# is reprinted so the GRPO rows read side by side against the certified pair.
# Costs: weak-first paths, budget = 929 (evaluation accounting, as certified).
import json, os

B = "/mnt/hdd/jonathan/runs/config_b"
GRID  = f"{B}/router_data_standard.jsonl"
PRICE = [80, 195, 894, 1009]   # WW, WS, SW, SS -- weak-first: count everything that ran
C_SS  = 929                    # budget = baseline strong-strong without the router

PAIRS = {"GRPO(Reward) + three insights (F6g), fresh":                                                   # F6G
           (f"{B}/grpo_ret_qa_a1_fresh_ep6_guided/ret_val_scores.jsonl", f"{B}/grpo_ret_qa_a1_fresh_ep6_guided/qa_val_scores.jsonl")}
val = [g for g in map(json.loads, open(GRID)) if g["split"] == "val"]
ss_acc = sum(g["em_standard"][3] for g in val) / len(val)
T = [i/20 for i in range(21)]
print(f"validation n={len(val)} | always-SS {100*ss_acc:.2f} at 100%")

for name, (rf, qf) in PAIRS.items():
    ret = {r["idx"]: r["pW"] for r in map(json.loads, open(rf))}
    qa  = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(qf))}
    rows = [(ret[g["idx"]], qa[(g["idx"],"W")], qa[(g["idx"],"S")], g["em_standard"]) for g in val]
    n = len(rows); res = []
    for tr in T:
        for tw in T:
            for ts in T:
                acc = cost = 0
                for pr, pqw, pqs, em in rows:
                    if pr >= tr:  c = 0 if pqw >= tw else 1
                    else:         c = 2 if pqs >= ts else 3
                    acc += em[c]; cost += PRICE[c]
                res.append((acc/n, cost/n/C_SS, tr, tw, ts))
    best = max(res)
    beats = [r for r in res if r[0] >= ss_acc]
    print(f"\n{name}:")
    print(f"  best accuracy : {100*best[0]:.2f} at {100*best[1]:.1f}%  (t={best[2]:.2f}/{best[3]:.2f}/{best[4]:.2f})")
    if beats:
        cheap = min(beats, key=lambda r: r[1])
        print(f"  cheapest >= SS: {100*cheap[0]:.2f} at {100*cheap[1]:.1f}%  (t={cheap[2]:.2f}/{cheap[3]:.2f}/{cheap[4]:.2f})")
    else:
        print("  no swept point reaches always-SS accuracy")
    # ---- F6G: freeze the <=70%-cap point (declared rule: best accuracy at cost <= 0.70; ties -> cheapest, then lowest t) ----
    cap = max((r for r in res if r[1] <= 0.70), key=lambda r: (r[0], -r[1], tuple(-x for x in r[2:])))
    print(f"  cap<=70% point: {100*cap[0]:.2f} at {100*cap[1]:.1f}%  (t={cap[2]:.2f}/{cap[3]:.2f}/{cap[4]:.2f})")
    fz = f"{B}/grpo_ret_qa_a1_fresh_ep6_guided/FROZEN_cap70.json"
    if os.path.exists(fz):
        print(f"  FROZEN file exists, not overwritten: {fz}")
    else:
        json.dump({"version": "grpo_guided_sweep_freeze v1.0 2026-11-11", "pair": name, "rule": "best val accuracy at cost<=70% of C_SS, ties cheapest",
                   "t_ret": cap[2], "t_qaW": cap[3], "t_qaS": cap[4], "val_acc": round(cap[0], 4), "val_cost_pct": round(100*cap[1], 1),
                   "greedy": (0.50, 0.50, 0.50)}, open(fz, "w"), indent=1)
        print(f"  FROZEN: {fz}")
