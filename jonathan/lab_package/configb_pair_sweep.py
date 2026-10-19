# configb_pair_sweep.py -- threshold sweep over all four router pairs (read-only).
# Pairs mirror Config A's frontier: label-only / reward answering / reward
# retrieval / both reward. Costs: weak-first paths, budget = 929.
import json

B = "/mnt/hdd/jonathan/runs/config_b"
GRID  = f"{B}/router_data_standard.jsonl"
PRICE = [80, 195, 894, 1009]   # WW, WS, SW, SS -- weak-first: count everything that ran
C_SS  = 929                    # budget = baseline strong-strong without the router

PAIRS = {
  "label only":       (f"{B}/sft_router5_ret/val_scores.jsonl",  f"{B}/sft_router5_qa/val_scores.jsonl"),
  "reward answering": (f"{B}/sft_router5_ret/val_scores.jsonl",  f"{B}/reward_train_qa/val_scores.jsonl"),
  "reward retrieval": (f"{B}/reward_train_ret/val_scores.jsonl", f"{B}/sft_router5_qa/val_scores.jsonl"),
  "both reward":      (f"{B}/reward_train_ret/val_scores.jsonl", f"{B}/reward_train_qa/val_scores.jsonl"),
}
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
