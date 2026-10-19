# configb_ablation_matrix.py -- the 4x4 matrix: retrieval-router size x
# answering-router size, composed on the VALIDATION split. CPU-only, read-only.
# Same grid, prices, cascade and threshold sweep as configb_pair_sweep.py and
# configb_ablation_readout.py; what varies per cell is WHICH score files pair,
# and each cell's overhead charges both routers at their own size.
import json

B = "/mnt/hdd/jonathan/runs/config_b"
GRID  = f"{B}/router_data_standard.jsonl"
PRICE = [80, 195, 894, 1009]   # WW, WS, SW, SS -- weak-first cascade
C_SS  = 929
OVH   = {"0.5B": 5.5, "1.5B": 16.5, "3B": 33.0, "7B": 77.0}   # units per router call
SIZES = ["0.5B", "1.5B", "3B", "7B"]

val = [g for g in map(json.loads, open(GRID)) if g["split"] == "val"]
n = len(val)
ss_acc = sum(g["em_standard"][3] for g in val) / n
ww_acc = sum(g["em_standard"][0] for g in val) / n
T = [i/20 for i in range(21)]
print(f"validation n={n} | always-SS {100*ss_acc:.2f} | always-WW {100*ww_acc:.2f}")

rets = {s: {r["idx"]: r["pW"] for r in map(json.loads, open(f"{B}/ablation_eval/val_scores_{s}.jsonl"))} for s in SIZES}
qas  = {s: {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(f"{B}/ablation_eval/val_scores_qa_{s}.jsonl"))} for s in SIZES}

results = {}
for R in SIZES:
    for Q in SIZES:
        ret, qa = rets[R], qas[Q]
        rows = [(ret[g["idx"]], qa[(g["idx"],"W")], qa[(g["idx"],"S")], g["em_standard"]) for g in val]
        res = []
        for tr in T:
            for tw in T:
                for ts in T:
                    acc = cost = 0
                    for pr, pqw, pqs, em in rows:
                        if pr >= tr:  c = 0 if pqw >= tw else 1
                        else:         c = 2 if pqs >= ts else 3
                        acc += em[c]; cost += PRICE[c]
                    res.append((acc/n, cost/n/C_SS))
        ovh = (OVH[R] + OVH[Q]) / C_SS
        best  = max(res, key=lambda r: (r[0], -r[1]))          # peak accuracy, ties -> cheapest
        beats = [r for r in res if r[0] >= ss_acc]
        cheap = min(beats, key=lambda r: r[1]) if beats else None
        results[(R, Q)] = (best, cheap, ovh)
        ctxt = f"{100*cheap[0]:.2f} @ {100*(cheap[1]+ovh):.1f}%" if cheap else "--"
        print(f"ret {R:>4} x qa {Q:>4} | peak {100*best[0]:.2f} @ {100*(best[1]+ovh):.1f}% | cheapest>=SS {ctxt}", flush=True)

print("\nMATRIX -- cost to reach always-SS accuracy, both routers' overhead included")
print("(rows = retrieval-router size, cols = answering-router size; -- = never reaches SS)")
print("        " + "".join(f"{q:>10}" for q in SIZES))
for R in SIZES:
    line = f"{R:>7} "
    for Q in SIZES:
        best, cheap, ovh = results[(R, Q)]
        line += f"{100*(cheap[1]+ovh):>9.1f}%" if cheap else f"{'--':>10}"
    print(line)
