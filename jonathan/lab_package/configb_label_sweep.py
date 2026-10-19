# configb_label_sweep.py -- threshold sweep over the SFT val scores (read-only).
# Grid: t_ret x t_qaW x t_qaS in 0.05 steps. Outcomes replayed from the grid
# file (standard ruler); costs = Table 5 path prices. Validation split only.
import json

GRID = "/mnt/hdd/jonathan/runs/config_b/router_data_standard.jsonl"
RET  = "/mnt/hdd/jonathan/runs/config_b/sft_router5_ret/val_scores.jsonl"
QA   = "/mnt/hdd/jonathan/runs/config_b/sft_router5_qa/val_scores.jsonl"
PRICE = [80, 195, 894, 1009]         # weak-first: count everything that ran
C_SS  = 929                          # budget = baseline SS without router (814+115)

ret = {r["idx"]: r["pW"] for r in map(json.loads, open(RET))}
qa  = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(QA))}
val = [g for g in map(json.loads, open(GRID)) if g["split"] == "val"]
rows = [(ret[g["idx"]], qa[(g["idx"],"W")], qa[(g["idx"],"S")], g["em_standard"]) for g in val]
n = len(rows)
ss_acc = sum(r[3][3] for r in rows) / n

T = [i/20 for i in range(21)]
results = []
for tr in T:
    for tw in T:
        for ts in T:
            acc = cost = 0
            for pr, pqw, pqs, em in rows:
                if pr >= tr:  c = 0 if pqw >= tw else 1   # weak retrieval
                else:         c = 2 if pqs >= ts else 3   # strong retrieval
                acc += em[c]; cost += PRICE[c]
            results.append((acc/n, cost/n/C_SS, tr, tw, ts))

best = max(results)
print(f"validation n={n} | always-SS {100*ss_acc:.2f} at 100%")
print(f"BEST ACCURACY: {100*best[0]:.2f} at {100*best[1]:.1f}% of SS cost  (t_ret={best[2]:.2f}, t_qaW={best[3]:.2f}, t_qaS={best[4]:.2f})")
beats = [r for r in results if r[0] >= ss_acc]
if beats:
    cheap = min(beats, key=lambda r: r[1])
    print(f"CHEAPEST >= SS: {100*cheap[0]:.2f} at {100*cheap[1]:.1f}%  (t_ret={cheap[2]:.2f}, t_qaW={cheap[3]:.2f}, t_qaS={cheap[4]:.2f})")
else:
    print("no swept point reaches always-SS accuracy on val")
print("frontier (best accuracy per cost band):")
for lo in range(0, 100, 10):
    band = [r for r in results if lo <= 100*r[1] < lo+10]
    if band:
        b = max(band)
        print(f"  {lo:3d}-{lo+10}% cost: {100*b[0]:.2f}  (t_ret={b[2]:.2f}, t_qaW={b[3]:.2f}, t_qaS={b[4]:.2f})")
