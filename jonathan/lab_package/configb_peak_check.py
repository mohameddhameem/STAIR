# configb_peak_check.py -- verifies WHY all answering-router sizes peak at the
# same composed accuracy (retrieval router fixed at 1.5B). Read-only, CPU-only.
# For each size: peak point + thresholds + path mix, vs the "escalate every
# answer" reference where the answering router is ignored.
import json

B = "/mnt/hdd/jonathan/runs/config_b"
PRICE = [80, 195, 894, 1009]
C_SS  = 929
SIZES = ["0.5B", "1.5B", "3B", "7B"]
T = [i/20 for i in range(21)]

val = [g for g in map(json.loads, open(f"{B}/router_data_standard.jsonl")) if g["split"] == "val"]
n = len(val)
ret = {r["idx"]: r["pW"] for r in map(json.loads, open(f"{B}/ablation_eval/val_scores_1.5B.jsonl"))}

def mix_at(rows, tr, tw, ts):
    cnt = [0, 0, 0, 0]; acc = cost = 0
    for pr, pqw, pqs, em in rows:
        if pr >= tr:  c = 0 if pqw >= tw else 1
        else:         c = 2 if pqs >= ts else 3
        cnt[c] += 1; acc += em[c]; cost += PRICE[c]
    return acc/n, cost/n/C_SS, cnt

for name in SIZES:
    qa = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(f"{B}/ablation_eval/val_scores_qa_{name}.jsonl"))}
    rows = [(ret[g["idx"]], qa[(g["idx"],"W")], qa[(g["idx"],"S")], g["em_standard"]) for g in val]
    # reference: answering router ignored -- every answer escalates to strong
    ref = max((sum(em[1] if pr >= tr else em[3] for pr, _, _, em in rows)/n,
               -sum(PRICE[1] if pr >= tr else PRICE[3] for pr, _, _, em in rows)/n/C_SS, tr) for tr in T)
    # full sweep
    res = []
    for tr in T:
        for tw in T:
            for ts in T:
                a, c, _ = 0, 0, None
                acc = cost = 0
                for pr, pqw, pqs, em in rows:
                    if pr >= tr:  cell = 0 if pqw >= tw else 1
                    else:         cell = 2 if pqs >= ts else 3
                    acc += em[cell]; cost += PRICE[cell]
                res.append((acc/n, cost/n/C_SS, tr, tw, ts))
    best = max(res, key=lambda r: (r[0], -r[1]))
    n_at_peak = sum(1 for r in res if abs(r[0] - best[0]) < 1e-12)
    pa, pc, pmix = mix_at(rows, best[2], best[3], best[4])
    print(f"\n{name}:")
    print(f"  escalate-all reference: {100*ref[0]:.2f} at {-100*ref[1]:.1f}% (tr={ref[2]:.2f})")
    print(f"  swept peak            : {100*best[0]:.2f} at {100*best[1]:.1f}%  thresholds {best[2]:.2f}/{best[3]:.2f}/{best[4]:.2f}")
    print(f"  points attaining peak : {n_at_peak} of 9261")
    print(f"  path mix at peak      : WW {pmix[0]}  WS {pmix[1]}  SW {pmix[2]}  SS {pmix[3]}")
